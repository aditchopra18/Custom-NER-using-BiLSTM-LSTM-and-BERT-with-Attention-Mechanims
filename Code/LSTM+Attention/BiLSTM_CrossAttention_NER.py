import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.utils import class_weight

# Importing the relevant files
train_file = '../../Data/NCBItrainset_corpus.txt'
model_name = '../../Models/BiLSTM_CrossAttention_NER_model.pth'

# Reading and parsing the dataset file
def read_dataset(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return lines

def parse_dataset(lines):
    paragraphs, paragraph = [], []
    for line in lines:
        line = line.strip()
        if line:
            paragraph.append(line)
        else:
            if paragraph:
                paragraphs.append(paragraph)
                paragraph = []
    if paragraph:
        paragraphs.append(paragraph)
    return paragraphs

def parse_paragraph(paragraph):
    sentences, annotations, sentence = [], [], []
    for line in paragraph:
        if re.match(r'^\d+\|\w\|', line):
            sentence.extend(line.split('|')[2].split())
        elif re.match(r'^\d+\t\d+\t\d+\t', line):
            start, end = int(line.split("\t")[1]), int(line.split("\t")[2])
            annotations.append((start, end, line.split("\t")[3], line.split("\t")[4]))
    if sentence:
        sentences.append(sentence)
    return sentences, annotations

def tag_annotations(sentences, annotations):
    tagged_sentences = []
    for sentence in sentences:
        tags = ['O'] * len(sentence)
        word_starts, word_ends, char_pos = [], [], 0
        for word in sentence:
            word_starts.append(char_pos)
            char_pos += len(word)
            word_ends.append(char_pos)
            char_pos += 1
        for start, end, _, label in annotations:
            for i, (word_start, word_end) in enumerate(zip(word_starts, word_ends)):
                if word_start >= start and word_end <= end:
                    tags[i] = 'I-' + label
                elif word_start < start < word_end or word_start < end < word_end:
                    tags[i] = 'I-' + label
        tagged_sentences.append((sentence, tags))
    return tagged_sentences

# Prepare the data
lines = read_dataset(train_file)
paragraphs = parse_dataset(lines)
all_sentences, all_tags = [], []
for paragraph in paragraphs:
    s, a = parse_paragraph(paragraph)
    tagged_sentences = tag_annotations(s, a)
    for sentence, tag in tagged_sentences:
        all_sentences.append(sentence)
        all_tags.append(tag)

class LSTM_Attention_NERDataset(Dataset):
    def __init__(self, sentences, tags, word_encoder, tag_encoder, unknown_token='<UNK>'):
        self.sentences = sentences
        self.tags = tags
        self.word_encoder = word_encoder
        self.tag_encoder = tag_encoder
        self.unknown_token = unknown_token

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]
        sentence_encoded = [self.word_encoder.get(word, self.word_encoder[self.unknown_token]) for word in sentence]
        tags_encoded = self.tag_encoder.transform(tags)
        return torch.tensor(sentence_encoded), torch.tensor(tags_encoded, dtype=torch.long)

all_words = [word for sentence in all_sentences for word in sentence]
all_tags_flat = [tag for tags in all_tags for tag in tags]

word_encoder = {word: idx for idx, word in enumerate(set(all_words))}
unknown_token = '<UNK>'
word_encoder[unknown_token] = len(word_encoder)

tag_encoder = LabelEncoder()
tag_encoder.fit(all_tags_flat)

# Store sentences and tags in DataFrame
df = pd.DataFrame({'sentence': all_sentences, 'tags': all_tags})

dataset = LSTM_Attention_NERDataset(all_sentences, all_tags, word_encoder, tag_encoder, unknown_token)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        query_proj = self.query(query)
        key_proj = self.key(key)
        value_proj = self.value(value)
        attention_scores = torch.matmul(query_proj, key_proj.transpose(-2, -1))
        attention_weights = self.softmax(attention_scores)
        context = torch.matmul(attention_weights, value_proj)
        return context

class BiLSTM_CrossAttention_NER_Model(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=128):
        super(BiLSTM_CrossAttention_NER_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.cross_attention = CrossAttention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        emb = self.embedding(x)
        bilstm_out, _ = self.bilstm(emb)
        cross_att_out = self.cross_attention(bilstm_out, bilstm_out, bilstm_out)
        tag_space = self.fc(cross_att_out)
        return tag_space

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Flatten tags for computing class weights
flattened_tags = [tag for sublist in df['tags'] for tag in sublist]
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(flattened_tags), y=flattened_tags)
class_weights_dict = {tag: weight for tag, weight in zip(np.unique(flattened_tags), class_weights)}
class_weights_tensor = torch.tensor([class_weights_dict[tag] for tag in tag_encoder.classes_], dtype=torch.float).to(device)

model = BiLSTM_CrossAttention_NER_Model(len(word_encoder), len(tag_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, ignore_index=-100)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()

# Training the Model
for epoch in range(40):
    model.train()
    total_loss = 0
    for batch in dataloader:
        sentences, tags = zip(*batch)
        sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True).to(device)
        tags = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=-100).to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(sentences)
            outputs = outputs.view(-1, outputs.shape[-1])
            tags = tags.view(-1)
            loss = criterion(outputs, tags)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), model_name)

# Testing the model, and evaluating the f1 score
test_file = 'NCBItestset_corpus.txt'
test_lines = read_dataset(test_file)
test_paragraphs = parse_dataset(test_lines)

# Parsing and storing the test dataset
test_sentences = []
test_tags = []

for paragraph in test_paragraphs:
    sentences, annotations = parse_paragraph(paragraph)
    tagged_sentences = tag_annotations(sentences, annotations)
    for sentence, tags in tagged_sentences:
        test_sentences.append(sentence)
        test_tags.append(tags)

# Importing the model file
model = BiLSTM_CrossAttention_NER_Model(len(word_encoder), len(tag_encoder.classes_)).to(device)
model.load_state_dict(torch.load(model_name))
model.eval()

# Prepare the test data
test_dataset = LSTM_Attention_NERDataset(test_sentences, test_tags, word_encoder, tag_encoder, '<UNK>')
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x)

# Evaluate the model
all_true_labels = []
all_pred_labels = []

result = "../../Result/TestResults_CrossAttention_BiLSTM_NER.txt"
with open(result, 'w') as t_file:
    with torch.no_grad():
        for batch in test_dataloader:
            sentences, tags = zip(*batch)
            sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True).to(device)
            tags = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=-100).to(device)

            outputs = model(sentences)
            outputs = outputs.view(-1, outputs.shape[-1])
            tags = tags.view(-1)

            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels = tags.cpu().numpy()

            mask = true_labels != -100
            pred_labels = predictions[mask]
            true_labels = true_labels[mask]

            pred_labels_decoded = tag_encoder.inverse_transform(pred_labels)
            true_labels_decoded = tag_encoder.inverse_transform(true_labels)

            for true_label, pred_label in zip(true_labels_decoded, pred_labels_decoded):
                t_file.write(f'True: {true_label}, Pred: {pred_label}\n')
                all_true_labels.append(true_label)
                all_pred_labels.append(pred_label)

# Printing classification report
report = classification_report(all_true_labels, all_pred_labels)
print(report)
