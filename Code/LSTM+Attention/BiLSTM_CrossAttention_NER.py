import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from collections import Counter
# from sklearn.model_selection import KFold

# Importing the relevant files
train_file = 'NCBItrainset_corpus.txt'
dev_file = 'NCBIdevelopset_corpus.txt'
model_name = '../../Models/BiLSTM_CrossAttention_NER_model.pth'
unknown_token = "<UNK>"

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

# Prepare the training and validation data
lines = read_dataset(train_file)
paragraphs = parse_dataset(lines)
all_sentences, all_tags = [], []
for paragraph in paragraphs:
    s, a = parse_paragraph(paragraph)
    tagged_sentences = tag_annotations(s, a)
    for sentence, tag in tagged_sentences:
        all_sentences.append(sentence)
        all_tags.append(tag)

dev_lines = read_dataset(dev_file)
dev_paragraphs = parse_dataset(dev_lines)
dev_all_sentences, dev_all_tags = [], []
for dev_paragraph in dev_paragraphs:
    dev_s, dev_a = parse_paragraph(dev_paragraph)
    dev_tagged_sentences = tag_annotations(dev_s, dev_a)
    for dev_sentence, dev_tag in dev_tagged_sentences:
        dev_all_sentences.append(dev_sentence)
        dev_all_tags.append(dev_tag)

class LSTM_Attention_NERDataset(Dataset):
    def __init__(self, sentences, tags, word_encoder, tag_encoder, unknown_token='<UNK>'):
        self.sentences = sentences
        self.tags = tags
        self.word_encoder = word_encoder
        self.tag_encoder = tag_encoder
        self.unknown_token = unknown_token

        self.unknown_token_count = 0  # Counter for unknown tokens
        self.unknown_tokens = []  # List to store unknown words and their tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]
        word_indices = []
        unknown_tokens_in_sentence = []
        for word, tag in zip(sentence, tags):
            if word in self.word_encoder.classes_:
                word_indices.append(self.word_encoder.transform([word])[0])
            else:
                word_indices.append(self.word_encoder.transform([self.unknown_token])[0])
                self.unknown_token_count += 1  # Increment counter for unknown tokens
                unknown_tokens_in_sentence.append((word, tag))
        self.unknown_tokens.extend(unknown_tokens_in_sentence)  # Add unknown tokens to the list
        tag_indices = self.tag_encoder.transform(tags)
        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)

all_words = [word for sentence in all_sentences for word in sentence] + [unknown_token]
all_tags_flat = [tag for tags in all_tags for tag in tags]

word_encoder = LabelEncoder()
word_encoder.fit(all_words)

tag_encoder = LabelEncoder()
tag_encoder.fit(all_tags_flat)

dataset = LSTM_Attention_NERDataset(all_sentences, all_tags, word_encoder, tag_encoder, unknown_token)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)

dev_dataset = LSTM_Attention_NERDataset(dev_all_sentences, dev_all_tags, word_encoder, tag_encoder, unknown_token)
dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)

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
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=128, dropout_prob=0.3):
        super(BiLSTM_CrossAttention_NER_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.batch_norm_emb = nn.BatchNorm1d(embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout_prob)
        self.cross_attention = CrossAttention(hidden_dim * 2)
        self.batch_norm_att = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.batch_norm_emb(emb.transpose(1, 2)).transpose(1, 2)
        emb = self.dropout(emb)
        
        bilstm_out, _ = self.bilstm(emb)
        bilstm_out = self.dropout(bilstm_out)
        
        cross_att_out = self.cross_attention(bilstm_out, bilstm_out, bilstm_out)
        cross_att_out = self.batch_norm_att(cross_att_out.transpose(1, 2)).transpose(1, 2)
        cross_att_out = self.dropout(cross_att_out)
        
        tag_space = self.fc(cross_att_out)
        return tag_space

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# all_sentences = list(all_sentences)
# all_tags = list(all_tags)

# Function for plotting (to be used to visualize the training loss and validation loss)
# Used to figure if the model is underfitting or overfitting
def graph_plot(title, x_label, y_label, x_data, y_data, z_data, color = 'blue', linestyle = '-'):
    plt.plot(x_data, y_data, color = color, linestyle = linestyle)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x_data, z_data, color = 'r', linestyle = '-')
    plt.legend()
    plt.savefig("../../Graphs/aBiLSTM_CrossAttention.png", bbox_inches = 'tight')


train_dataset = LSTM_Attention_NERDataset(all_sentences, all_tags, word_encoder, tag_encoder, unknown_token)
val_dataset = LSTM_Attention_NERDataset(dev_all_sentences, dev_all_tags, word_encoder, tag_encoder, unknown_token)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: x)
print(train_dataloader)
model = BiLSTM_CrossAttention_NER_Model(len(word_encoder.classes_), len(tag_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-100) 
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
scaler = torch.cuda.amp.GradScaler()

model.train()
loss_dic = {}
valid_loss_dic = {}

print("Starting Training\n")
for epoch in range(40):
    total_loss = 0
    total_valid_loss = 0

    print (f'Epoch: {epoch + 1}')
    for batch_idx, batch in enumerate(train_dataloader):
        # Debugging print statement
        print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_dataloader)}")

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

    avg_train_loss = total_loss / len(train_dataloader)

    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(val_dataloader):
            # Debugging print statement
            print(f"Validation Batch {val_batch_idx + 1}/{len(val_dataloader)}")

            val_sentences, val_tags = zip(*val_batch)
            val_sentences = torch.nn.utils.rnn.pad_sequence(val_sentences, batch_first=True).to(device)
            val_tags = torch.nn.utils.rnn.pad_sequence(val_tags, batch_first=True, padding_value=-100).to(device)

            with torch.cuda.amp.autocast():
                val_outputs = model(val_sentences)
                val_outputs = val_outputs.view(-1, val_outputs.shape[-1])
                val_tags = val_tags.view(-1)
                valid_loss = criterion(val_outputs, val_tags)
                total_valid_loss += valid_loss.item()
        
    avg_valid_loss = total_valid_loss / len(val_dataloader)

    scheduler.step(avg_train_loss)

    print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")
    loss_dic[epoch] = avg_train_loss
    valid_loss_dic[epoch] = avg_valid_loss

    torch.save(model.state_dict(), model_name)