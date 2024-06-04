import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Importing the relevant files
train_file = '../../Data/NCBItrainset_corpus.txt'
dev_file = '../../Data/NCBIdevelopset_corpus.txt'
model_name = '../../Models/LSTM_Attention_NER_model.pth'

# Reading the dataset file
def read_dataset(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return lines

def parse_dataset(lines):
    paragraphs = []
    paragraph = []

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

# Parsing the different paragraphs and annotations
def parse_paragraph(paragraph):
    sentences = []
    annotations = []
    sentence = []

    for line in paragraph:
        if re.match(r'^\d+\|\w\|', line):
            sentence.extend(line.split('|')[2].split())
        elif re.match(r'^\d+\t\d+\t\d+\t', line):
            start, end = int(line.split('\t')[1]), int(line.split('\t')[2])
            annotations.append((start, end, line.split('\t')[3], line.split('\t')[4]))

        if sentence:
            sentences.append(sentence)
            sentence = []

    if sentence:
        sentences.append(sentence)

    return sentences, annotations

# Data Labelling
def tag_annotations(sentences, annotations):
    tagged_sentences = []
    char_count = 0

    for sentence in sentences:
        tags = ['O'] * len(sentence)  # Initialize all tags at "O"
        word_starts = []
        word_ends = []
        char_pos = 0

        for word in sentence:
            word_starts.append(char_pos)
            char_pos += len(word)
            word_ends.append(char_pos)
            char_pos += 1  # WhiteSpace Character

        for start, end, disease_info, label in annotations:
            for i, (word_start, word_end) in enumerate(zip(word_starts, word_ends)):
                if word_start >= start and word_end <= end:
                    tags[i] = 'I-' + label
                elif word_start < start < word_end or word_start < end < word_end:
                    tags[i] = 'I-' + label

        tagged_sentences.append((sentence, tags))

    return tagged_sentences

# Parsing the dataset file
lines = read_dataset(train_file)
paragraphs = parse_dataset(lines)

all_sentences = []
all_tags = []

for paragraph in paragraphs:
    sentences, annotations = parse_paragraph(paragraph)
    tagged_sentences = tag_annotations(sentences, annotations)
    for sentence, tags in tagged_sentences:
        all_sentences.append(sentence)
        all_tags.append(tags)

# Parsing the development dataset file
dev_lines = read_dataset(dev_file)
dev_paragraphs = parse_dataset(dev_lines)

dev_sentences = []
dev_tags = []

for paragraph in dev_paragraphs:
    sentences, annotations = parse_paragraph(paragraph)
    tagged_sentences = tag_annotations(sentences, annotations)
    for sentence, tags in tagged_sentences:
        dev_sentences.append(sentence)
        dev_tags.append(tags)

# Define Dataset class
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

# Prepare data
all_words = [word for sentence in all_sentences for word in sentence]
all_tags_flat = [tag for tags in all_tags for tag in tags]

word_encoder = {word: idx for idx, word in enumerate(set(all_words))}
unknown_token = '<UNK>'
word_encoder[unknown_token] = len(word_encoder)  # Add unknown token

tag_encoder = LabelEncoder()
tag_encoder.fit(all_tags_flat)

dataset = LSTM_Attention_NERDataset(all_sentences, all_tags, word_encoder, tag_encoder, unknown_token)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)

# Defining the Global Attention class
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        weighted_output = lstm_output * attention_weights  # (batch_size, seq_len, hidden_dim)
        return weighted_output  # (batch_size, hidden_dim)

class Attention_LSTM_NER_Model(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=128):
        super(Attention_LSTM_NER_Model, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(hidden_dim, tagset_size)
        self.attention = Attention(hidden_dim)

    def forward(self, i):
        emb = self.embedding(i)
        lstm_out, _ = self.lstm(emb)
        att_out = self.attention(lstm_out)
        tag_space = self.fc(att_out)
        return tag_space

# Defining the model characteristics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Attention_LSTM_NER_Model(len(word_encoder), len(tag_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(model.parameters(), lr=0.005)

# Training the model
for epoch in range(500):
    model.train()
    total_loss = 0
    for batch in dataloader:
        sentence, tags = batch
        sentence, tags = sentence.to(device), tags.to(device)
        optimizer.zero_grad()
        output = model(sentence)
        loss = criterion(output, tags)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Testing the model
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for batch in dataloader:
        sentence, tags = batch
        sentence, tags = sentence.to(device), tags.to(device)
        output = model(sentence)
        loss = criterion(output, tags)
        test_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == tags).sum().item()

accuracy = correct / len(dev_tags)
print(f"Test Loss: {test_loss / len(dataloader)}")
print(f"Test Accuracy: {accuracy:.4f}")