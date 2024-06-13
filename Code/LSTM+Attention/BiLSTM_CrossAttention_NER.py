import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.utils import shuffle, class_weight
import seaborn as sns
import pandas as pd
import numpy as np
# from focal_loss import FocalLoss
from imblearn.over_sampling import ADASYN
from math import sqrt

# Function to read and parse the dataset file
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
                    if label == 'Modifier':
                        tags[i] = 'I-' + 'SpecificDisease'
                    else:
                        tags[i] = 'I-' + label
                elif word_start < start < word_end or word_start < end < word_end:
                    if label == 'Modifier':
                        tags[i] = 'I-' + 'SpecificDisease'
                    else:
                        tags[i] = 'I-' + label
        tagged_sentences.append((sentence, tags))
    return tagged_sentences

# Prepare the combined dataset
def prepare_combined_dataset(train_file, dev_file):
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
    for dev_paragraph in dev_paragraphs:
        dev_s, dev_a = parse_paragraph(dev_paragraph)
        dev_tagged_sentences = tag_annotations(dev_s, dev_a)
        for dev_sentence, dev_tag in dev_tagged_sentences:
            all_sentences.append(dev_sentence)
            all_tags.append(dev_tag)

    return all_sentences, all_tags

# Function to encode sentences and tags
def encode_data(sentences, tags, word_encoder, tag_encoder, unknown_token='<UNK>'):
    encoded_sentences = []
    encoded_tags = []
    for sentence, tag_sequence in zip(sentences, tags):
        encoded_sentence = [word_encoder.get(word, word_encoder[unknown_token]) for word in sentence]
        encoded_tag_sequence = tag_encoder.transform(tag_sequence)
        encoded_sentences.append(encoded_sentence)
        encoded_tags.append(encoded_tag_sequence)
    return encoded_sentences, encoded_tags

train_file = '../../Data/NCBItrainset_corpus.txt'
dev_file = '../../Data/NCBIdevelopset_corpus.txt'

all_sentences, all_tags = prepare_combined_dataset(train_file, dev_file)

# Create word and tag encoders
all_words = [word for sentence in all_sentences for word in sentence]
all_tags_flat = [tag for tags in all_tags for tag in tags]

word_encoder = {word: idx for idx, word in enumerate(set(all_words))}
unknown_token = '<UNK>'
word_encoder[unknown_token] = len(word_encoder)

tag_encoder = LabelEncoder()
tag_encoder.fit(all_tags_flat)

# Encode the complete dataset
encoded_all_sentences, encoded_all_tags = encode_data(all_sentences, all_tags, word_encoder, tag_encoder, unknown_token)

class LSTM_Attention_NERDataset(Dataset):
    def __init__(self, sentences, tags, word_encoder, tag_encoder, unknown_token='<UNK>'):
        self.sentences = sentences
        self.tags = tags
        self.word_encoder = word_encoder
        self.tag_encoder = tag_encoder
        self.unknown_token = unknown_token
        self.unk_count = 0
        self.unk_tags = []
        self.adasyn_resample()
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]
        sentence_encoded = [self.word_encoder.get(word, self.word_encoder[self.unknown_token]) for word in sentence]
        
        # Count <UNK> tokens and collect their tags
        for word, tag in zip(sentence, tags):
            if word not in self.word_encoder:
                self.unk_count += 1
                self.unk_tags.append(tag)
        
        tags_encoded = self.tag_encoder.transform(tags)
        return torch.tensor(sentence_encoded), torch.tensor(tags_encoded, dtype=torch.long)

    def adasyn_resample(self):
        flattened_sentences = [word for sentence in self.sentences for word in sentence]
        flattened_tags = [tag for tags in self.tags for tag in tags]
        encoded_sentences = np.array([self.word_encoder.get(word, self.word_encoder[self.unknown_token]) for word in flattened_sentences]).reshape(-1, 1)
        encoded_tags = np.array(self.tag_encoder.transform(flattened_tags))
        
        adasyn = ADASYN(sampling_strategy='minority')
        resampled_sentences, resampled_tags = adasyn.fit_resample(encoded_sentences, encoded_tags)
        
        # Split resampled data back into sentences
        sentence_lengths = [len(sentence) for sentence in self.sentences]
        new_sentences, new_tags = [], []
        
        start_idx = 0
        for length in sentence_lengths:
            end_idx = start_idx + length
            new_sentences.append(resampled_sentences[start_idx:end_idx].flatten().tolist())
            new_tags.append(resampled_tags[start_idx:end_idx].flatten().tolist())
            start_idx = end_idx
        
        self.sentences = [[list(self.word_encoder.keys())[idx] for idx in sentence] for sentence in new_sentences]
        self.tags = [[self.tag_encoder.inverse_transform([tag])[0] for tag in tags] for tags in new_tags]

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
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=128, dropout_prob=0.35):
        super(BiLSTM_CrossAttention_NER_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.batch_norm_emb = nn.BatchNorm1d(embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.cross_attention = CrossAttention(hidden_dim * 2)
        self.batch_norm_att = nn.BatchNorm1d(hidden_dim * 2)
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

class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-100):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)(outputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1-pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# Initialize the KFold Cross Validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Lists to hold training and validation losses for each fold
training_losses = []
validation_losses = []

# K-Fold Cross Validation training loop
for fold, (train_index, val_index) in enumerate(kf.split(encoded_all_sentences)):
    print(f"Fold {fold + 1}")
    
    train_sentences = [encoded_all_sentences[i] for i in train_index]
    train_tags = [encoded_all_tags[i] for i in train_index]
    val_sentences = [encoded_all_sentences[i] for i in val_index]
    val_tags = [encoded_all_tags[i] for i in val_index]
    
    train_dataset = LSTM_Attention_NERDataset(train_sentences, train_tags, word_encoder, tag_encoder, unknown_token)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)
    
    val_dataset = LSTM_Attention_NERDataset(val_sentences, val_tags, word_encoder, tag_encoder, unknown_token)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)
    
    # Model initialization
    model = BiLSTM_CrossAttention_NER_Model(len(word_encoder), len(tag_encoder.classes_)).to(device)
    criterion = Focal_Loss(alpha=0.95, gamma=0, ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.4)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training
    model.train()
    for epoch in range(80):
        total_loss = 0
        total_valid_loss = 0

        for batch in train_dataloader:
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
        training_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                val_sentences, val_tags = zip(*batch)
                val_sentences = torch.nn.utils.rnn.pad_sequence(val_sentences, batch_first=True).to(device)
                val_tags = torch.nn.utils.rnn.pad_sequence(val_tags, batch_first=True, padding_value=-100).to(device)

                with torch.cuda.amp.autocast():
                    val_outputs = model(val_sentences)
                    val_outputs = val_outputs.view(-1, val_outputs.shape[-1])
                    val_tags = val_tags.view(-1)
                    valid_loss = criterion(val_outputs, val_tags)
                    total_valid_loss += valid_loss.item()

        avg_valid_loss = total_valid_loss / len(val_dataloader)
        validation_losses.append(avg_valid_loss)
        
        scheduler.step(avg_train_loss)
        
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")

# Plot Training and Validation Loss for each fold
def plot_kfold_losses(training_losses, validation_losses, k):
    epochs = list(range(1, len(training_losses)//k + 1))
    for fold in range(k):
        plt.plot(epochs, training_losses[fold*len(epochs):(fold+1)*len(epochs)], label=f'Train Fold {fold + 1}')
        plt.plot(epochs, validation_losses[fold*len(epochs):(fold+1)*len(epochs)], label=f'Val Fold {fold + 1}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Each Fold')
    plt.legend()
    plt.savefig("../../Graphs/aBiLSTM_CrossAttention_KFold.png", bbox_inches='tight')
    plt.show()

plot_kfold_losses(training_losses, validation_losses, k=5)