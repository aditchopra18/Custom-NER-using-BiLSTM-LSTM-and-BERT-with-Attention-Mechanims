# Import the Relevant Libraries
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Importing the relevant files
train_file = 'Data/NCBItrainset_corpus.txt'
model_name = 'Models/BERT_NER_model.pth'

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

def parse_paragraph(paragraph):
    sentences = []
    annotations = []
    sentence = []

    for line in paragraph:
        if re.match(r'^\d+\|\w\|', line):
            sentence.extend(line.split('|')[2].split())

        elif re.match(r'^\d+\t\d+\t\d+\t', line):
            start, end = int(line.split("\t")[1]), int(line.split("\t")[2])
            annotations.append((start, end, line.split("\t")[3], line.split("\t")[4]))

    if sentence:
        sentences.append(sentence)
    return sentences, annotations

# Data Labelling
def tag_annotations(sentences, annotations):
    tagged_sentences = []
    char_count = 0

    for sentence in sentences:
        tags = ['O'] * len(sentence)    # Initialize all tags at "O"
        word_starts = []
        word_ends = []
        char_pos = 0

        for word in sentence:
            word_starts.append(char_pos)
            char_pos += len(word)
            word_ends.append(char_pos)
            char_pos += 1               # WhiteSpace Character

        '''
        Based on the character limits, the annotations are assigned
        A custom IO tagging scheme is used
        Labels are assigned on the basis of disease label in annotations
        '''

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

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

# Define Dataset class
class NERDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, tag_encoder):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.tag_encoder = tag_encoder

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]

        encoding = self.tokenizer(sentence, is_split_into_words=True, return_offsets_mapping=True, padding='max_length', truncation=True, return_tensors="pt")
        labels = [-100] * len(encoding['input_ids'][0])

        word_ids = encoding.word_ids()
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                labels[i] = -100
            else:
                labels[i] = self.tag_encoder.transform([tags[word_idx]])[0]

        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(), torch.tensor(labels)

# Prepare data
all_tags_flat = [tag for tags in all_tags for tag in tags]

tag_encoder = LabelEncoder()
tag_encoder.fit(all_tags_flat)

# Calculate class weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = tag_encoder.classes_
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=all_tags_flat)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

dataset = NERDataset(all_sentences, all_tags, tokenizer, tag_encoder)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define BERT-based NER Model with weighted loss
class BertNERModel(nn.Module):
    def __init__(self, model_name, num_labels, class_weights):
        super(BertNERModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(0.3)  # Add dropout layer
        self.class_weights = class_weights

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = self.dropout(output.logits)  # Apply dropout
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.bert.config.num_labels), labels.view(-1))
            return loss, logits
        return logits

# Defining the model characteristics
print(device)

model = BertNERModel('bert-base-uncased', len(tag_encoder.classes_), class_weights).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Training using PyTorch, AdamW Optimizer, CrossEntropyLoss function and "CUDA"
model.train()
num_epochs = 10
max_grad_norm = 1.0  # Gradient clipping

for epoch in range(num_epochs):
    total_loss = 0
    print(f"Starting Epoch {epoch + 1}")
    for batch_idx, batch in enumerate(dataloader):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        optimizer.zero_grad()
        loss, outputs = model(input_ids, attention_mask, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Clip gradients
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")

    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)  # Adjust learning rate based on average loss
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
print("Finished Training")

# Saving the model
torch.save(model.state_dict(), model_name)

# Testing the model on the testing dataset
# Load the test dataset
test_file = 'Data/NCBItestset_corpus.txt'
test_lines = read_dataset(test_file)
test_paragraphs = parse_dataset(test_lines)

test_sentences = []
test_tags = []

for paragraph in test_paragraphs:
    sentences, annotations = parse_paragraph(paragraph)
    tagged_sentences = tag_annotations(sentences, annotations)
    for sentence, tags in tagged_sentences:
        test_sentences.append(sentence)
        test_tags.append(tags)

# Prepare test data
test_dataset = NERDataset(test_sentences, test_tags, tokenizer, tag_encoder)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the saved model
model.load_state_dict(torch.load(model_name))
model.eval()

all_true_labels = []
all_pred_labels = []

# Evaluate the model
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        outputs = model(input_ids, attention_mask)
        active_logits = outputs.view(-1, model.bert.config.num_labels)
        active_labels = labels.view(-1)
        active_predictions = torch.argmax(active_logits, axis=1)

        active_indices = active_labels != -100

        all_true_labels.extend(active_labels[active_indices].cpu().numpy())
        all_pred_labels.extend(active_predictions[active_indices].cpu().numpy())

# Decode labels
true_labels = tag_encoder.inverse_transform(all_true_labels)
pred_labels = tag_encoder.inverse_transform(all_pred_labels)

# Print the classification report
print(classification_report(true_labels, pred_labels))