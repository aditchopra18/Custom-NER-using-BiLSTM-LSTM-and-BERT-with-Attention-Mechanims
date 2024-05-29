# Import the Relevant Libraries
import re
from torch import cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast, BertForTokenClassification, AdamW
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence

# Importing the relevant files
train_file = "Data/NCBItrainset_corpus.txt"
model_name = "bert-base-cased"
ann_file = "bert_train_file.txt"

device = 'cuda' if cuda.is_available() else 'cpu'

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

    for sentence in sentences:
        tags = ['O'] * len(sentence)  # Initially all tags are set at "O"
        word_starts = []
        word_ends = []
        char_pos = 0

        for word in sentence:
            word_starts.append(char_pos)
            char_pos += len(word)
            word_ends.append(char_pos)
            char_pos += 1  # WhiteSpace Character

        # Based on the character limits, change the annotations
        # A custom IO tagging scheme is used
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

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
token_sentences = []
token_sentence = []

for paragraph in paragraphs:
    sentences, annotations = parse_paragraph(paragraph)
    tagged_sentences = tag_annotations(sentences, annotations)
    for sentence, tags in tagged_sentences:
        all_sentences.append(sentence)
        all_tags.append(tags)

for sentence in all_sentences:
    for word in sentence:
        token_word = tokenizer.tokenize(word)
        token_sentence.append(token_word)
    token_sentences.extend(token_sentence)
    token_sentence = []

with open (ann_file, "w") as bert_file:
    for i in token_sentences:
        for j in i:
            bert_file.write(j)
        bert_file.write(f'\n')

# # Custom Dataset Class
# class NERDataset(Dataset):
#     def get_tag2id(self):
#         unique_tags = set(tag for tag_seq in self.tags for tag in tag_seq)
#         tag2id = {tag: idx for idx, tag in enumerate(unique_tags)}
#         return tag2id
        
#     def __init__(self, sentences, tags, tokenizer, max_len):
#         self.sentences = sentences
#         self.tags = tags
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.tag2id = self.get_tag2id()

#     def __len__(self):
#         return len(self.sentences)

#     def __getitem__(self, idx):
#         sentence = self.sentences[idx]
#         tags = self.tags[idx]
        
#         encoding = self.tokenizer(sentence, is_split_into_words=True,
#                                   truncation=True,
#                                   padding='max_length',
#                                   max_length=self.max_len,
#                                   return_tensors='pt')
        
#         labels = [self.tag2id[tag] for tag in tags]
#         labels = labels + [self.tag2id['O']] * (self.max_len - len(labels))
        
#         item = {key: val.squeeze() for key, val in encoding.items()}
#         item['labels'] = torch.tensor(labels, dtype=torch.long)
        
#         return item

# # Custom Collate Function
# def collate_fn(batch):
#     input_ids = [item['input_ids'] for item in batch]
#     attention_mask = [item['attention_mask'] for item in batch]
#     labels = [item['labels'] for item in batch]

#     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
#     attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
#     labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Ignore index for padding

#     max_len = input_ids.size(1)
#     if labels.size(1) != max_len:
#         padded_labels = torch.full((labels.size(0), max_len), -100, dtype=torch.long)
#         padded_labels[:, :labels.size(1)] = labels
#         labels = padded_labels

#     return {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'labels': labels
#     }

# # Parameters
# MAX_LEN = 128
# BATCH_SIZE = 16

# dataset = NERDataset(all_sentences, all_tags, tokenizer, MAX_LEN)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# # Model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(dataset.tag2id))

# # Optimizer
# optimizer = AdamW(model.parameters(), lr=5e-5)

# EPOCHS = 20
# model.train()
# for epoch in range(EPOCHS):
#     total_loss = 0
#     for batch in dataloader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)

#         print(f"input_ids shape: {input_ids.shape}")
#         print(f"labels shape: {labels.shape}")

#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         total_loss += loss.item()

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#     avg_loss = total_loss / len(dataloader)
#     print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss}")