# Import the Relevant Libraries
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence

# Importing the relevant files
train_file = "Data/NCBItrainset_corpus.txt"
model = "bert-base-cased"
model_name = "Bert_NERModel.pth"

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

for paragraph in paragraphs:
    sentences, annotations = parse_paragraph(paragraph)
    tagged_sentences = tag_annotations(sentences, annotations)
    for sentence, tags in tagged_sentences:
        all_sentences.append(sentence)
        all_tags.append(tags)

# Now using the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(model)

def tokenized_sentences_and_labels(sentences, text_labels):
    labels = []
    tokenized_sentence = []

    for word, label in zip(sentences, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * len(tokenized_word))

    return tokenized_sentence, labels

# Converting the text in dataset to encoded IDs
tokenized_text_label = [tokenized_sentences_and_labels(sent, labs) for sent, labs in zip(all_sentences, all_tags)]

tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_text_label]

labels = [token_label_pair[1] for token_label_pair in tokenized_text_label]

input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts]

class NERDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

MAX_LEN = 128

input_ids = pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=0)
tags = pad_sequence([torch.tensor([tag2idx.get(l) for l in lab]) for lab in labels], batch_first=True, padding_value=-100)
attention_masks = (input_ids != 0).float()

dataset = NERDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Defining the model parameters
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(tag2idx))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tag2idx['O'])
optimizer = AdamW(model.parameters(), lr=3e-5)