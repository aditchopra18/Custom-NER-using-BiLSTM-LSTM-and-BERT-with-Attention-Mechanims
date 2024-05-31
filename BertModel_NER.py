import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
import pandas as pd

# Define important variables
train_file = 'Data/NCBItrainset_corpus.txt'
dev_file = 'Data/NCBIdevset_corpus.txt'
test_file = 'Data/NCBItestset_corpus.txt'
trained_model = "BERT_NER_model.pth"

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

def tag_annotations(sentences, annotations):
    tagged_sentences = []
    for sentence in sentences:
        tags = ['O'] * len(sentence)
        word_starts = []
        word_ends = []
        char_pos = 0
        for word in sentence:
            word_starts.append(char_pos)
            char_pos += len(word)
            word_ends.append(char_pos)
            char_pos += 1
        for start, end, ann_type, _ in annotations:
            for i, (w_start, w_end) in enumerate(zip(word_starts, word_ends)):
                if start <= w_start < end or start < w_end <= end:
                    tags[i] = 'I-' + ann_type
        tagged_sentences.append((sentence, tags))
    return tagged_sentences

def encode_tags(tags, tag2id):
    return [tag2id[tag] for tag in tags]

def create_tag_maps(tagged_sentences):
    tags = {tag for _, tags in tagged_sentences for tag in tags}
    tag2id = {tag: idx for idx, tag in enumerate(sorted(tags))}
    id2tag = {idx: tag for tag, idx in tag2id.items()}
    return tag2id, id2tag

# Debug function to print unique labels and tag2id mapping
def debug_labels_and_mapping(tag2id, labels):
    print("Tag2ID Mapping:", tag2id)
    unique_labels = set()
    for label_list in labels:
        unique_labels.update(label_list)
    print("Unique Labels in Dataset:", unique_labels)

class NERDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, max_len):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.sentences[idx], truncation=True, padding='max_length', max_length=self.max_len, is_split_into_words=True, return_tensors="pt")
        labels = encode_tags(self.tags[idx], tag2id)
        labels = labels + [-100] * (self.max_len - len(labels))
        tokens['labels'] = torch.tensor(labels, dtype=torch.long)
        return tokens

def collate_fn(batch):
    max_len = max([item['input_ids'].shape[1] for item in batch])
    
    input_ids = torch.stack([torch.nn.functional.pad(item['input_ids'].squeeze(), (0, max_len - item['input_ids'].shape[1]), value=-100) for item in batch])
    attention_mask = torch.stack([torch.nn.functional.pad(item['attention_mask'].squeeze(), (0, max_len - item['attention_mask'].shape[1]), value=-100) for item in batch])
    labels = torch.stack([torch.nn.functional.pad(item['labels'], (0, max_len - item['labels'].shape[0]), value=-100) for item in batch])

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Read and parse the dataset
train_lines = read_dataset(train_file)
train_paragraphs = parse_dataset(train_lines)

all_sentences = []
all_tags = []

for paragraph in train_paragraphs:
    sentences, annotations = parse_paragraph(paragraph)
    tagged_sentences = tag_annotations(sentences, annotations)
    for sentence, tags in tagged_sentences:
        all_sentences.append(sentence)
        all_tags.append(tags)

# Create tag mappings
tag2id, id2tag = create_tag_maps(list(zip(all_sentences, all_tags)))

# Initialize tokenizer and other components
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
max_len = 512
batch_size = 16

train_dataset = NERDataset(all_sentences, all_tags, tokenizer, max_len)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Debugging: Check the collated batch
batch = next(iter(train_dataloader))
collated_batch = collate_fn([batch])

print("\nCollated Batch:\n")
print(collated_batch)

# Check for invalid token IDs
def check_token_ids(batch, vocab_size):
    invalid_ids = (batch['input_ids'] >= vocab_size).nonzero(as_tuple=True)
    if len(invalid_ids[0]) > 0:
        print(f"Invalid token IDs found: {batch['input_ids'][invalid_ids]}")
    else:
        print("All token IDs are valid.")

# Check token IDs in the collated batch
vocab_size = tokenizer.vocab_size
check_token_ids(collated_batch, vocab_size)

# Initialize the model
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(tag2id))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(model.parameters(), lr=3e-5)

# Check label ranges in the collated batch
def check_label_ranges(batch, num_labels):
    invalid_labels = (batch['labels'] >= num_labels).nonzero(as_tuple=True)
    if len(invalid_labels[0]) > 0:
        print(f"Invalid label IDs found: {batch['labels'][invalid_labels]}")
    else:
        print("All label IDs are valid.")

check_label_ranges(collated_batch, len(tag2id))

# Training the model
model.train()
for epoch in range(20):
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].squeeze(0).to(device)
        attention_mask = batch['attention_mask'].squeeze(0).to(device)
        labels = batch['labels'].squeeze(0).to(device)

        # Debug: Check if any label is out of the range of tag2id
        invalid_label_indices = (labels >= len(tag2id)).nonzero(as_tuple=True)
        if len(invalid_label_indices[0]) > 0:
            print(f"Invalid label indices found in batch: {labels[invalid_label_indices]}")
            continue

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

# Save the model as a .pth file
torch.save(model.state_dict(), trained_model)