import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification

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

def tag_annotations(sentences, annotations):
    tagged_sentences = []

    for sentence in sentences:
        tags = ['O'] * len(sentence)  # Set tags at "O"
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

# Reading and parsing the dataset
train_file = 'Data/NCBItrainset_corpus.txt'
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

class NERDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, max_len):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        tags = self.tags[item]

        encoding = self.tokenizer(sentence, is_split_into_words=True, return_offsets_mapping=True, padding='max_length', truncation=True, max_length=self.max_len)
        word_ids = encoding.word_ids()

        labels = ['O'] * self.max_len
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id < len(tags):
                labels[idx] = tags[word_id]

        labels = [tag2id[label] for label in labels]

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# # Initialize tokenizer and other components
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# max_len = 128
# batch_size = 16

tag2id = {tag: idx for idx, tag in enumerate(set(tag for tags in all_tags for tag in tags))}
id2tag = {idx: tag for tag, idx in tag2id.items()}

# train_dataset = NERDataset(all_sentences, all_tags, tokenizer, max_len)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tag2id))

optimizer = optim.AdamW(model.parameters(), lr=3e-5)

# epochs = 3
# for epoch in range(epochs):
#     model.train()
#     for batch in train_dataloader:
#         optimizer.zero_grad()
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['labels']
        
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# # Save the model
# model_name = 'BERT_NER_model.pth'
# torch.save(model.state_dict(), model_name)