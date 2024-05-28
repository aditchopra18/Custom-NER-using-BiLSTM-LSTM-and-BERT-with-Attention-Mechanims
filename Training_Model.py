# Import the Relevant Libraries
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# Importing the relevant files
train_file = 'NCBItrainset_corpus.txt'
test_file = 'NCBItestset_corpus.txt'
output_file = 'Tagged_Test_File.txt'
model_name = 'NER_model.pth'

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
        tags = ['O'] * len(sentence) # Initially all tags are set at "O"
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

# Define Dataset class
class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_encoder, tag_encoder):
        self.sentences = sentences
        self.tags = tags
        self.word_encoder = word_encoder
        self.tag_encoder = tag_encoder

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]

        sentence_encoded = [self.word_encoder[word] for word in sentence]
        tags_encoded = self.tag_encoder.transform(tags)

        return torch.tensor(sentence_encoded), torch.tensor(tags_encoded, dtype=torch.long)

# Prepare data
all_words = [word for sentence in all_sentences for word in sentence]
all_tags_flat = [tag for tags in all_tags for tag in tags]

word_encoder = {word: idx for idx, word in enumerate(set(all_words))}
tag_encoder = LabelEncoder()
tag_encoder.fit(all_tags_flat)

dataset = NERDataset(all_sentences, all_tags, word_encoder, tag_encoder)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: x)

# Define Model for training
class NERModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=128):
        super(NERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.fc(lstm_out)
        return tag_space

# Training using PyTorch and "CUDA"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NERModel(len(word_encoder), len(tag_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

model.train()
for epoch in range(20):
    total_loss = 0
    for batch in dataloader:
        sentences, tags = zip(*batch)
        sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True).to(device)
        tags = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=-100).to(device)

        optimizer.zero_grad()
        outputs = model(sentences)
        outputs = outputs.view(-1, outputs.shape[-1])
        tags = tags.view(-1)
        loss = criterion(outputs, tags)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Saving the model
torch.save(model.state_dict(), model_name)

# Save annotation results
model.eval()
all_annotations = []

with torch.no_grad():
    for sentence in all_sentences:
        sentence_encoded = torch.tensor([word_encoder[word] for word in sentence], dtype=torch.long).unsqueeze(0).to(device)
        outputs = model(sentence_encoded)
        predictions = torch.argmax(outputs, dim=2).squeeze(0).cpu().numpy()
        predicted_tags = tag_encoder.inverse_transform(predictions)
        all_annotations.append((sentence, predicted_tags))

with open(output_file, 'w') as file:
    for sentence, tags in all_annotations:
        for word, tag in zip(sentence, tags):
            file.write(f'{word}\t{tag}\n')
        file.write('\n')