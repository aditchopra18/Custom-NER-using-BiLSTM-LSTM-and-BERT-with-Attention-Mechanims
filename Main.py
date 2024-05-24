import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import transformers

# Training Dataset
train_file = 'parsing.txt'  # Adjust path if necessary

# Parsing the file
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
    text = ""
    
    for line in paragraph:
        if re.match(r'^\d+\|\w\|', line):
            if sentence:
                sentences.append(sentence)
                sentence = []
            text += line.split('|')[2] + ' '  # Concatenate lines with a space
            sentence.extend(line.split('|')[2].split())
        elif re.match(r'^\d+\t\d+\t\d+\t', line):
            start, end = int(line.split("\t")[1]), int(line.split("\t")[2])
            text_segment = text[start:end]
            annotations.append((start, end, line.split("\t")[3], line.split("\t")[4]))

    if sentence:
        sentences.append(sentence)

    return sentences, annotations

def tag_annotations(sentences, annotations):
    tagged_sentences = []
    char_count = 0
    
    for sentence in sentences:
        tags = ['O'] * len(sentence)  # Start with all words tagged as 'O'
        
        for i, word in enumerate(sentence):
            word_start = char_count
            char_count += len(word) + 1  # +1 for the space between words
            word_end = char_count - 1

            for annotation in annotations:
                start, end, _, label = annotation
                if word_start >= start and word_end <= end:
                    tags[i] = 'I-' + label
        
        tagged_sentences.append((sentence, tags))
        
    return tagged_sentences

lines = read_dataset(train_file)
paragraphs = parse_dataset(lines)

all_tagged_sentences = []

for paragraph in paragraphs:
    sentences, annotations = parse_paragraph(paragraph)
    tagged_sentences = tag_annotations(sentences, annotations)
    all_tagged_sentences.extend(tagged_sentences)

# Save the tagged sentences to a new file
output_file = 'Tagged_File.txt'
with open(output_file, 'w') as file:
    for sentence, tags in all_tagged_sentences:
        for word, tag in zip(sentence, tags):
            file.write(f'{word}\t{tag}\n')
        file.write('\n')