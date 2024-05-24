# Import Relevant Libraries
import os
import re
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import dataset, dataloader
import transformers

# Training Dataset
train_file = 'parsing.txt'

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

    for line in paragraph:
        if re.match(r'^\d+\|\w\|', line):
            if sentence:
                sentences.append(sentence)
                sentence = []
            sentence.extend(line.split('|')[2].split())
        
        elif re.match(r'^\d+\t\d+\t\d+\t', line):
            annotations.append(line.split("\t"))
    
    if sentence:
        sentences.append(sentence)
    return sentences, annotations

lines = read_dataset(train_file)
paragraphs = parse_dataset(lines)
sentences, annotations = parse_paragraph(paragraphs)

# Debug print
# for i in sentences:
#     print(i)
# for j in annotations:
#     print (j)

def tag_annotations(sentences, annotations):
    tagged_sentences = []
    for sentence in sentences:
        tags = ['O'] * len(sentence)
        for annotation in annotations:
            descript_ID, start, end, disease, disease_label, disease_ID = annotation
            start = int(start)
            end = int(end)
    # Creating tag file based on character limits in the dataset file
    # using the IO tagging scheme
            count_char = 0
            # Correctly assigning the entites with custom tags
            for i, words in enumerate(sentence):
                word_start = count_char
                count_char += len(words) + 1
                word_end = count_char - 1
                if word_start > start and word_end <= end:
                    tags[i] = "I-" + disease_label
        tagged_sentences.append((sentence, tags))
    
    return tagged_sentences

all_tagged_sentences = []

for paragraph in paragraphs:
    sentences, annotations = parse_paragraph(paragraph)
    tagged_sentences = tag_annotations(sentences, annotations)
    all_tagged_sentences.extend(tagged_sentences)

# Saving the tagged sentences in a different file    
output_tag_file = 'Tagged_File.txt'
with open(output_tag_file, 'w') as output_file:
    for s, a in all_tagged_sentences:
        for word, tag in zip(s, a):
            output_file.write(f'{word}\t{tag}\n')
    output_file.write('\n')    

# # Preprocessing the data

# # Training the data