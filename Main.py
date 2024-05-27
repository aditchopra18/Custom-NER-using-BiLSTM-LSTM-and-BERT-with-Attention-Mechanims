import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
import string

train_file = 'parsing.txt'
def read_dataset(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return lines

# def remove_punctuations(string_list):
#     translator = str.maketrans('', '', string.punctuation)
#     return [s.translate(translator) for s in string_list]

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
    split_sentence = []

    for line in paragraph:
        if re.match(r'^\d+\|\w\|', line):
            # if sentence:
            #     sentences.append(sentence)
            #     sentence = []
            text += line.split('|')[2] + ' '
            sentence.extend(line.split('|')[2].split())
            for strings in sentence:
                tokens = re.findall(r'\w+|[^\w\s]', strings, re.UNICODE)
                split_sentence.extend(tokens)

        elif re.match(r'^\d+\t\d+\t\d+\t', line):
            start, end = int(line.split("\t")[1]), int(line.split("\t")[2])
            annotations.append((start, end, line.split("\t")[3], line.split("\t")[4]))

    if split_sentence:
        sentences.append(split_sentence)
    return sentences, annotations

def tag_annotations(sentences, annotations):
    tagged_sentences = []
    char_count = 0
    
    for sentence in sentences:
        # edited_sent = remove_punctuations(sentence)
        tags = ['O'] * len(sentence)
        word_starts = []
        char_pos = 0
        for word in sentence:
            word_starts.append(char_pos)
            char_pos += len(word) + 1
        
        for start, end, disease_info, label in annotations:
            for i, word_start in enumerate(word_starts):
                word_end = word_start + len(sentence[i])
                if word_start <= start < word_end or word_start < end <= word_end: # The words having punctuation either at start or end
                    tags[i] = 'I-' + label
                if start <= word_start < end or start < word_end <= end: # The words without punctuations 
                    tags[i] = 'I-' + label
        
        tagged_sentences.append((sentence, tags))
        
    return tagged_sentences

lines = read_dataset(train_file)
paragraphs = parse_dataset(lines)

all_tagged_sentences = []

for paragraph in paragraphs:
    sentences, annotations = parse_paragraph(paragraph)
    print (sentences)
    tagged_sentences = tag_annotations(sentences, annotations)
    all_tagged_sentences.extend(tagged_sentences)

output_file = 'Tagged_File.txt'
with open(output_file, 'w') as file:
    for sentence, tags in all_tagged_sentences:
        for word, tag in zip(sentence, tags):
            file.write(f'{word}\t{tag}\n')
        file.write('\n')