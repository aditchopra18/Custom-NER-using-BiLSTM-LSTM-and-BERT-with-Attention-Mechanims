import os
import re
import string

train_file = 'parsing.txt'

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
            text += line.split('|')[2] + ' '
            sentence.extend(line.split('|')[2].split())

        elif re.match(r'^\d+\t\d+\t\d+\t', line):
            start, end = int(line.split("\t")[1]), int(line.split("\t")[2])
            annotations.append((start, end, line.split("\t")[3], line.split("\t")[4]))

    if sentence:
        sentences.append(sentence)
    return sentences, annotations, text.strip()

def tag_annotations(sentences, annotations, text):
    tagged_sentences = []
    char_count = 0

    for sentence in sentences:
        tags = ['O'] * len(sentence)
        word_starts = []
        word_ends = []
        char_pos = 0

        for word in sentence:
            word_starts.append(char_pos)
            char_pos += len(word)
            word_ends.append(char_pos)
            char_pos += 1  # Account for the space

        for start, end, disease_info, label in annotations:
            for i, (word_start, word_end) in enumerate(zip(word_starts, word_ends)):
                if word_start >= start and word_end <= end:
                    tags[i] = 'I-' + label
                elif word_start < start < word_end or word_start < end < word_end:
                    tags[i] = 'I-' + label

        tagged_sentences.append((sentence, tags))

    return tagged_sentences

lines = read_dataset(train_file)
paragraphs = parse_dataset(lines)

all_tagged_sentences = []

for paragraph in paragraphs:
    sentences, annotations, text = parse_paragraph(paragraph)
    tagged_sentences = tag_annotations(sentences, annotations, text)
    all_tagged_sentences.extend(tagged_sentences)

output_file = 'Tagged_File.txt'
with open(output_file, 'w') as file:
    for sentence, tags in all_tagged_sentences:
        for word, tag in zip(sentence, tags):
            file.write(f'{word}\t{tag}\n')
        file.write('\n')
