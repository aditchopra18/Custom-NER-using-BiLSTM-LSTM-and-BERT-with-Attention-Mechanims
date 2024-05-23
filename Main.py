import os
import re
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import dataset, dataloader
import transformers

train_file = 'parsing.txt'

# Parsing the file
def read_dataset(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return lines

def parse_dataset(lines):
    sentences = []
    annotations = []
    sentence = []
    
    for line in lines:
        line = line.strip()
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
sentences, annotations = parse_dataset(lines)

