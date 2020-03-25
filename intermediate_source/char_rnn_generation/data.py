from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import random
import torch
import itertools


all_letters = "▁" + string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker
PAD_token = 0


def findFiles(path): return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# Build the category_lines dictionary, a list of lines per category
def load_data(path_regex='../data/names/*.txt'):
    category_lines = {}
    all_categories = []
    for filename in findFiles(path_regex):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    return category_lines, all_categories


# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# Get a random category and random line from that category
def randomTrainingPair(category_lines, all_categories):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


# One-hot vector for category
def categoryTensor(category, all_categories):
    n_categories = len(all_categories)
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)


# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample(category_lines, all_categories):
    category, line = randomTrainingPair(category_lines, all_categories)
    category_tensor = categoryTensor(category, all_categories)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


def categoryIdxTensor(categories, all_categories):
    li = [all_categories.index(category) for category in categories]
    return torch.LongTensor(li)


def letterToIndex(letter):
    return all_letters.find(letter)


def lineToIndex(line):
    return [letterToIndex(l) for l in line]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def inputVar(lines):
    indexes_batch = [lineToIndex(line) for line in lines]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(lines):
    indexes_batch = [lineToIndex(sentence[1:]) for sentence in lines]
    for idx in indexes_batch:
        idx.append(n_letters - 1) # EOS
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2TrainData(pair_batch, all_categories):
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, category_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        category_batch.append(all_categories.index(pair[1]))
    inp, lengths = inputVar(input_batch)
    target, mask, max_target_len = outputVar(input_batch)
    categories = torch.LongTensor(category_batch)
    return inp, lengths, categories, target, mask, max_target_len


def create_batches(category_lines, batch_size):
    training_examples = []
    for category, lines in category_lines.items():
        random.shuffle(lines)
        training_examples.extend([line, category] for line in lines[:batch_size])
    random.shuffle(training_examples)
    return batch(training_examples, batch_size)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]



