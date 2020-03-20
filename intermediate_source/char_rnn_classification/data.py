from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
import unicodedata
import string
import random
import itertools

all_letters = "▁" + string.ascii_letters + " .,;'"
n_letters = len(all_letters)
# Build the category_lines dictionary, a list of names per language
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


def load_data(path_regex='../data/names/*.txt'):
    category_lines = {}
    all_categories = []
    for filename in findFiles(path_regex):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    return category_lines, all_categories


def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def lineToIndex(line):
    return [letterToIndex(l) for l in line]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def linesToTensor(lines):
    indexes_batch = [lineToIndex(line) for line in lines]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def batch2TrainData(pair_batch, all_categories):
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(all_categories.index(pair[1]))
    inp, lengths = linesToTensor(input_batch)
    target = torch.LongTensor(output_batch)
    return inp, lengths, target


def create_batches(category_lines, batch_size, seed=7):
    training_examples = []
    for category, lines in category_lines.items():
        training_examples.extend([line, category] for line in lines)
    random.seed(seed)
    random.shuffle(training_examples)
    return batch(training_examples, batch_size)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample(category_lines, all_categories):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
