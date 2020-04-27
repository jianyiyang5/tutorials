import os

import torch
from torchtext.datasets import text_classification


NGRAMS = 2


def get_datasets():
    if not os.path.isdir('./.data'):
        os.mkdir('./.data')
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
        root='./.data', ngrams=NGRAMS, vocab=None)
    return train_dataset, test_dataset


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label