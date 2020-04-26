import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator


def get_data(batch_size=128, device=torch.device('cpu')):
    SRC = Field(tokenize="spacy",
                tokenizer_language="de",
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
    TRG = Field(tokenize="spacy",
                tokenizer_language="en",
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device)
    return train_iterator, valid_iterator, test_iterator, SRC, TRG


if __name__ == '__main__':
    train_iterator, valid_iterator, test_iterator, SRC, TRG = get_data(batch_size=64)
    for _, batch in enumerate(train_iterator):
        print(batch.src, batch.trg)
        break
