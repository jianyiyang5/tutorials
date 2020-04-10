import random
import os
import time
import math
import torch
from torch import nn, optim
from data import SOS_token, batch_to_transformer_data, create_batches, prepareData, Lang
from model_transformer import TransformerMT, generate_square_subsequent_mask


def maskNLLLoss(inp, target, mask, device):
    nTotal = (mask == False).sum()
    # inp = inp.transpose(0, 1)
    crossEntropy = -torch.log(torch.gather(inp, 2, target.unsqueeze(2)).squeeze(2))
    loss = crossEntropy.masked_select(mask == False).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train_batch(batch, model, optimizer, device, src_voc, tgt_voc):
    optimizer.zero_grad()
    src_tensor, tgt_tensor, src_pad_mask, tgt_pad_mask, mem_pad_mask = batch_to_transformer_data(src_voc, tgt_voc, batch)
    src_tensor = src_tensor.to(device)
    tgt_tensor = tgt_tensor.to(device)
    src_pad_mask = src_pad_mask.to(device)
    tgt_pad_mask = tgt_pad_mask.to(device)
    mem_pad_mask = mem_pad_mask.to(device)
    tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(0)).to(device)

    output = model(src_tensor, tgt_tensor, src_pad_mask, tgt_pad_mask, mem_pad_mask, tgt_mask)
    mean_loss, n_tokens = maskNLLLoss(output, tgt_tensor, tgt_pad_mask, device)
    mean_loss.backward()
    optimizer.step()

    return mean_loss, n_tokens, output


def train_epoch(src_voc, tgt_voc, pairs, model, optimizer, device, epoch, batch_size, since):
    batches = create_batches(pairs, batch_size)
    total_loss = 0
    total_tokens = 0

    i = 0
    for batch in batches:
        i += 1
        mean_loss, n_tokens, output = train_batch(batch, model, optimizer, device, src_voc, tgt_voc)
        total_loss += mean_loss * n_tokens
        total_tokens += n_tokens
        if i % 10 == 0:
            predicted = []
            for j in range(output.size(0)):
                _, indices = torch.topk(output[j][0], 1)
                predicted.append(tgt_voc.index2word[indices[0].item()])
            print(f'Batch: {i}; Mean Loss: {mean_loss}; tokens: {n_tokens}')
    print(f'Epoch: {epoch}; Average loss: {total_loss/total_tokens}; Wall time: {timeSince(since)}; Predicted: f{" ".join(predicted)}')
    print('------------------------------------------------------------')


def train(src_voc, tgt_voc, pairs, model, optimizer, device, max_epochs=25, batch_size=64, out_path='output/transformer.pt'):
    model = model.to(device)
    model.train()

    since = time.time()
    print('start training ...')
    for epoch in range(0, max_epochs):
        train_epoch(src_voc, tgt_voc, pairs, model, optimizer, device, epoch, batch_size, since)
        save_model(model, optimizer, out_path, epoch)
    print('training completed')


def save_model(model, optimizer, out_path, epoch):
    torch.save({
        'epoch': epoch,
        'model_dict': model.state_dict(),
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'optimizer_dict': optimizer.state_dict()
    }, f'{out_path}.{epoch}')
    print(f'Model saved to {out_path}.{epoch}')


def load_model(model_path):
    checkpoint = torch.load(model_path)
    src_vocab_size = checkpoint['src_vocab_size']
    tgt_vocab_size = checkpoint['tgt_vocab_size']
    model = create_model(src_vocab_size, tgt_vocab_size)
    model.load_state_dict(checkpoint['model_dict'])
    optimizer = create_optimizer(model)
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    print(f'Load model from {model_path}')
    return model, optimizer


def create_model(src_vocab_size, tgt_vocab_size, d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=1024, max_seq_length=30, pos_dropout=0.1, trans_dropout=0.1):
    return TransformerMT(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                         dim_feedforward, max_seq_length, pos_dropout, trans_dropout)


def create_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.0001)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_voc, tgt_voc, pairs = prepareData('eng', 'fra', True, '../data')
    src_vocab_size = src_voc.n_words
    tgt_vocab_size = tgt_voc.n_words

    # model_size_factor = 2
    # d_model = int(512/model_size_factor)
    # nhead = int(8/model_size_factor)
    # num_encoder_layers = int(6/model_size_factor)
    # num_decoder_layers = int(6/model_size_factor)
    # dim_feedforward = int(2048/model_size_factor)
    # max_seq_length = 30
    # pos_dropout = 0.1
    # trans_dropout = 0.1
    # model = TransformerMT(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
    #                       dim_feedforward, max_seq_length, pos_dropout, trans_dropout)

    out_path = 'output/transformer.pt'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model = create_model(src_vocab_size, tgt_vocab_size)
    # optimizer = optim.Adam(model.parameters(), lr=
    optimizer = create_optimizer(model)
    train(src_voc, tgt_voc, pairs, model, optimizer, device, max_epochs=10, batch_size=64, out_path=out_path)

