import random
import os
import torch
from torch import nn, optim
from data import SOS_token, batch_to_transformer_data, create_batches, prepareData, Lang
from model_transformer import TransformerMT, generate_square_subsequent_mask

# For pytorch 1.1
# def maskNLLLoss(inp, target, mask, device):
#     nTotal = (mask == True).sum()
#     crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
#     loss = crossEntropy.masked_select(mask == True).mean()
#     loss = loss.to(device)
#     return loss, nTotal.item()

def maskNLLLoss(inp, target, mask, device):
    nTotal = mask.sum()
    # inp = inp.transpose(0, 1)
    crossEntropy = -torch.log(torch.gather(inp, 2, target.unsqueeze(2)).squeeze(2))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


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

    return mean_loss, n_tokens


if __name__ == '__main__':
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.0001
    src_voc, tgt_voc, pairs = prepareData('eng', 'fra', True, '../data')

    src_vocab_size = src_voc.n_words
    tgt_vocab_size = tgt_voc.n_words

    model_size_factor = 2
    d_model = 512/model_size_factor
    nhead = 8/model_size_factor
    num_encoder_layers = 6/model_size_factor
    num_decoder_layers = 6/model_size_factor
    dim_feedforward = 2048/model_size_factor
    max_seq_length = 30
    pos_dropout = 0.1
    trans_dropout = 0.1
    model = TransformerMT(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                          dim_feedforward, max_seq_length, pos_dropout, trans_dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

