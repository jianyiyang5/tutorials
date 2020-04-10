import unittest
import time
import math
import torch
from torch import nn
from torch import optim

from data import *
from model_transformer import *
from train_transformer import train_batch, train_epoch, load_model

class TransformerTestCase(unittest.TestCase):
    def testPositionalEncoding(self):
        max_len = 5
        position = torch.arange(0, max_len, dtype=torch.float)
        print(position)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        print(position)

        d_model = 7
        print(torch.arange(0, d_model, 2).float())
        print((-math.log(10000.0) / d_model))

        print(torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)))

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        print(position.size(), div_term.size())
        print(torch.sin(position * div_term), torch.sin(position * div_term).size())
        print(torch.cos(position * div_term))

        pe = torch.zeros(max_len, d_model).unsqueeze(0).transpose(0, 1)
        print(pe.size())

    def test_generate_subsequent_mask(self):
        print(generate_subsequent_mask(3, 5))

    def test_model(self):
        input_lang, output_lang, pairs = prepareData('eng', 'fra', True, '../../data')
        src_sentences = [src for src, _ in pairs[0:3]]
        tgt_sentences = [tgt for _, tgt in pairs[0:3]]
        src_tensor, src_pad_mask = tensor_with_mask(src_sentences, input_lang)
        tgt_tensor, tgt_pad_mask = tensor_with_mask(tgt_sentences, output_lang)
        mem_pad_mask = src_pad_mask.clone()
        tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(0))
        src_vocab_size = input_lang.n_words
        tgt_vocab_size = output_lang.n_words
        d_model = 8
        nhead = 2
        num_encoder_layers = 2
        num_decoder_layers = 2
        dim_feedforward = 16
        max_seq_length = 11
        pos_dropout = 0
        trans_dropout = 0
        model = TransformerMT(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                              dim_feedforward, max_seq_length, pos_dropout, trans_dropout)
        output = model(src_tensor, tgt_tensor, src_pad_mask, tgt_pad_mask, mem_pad_mask, tgt_mask)
        print(output.size())
        print(output)

    def test_train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        src_voc, tgt_voc, pairs = prepareData('eng', 'fra', True, '../../data')
        src_vocab_size = src_voc.n_words
        tgt_vocab_size = tgt_voc.n_words
        d_model = 8
        nhead = 2
        num_encoder_layers = 2
        num_decoder_layers = 2
        dim_feedforward = 16
        max_seq_length = 11
        pos_dropout = 0
        trans_dropout = 0
        model = TransformerMT(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                              dim_feedforward, max_seq_length, pos_dropout, trans_dropout)
        optimizer = optim.Adam(model.parameters())
        mean_loss, n_tokens = train_batch(pairs[0:3], model, optimizer, device, src_voc, tgt_voc)
        print(mean_loss, n_tokens)

    def test_epoch(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        src_voc, tgt_voc, pairs = prepareData('eng', 'fra', True, '../../data')
        src_vocab_size = src_voc.n_words
        tgt_vocab_size = tgt_voc.n_words
        d_model = 8
        nhead = 2
        num_encoder_layers = 2
        num_decoder_layers = 2
        dim_feedforward = 16
        max_seq_length = 11
        pos_dropout = 0
        trans_dropout = 0
        batch_size = 64
        model = TransformerMT(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                              dim_feedforward, max_seq_length, pos_dropout, trans_dropout)
        optimizer = optim.Adam(model.parameters())
        train_epoch(src_voc, tgt_voc, pairs, model, optimizer, device, 0, batch_size, time.time())

    def test_predict(self):
        model, _ = load_model('../output/transformer.pt.9')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        src_voc, tgt_voc, pairs = prepareData('eng', 'fra', True, '../../data')
        batches = create_batches(pairs, 3)
        batch = next(batches)
        src_tensor, tgt_tensor, src_pad_mask, tgt_pad_mask, mem_pad_mask = batch_to_transformer_data(src_voc, tgt_voc, batch)
        src_tensor = src_tensor.to(device)
        tgt_tensor = tgt_tensor.to(device)
        src_pad_mask = src_pad_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)
        mem_pad_mask = mem_pad_mask.to(device)
        tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(0)-1).to(device)
        outputs = model.forward(src_tensor, tgt_tensor[:-1, :], src_key_padding_mask=src_pad_mask,
                                tgt_key_padding_mask=tgt_pad_mask[:-1, :], memory_key_padding_mask=mem_pad_mask,
                                tgt_mask=tgt_mask)
        print(outputs.size())
        print(batch)
        for i in range(outputs.size(0)):
            _, indices = torch.topk(outputs[i][0], 1)
            print(tgt_voc.index2word[indices[0].item()])



if __name__ == '__main__':
    unittest.main()
