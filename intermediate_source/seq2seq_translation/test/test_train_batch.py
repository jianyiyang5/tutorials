import unittest
import torch
from torch import optim, nn

from data import prepareData, batch2TrainData, create_batches
from model_batch import EncoderRNNBatch, LuongAttnDecoderRNN
from train_batch import train


class TestTrainBatch(unittest.TestCase):

    def test_train(self):
        batch_size = 3
        clip = 50
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learning_rate = 0.0001
        decoder_learning_ratio = 5.0
        hidden_size = 16
        encoder_n_layers = 2
        decoder_n_layers = 2
        dropout = 0.1
        attn_model = 'dot'

        input_lang, output_lang, pairs = prepareData('eng', 'fra', True, '../../data')
        batches = create_batches(pairs, batch_size)
        input_variable, lengths, target_variable, mask, max_target_len = \
            batch2TrainData(input_lang, output_lang, next(batches))

        encoder = EncoderRNNBatch(hidden_size, nn.Embedding(input_lang.n_words, hidden_size), encoder_n_layers, dropout)
        decoder = LuongAttnDecoderRNN(attn_model, nn.Embedding(output_lang.n_words, hidden_size), hidden_size,
                                      input_lang.n_words, decoder_n_layers, dropout)

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, batch_size, clip, device, teacher_forcing_ratio=1.0)
        print(loss)
