import unittest
import torch

from train_batch import load_model
from predict_batch import GreedySearchDecoder, evaluate
from data import indexesFromSentence2, MAX_LENGTH

class TestTrainBatch(unittest.TestCase):

    def test_predict(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_size = 256
        encoder_n_layers = 2
        decoder_n_layers = 2
        dropout = 0.1

        encoder, decoder, src_voc_dict, tgt_voc_dict = \
            load_model('output', 'test_batch', 'en-fr', encoder_n_layers, decoder_n_layers, hidden_size, dropout)
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        searcher = GreedySearchDecoder(encoder, decoder, device)

        indexes_batch = [indexesFromSentence2(src_voc_dict, 'elle porte une belle montre .')]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, lengths, MAX_LENGTH)
        print(tokens, scores)

    def test_evaluate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_size = 256
        encoder_n_layers = 2
        decoder_n_layers = 2
        dropout = 0.1

        encoder, decoder, src_voc_dict, tgt_voc_dict = \
            load_model('output', 'test_batch', 'en-fr', encoder_n_layers, decoder_n_layers, hidden_size, dropout)
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        searcher = GreedySearchDecoder(encoder, decoder, device)

        decoded_words = evaluate(searcher, device, src_voc_dict, tgt_voc_dict, 'elle porte une belle montre .')
        print(decoded_words)
        decoded_words = evaluate(searcher, device, src_voc_dict, tgt_voc_dict, 'je suis content de vous revoir .')
        print(decoded_words)
        decoded_words = evaluate(searcher, device, src_voc_dict, tgt_voc_dict, 'vous etes vraiment tres productive aujourd hui .')
        print(decoded_words)
        decoded_words = evaluate(searcher, device, src_voc_dict, tgt_voc_dict, 'il adore marcher abc.')
        print(decoded_words)

