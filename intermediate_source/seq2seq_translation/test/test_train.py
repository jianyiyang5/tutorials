import unittest

from train import *
from data import *
from model import *
from utils import *


class TestTrain(unittest.TestCase):
    def test_train(self):
        input_lang, output_lang, pairs = prepareData('eng', 'fra', True, '../../data')
        hidden_size = 256
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device=device).to(device)
        attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, device=device).to(device)

        trainIters(encoder1, attn_decoder1, device, input_lang, output_lang, pairs, 500, print_every=100)

        evaluateRandomly(encoder1, attn_decoder1, device, input_lang, output_lang, pairs)

        output_words, attentions = evaluate(encoder1, attn_decoder1, device, "je suis trop froid .",
                                            input_lang, output_lang)
        plt.matshow(attentions.numpy())

        evaluateAndShowAttention(encoder1, attn_decoder1, device, input_lang, output_lang, "elle a cinq ans de moins que moi .")

