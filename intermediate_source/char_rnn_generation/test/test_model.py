import unittest
from model import EncoderRNN
from data import *


class TestModel(unittest.TestCase):
    def test_rnn(self):
        random.seed(2)
        batch_size = 3
        category_lines, all_categories = load_data('../../data/names/*.txt')
        batches = create_batches(category_lines, batch_size)
        batch = next(batches)
        inp, lengths, categories, target, mask, max_target_len = batch2TrainData(batch, all_categories)
        max_len = max(lengths).item()

        n_hidden = 16
        n_hidden_cat = 8
        n_categories = len(all_categories)
        rnn = EncoderRNN(n_hidden_cat, n_hidden, torch.nn.Embedding(n_categories, n_hidden_cat),
                         torch.nn.Embedding(n_letters, n_hidden), n_letters)
        outputs, hidden = rnn(inp, categories, lengths)
        print(outputs)
        self.assertEqual(torch.Size([max_len, batch_size, n_letters]), outputs.size())


if __name__ == '__main__':
    unittest.main()
