import unittest
from model import EncoderRNN
from data import *
from train_batch import *


class TestModel(unittest.TestCase):
    def test_rnn(self):
        random.seed(2)
        batch_size = 3
        category_lines, all_categories = load_data('../../data/names/*.txt')
        batches = create_batches(category_lines, batch_size)
        batch = next(batches)
        print('batch:', batch)
        inp, lengths, categories, target, mask, max_target_len = batch2TrainData(batch, all_categories)
        max_len = max(lengths).item()

        n_hidden = 16
        n_hidden_cat = 8
        n_categories = len(all_categories)
        rnn = EncoderRNN(n_hidden_cat, n_hidden, torch.nn.Embedding(n_categories, n_hidden_cat),
                         torch.nn.Embedding(n_letters, n_hidden), n_letters)
        rnn.zero_grad()
        outputs, hidden = rnn(inp, categories, lengths)
        # print(outputs)
        print('outputs size:', outputs.size())
        self.assertEqual(torch.Size([max_len, batch_size, n_letters]), outputs.size())
        outputs = outputs.transpose(0, 1)
        print('outputs size after transpose:', outputs.size())
        print('lengths:', lengths)
        # output = outputs[torch.arange(outputs.size(0)), lengths-1]
        # output = outputs[torch.arange(outputs.size(0)), 0]
        # print(output)
        # print('output size:', output.size())
        #
        # mask_loss, nTotal = maskNLLLoss(output, target[0], mask[0])
        # print(mask_loss, nTotal)
        # # print(next(rnn.parameters()).grad.data)
        # mask_loss.backward()
        # print(next(rnn.parameters()).grad.data)

        mask_loss, nTotal = maskNLLLoss(outputs.transpose(0, 1), target, mask)
        print(mask_loss, nTotal)
        mask_loss.backward()
        print(next(rnn.parameters()).grad.data)


if __name__ == '__main__':
    unittest.main()
