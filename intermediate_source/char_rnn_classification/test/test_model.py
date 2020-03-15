import unittest
from model import RNN
from data import *


class TestModel(unittest.TestCase):
    def test_rnn(self):
        n_hidden = 128
        n_categories = 3
        rnn = RNN(n_letters, n_hidden, n_categories)

        input = letterToTensor('A')
        hidden = torch.zeros(1, n_hidden)
        output1, next_hidden1 = rnn(input, hidden)
        # print(output1, next_hidden1)

        input = lineToTensor('Albert')
        hidden = torch.zeros(1, n_hidden)
        output2, next_hidden2 = rnn(input[0], hidden)
        print(output2, output2.size())
        print(next_hidden2, next_hidden2.size())

        self.assertTrue(torch.equal(output1, output2))
        self.assertTrue(torch.equal(next_hidden1, next_hidden2))


if __name__ == '__main__':
    unittest.main()
