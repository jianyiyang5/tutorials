import unittest
import math
import torch

from model_transformer import *

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


if __name__ == '__main__':
    unittest.main()
