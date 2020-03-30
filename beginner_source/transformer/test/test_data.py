import unittest
import random
import torch

from data import load_data, get_batch


class TestData(unittest.TestCase):
    def test_load_data(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TEXT, train_data, val_data, test_data = load_data(device)
        print(test_data)

        data, target = get_batch(train_data, 1)
        print(data.size())
        print(target.size())
        self.assertEqual(torch.Size([35, 20]), data.size())
        self.assertEqual(torch.Size([700]), target.size())