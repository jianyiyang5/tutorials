import unittest
import torch.nn as nn
from data import *


class TestData(unittest.TestCase):
    def test_findFiles(self):
        files = findFiles('../../data/names/*.txt')
        self.assertIn('../../data/names/Czech.txt', files)

    def test_unicodeToAscii(self):
        a = unicodeToAscii('Ślusàrski')
        self.assertEqual(a, 'Slusarski')

    def test_load_data(self):
        category_lines, all_categories = load_data('../../data/names/Chinese.txt')
        self.assertEqual(len(all_categories), 1)
        self.assertEqual(len(category_lines['Chinese']), 268)

    def test_categoryTensor(self):
        category_lines, all_categories = load_data('../../data/names/*.txt')
        tensor = categoryIdxTensor(['Chinese', 'English'], all_categories)
        print(tensor.size())
        self.assertEqual(tensor.size(), torch.Size([2]))

        embedding = nn.Embedding(18, 8)
        embedded = embedding(tensor)
        print(embedded.size())
        self.assertEqual(embedded.size(), torch.Size([2, 8]))

        lengths = torch.tensor([5, 3])
        print(lengths.size())

        embedded = embedded.view(1, 2, 8)
        print(embedded)

        embedded = embedded.repeat(5, 1, 1)
        self.assertEqual(embedded.size(), torch.Size([5, 2, 8]))
        print(embedded)

        embedding2 = nn.Embedding(12, 4)
        input_seq = torch.LongTensor([[3,4,5,9,10],[1,11,3,0,0]])
        input_seq = input_seq.transpose(0, 1)
        input_embeded = embedding2(input_seq)
        print(input_embeded.size())

        combined = torch.cat((input_embeded, embedded), 2)
        print(combined.size())


if __name__ == '__main__':
    unittest.main()