import unittest
from data import *
import torch


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

    def test_letterToIndex(self):
        self.assertEqual(letterToIndex('a'), 1)
        self.assertEqual(letterToIndex('Z'), 52)

    def test_letterToTensor(self):
        tensor = torch.zeros(1, n_letters)
        tensor[0][3] = 1
        # print(tensor)
        # print(letterToTensor('c'))
        self.assertTrue(torch.equal(letterToTensor('c'), tensor))

    def test_lineToTensor(self):
        s = 'test_lineToTensor'
        tensor1 = lineToTensor(s)
        tensor2 = torch.zeros(len(s), 1, n_letters)
        for li, c in enumerate(s):
            tensor2[li][0][letterToIndex(c)] = 1
        self.assertTrue(torch.equal(tensor1, tensor2))
        self.assertEqual(tensor1.size(), torch.Size([17, 1, 58]))

    def test_randomTrainingExample(self):
        random.seed(0)
        category_lines, all_categories = load_data('../../data/names/Chinese.txt')
        category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
        self.assertEqual('Chinese', category)
        self.assertEqual('Guan', line)
        self.assertTrue(torch.equal(torch.tensor([all_categories.index(category)], dtype=torch.long), category_tensor))
        self.assertTrue(torch.equal(lineToTensor(line), line_tensor))

    def test_linesToTensor(self):
        lines = ['abcd', 'def']
        input, lens = inputVar(lines)
        self.assertTrue(torch.equal(torch.tensor([4, 3]), lens))
        expect = torch.tensor([[1, 4], [2, 5], [3, 6], [4, 0]])
        self.assertTrue(torch.equal(expect, input))

    def test_batch2TrainData(self):
        pairs = [['abc', '2'], ['de', '7']]
        inp, lengths, target = batch2TrainData(pairs, [str(i) for i in range(8)])
        self.assertTrue(torch.equal(torch.tensor([3, 2]), lengths))
        expect = torch.tensor([[1, 4],
                               [2, 5],
                               [3, 0]])
        self.assertTrue(torch.equal(expect, inp))
        self.assertTrue(torch.equal(torch.tensor([2, 7]), target))


if __name__ == '__main__':
    unittest.main()
