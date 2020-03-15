import unittest
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

    def test_letterToIndex(self):
        self.assertEqual(letterToIndex('a'), 0)
        self.assertEqual(letterToIndex('Z'), 51)

    def test_letterToTensor(self):
        tensor = torch.zeros(1, n_letters)
        tensor[0][2] = 1
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
        self.assertEqual(tensor1.size(), torch.Size([17, 1, 57]))


if __name__ == '__main__':
    unittest.main()
