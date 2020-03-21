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


if __name__ == '__main__':
    unittest.main()