import unittest
import random

from data import *


class TestData(unittest.TestCase):
    def test_prepareData(self):
        input_lang, output_lang, pairs = prepareData('eng', 'fra', True, '../../data')
        print(random.choice(pairs))
