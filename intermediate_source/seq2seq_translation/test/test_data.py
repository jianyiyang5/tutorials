import unittest
import random

from data import *


class TestData(unittest.TestCase):
    def test_prepareData(self):
        input_lang, output_lang, pairs = prepareData('eng', 'fra', True, '../../data')
        print(random.choice(pairs))


    def test_batch2TrainData(self):
        input_lang, output_lang, pairs = prepareData('eng', 'fra', True, '../../data')
        # Example for validation
        small_batch_size = 5
        batches = batch2TrainData(input_lang, output_lang, [random.choice(pairs) for _ in range(small_batch_size)])
        input_variable, lengths, target_variable, mask, max_target_len = batches

        print("input_variable:", input_variable)
        print("lengths:", lengths)
        print("target_variable:", target_variable)
        print("mask:", mask)
        print("max_target_len:", max_target_len)

    def test_create_batches(self):
        input_lang, output_lang, pairs = prepareData('eng', 'fra', True, '../../data')
        print(next(create_batches(pairs, 3)))