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

    def test_input_tensor_with_mask(self):
        input_lang, output_lang, pairs = prepareData('eng', 'fra', True, '../../data')
        src_sentences = [src for src, _ in pairs[0:3]]
        print(src_sentences)
        print(input_tensor_with_mask(src_sentences, input_lang))
        # row batch, col sequence
        expected = [[ 4,  8, 11],
        [ 5,  9, 12],
        [ 6, 10,  7],
        [ 7,  7,  2],
        [ 2,  2,  0]]
        expected_mask = [[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True],
        [ True,  True, False]]
        input_tensor, mask = input_tensor_with_mask(src_sentences, input_lang)
        self.assertTrue(torch.equal(torch.LongTensor(expected), input_tensor))
        self.assertTrue(torch.equal(torch.BoolTensor(expected_mask), mask))

        tgt_sentences = [tgt for _, tgt in pairs[0:3]]
        tgt_tensor, tgt_mask, _ = outputVar(tgt_sentences, output_lang)
        print(tgt_sentences)
        print(tgt_tensor, tgt_mask)
        print(tgt_tensor.size())
