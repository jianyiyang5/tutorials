import unittest
import math

from helper import *


class HelperTestCase(unittest.TestCase):
    def test_argmax(self):
        idx = argmax(vec=torch.tensor([[3, 2, 4]]))
        print(idx)

    def test_prepare_sequence(self):
        to_ix = {'a': 1, 'b': 2, 'c': 3}
        seq = ['b', 'a', 'c']
        t = prepare_sequence(seq, to_ix)
        print(t)

    def test_log_sum_exp(self):
        vec = torch.tensor([[7, 2, 5]], dtype=float)
        score = log_sum_exp(vec)
        print(score)

        s = sum(math.exp(x) for x in [7, 2, 5])
        print(math.log(s))

        s = sum(math.exp(x-7) for x in [7, 2, 5])
        print(7+math.log(s))


if __name__ == '__main__':
    unittest.main()
