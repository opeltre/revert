from revert.transforms import scholkmann

import test
import torch

max_scale = 3
LMS = scholkmann(max_scale, 10)
x = (torch.arange(10).float() - 5) ** 2

class TestScholkmann(test.TestCase):

    def test_lms(self):
        result = LMS(-x)[:,5]
        expect = torch.tensor([True] * max_scale)
        self.assertTrue((expect == result).prod())
        result = LMS(x, -1)[:,5]
        self.assertTrue((expect == result).prod())
