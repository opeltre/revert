from revert.transforms import Troughs

import test
import torch
import math

troughs = Troughs(1200, 40)
# 12s at 100Hz
t = torch.linspace(0, 12 * 2 * math.pi, 1200)
x = torch.cos(t) + torch.cos(0.3 * t)

class TestTroughs(test.TestCase):

    def test_troughs(self):
        result = troughs(x)
        expect = torch.arange(12) * 100 + 50
        self.assertClose(expect, result, tol=5)
