from infusion import rates

import torch
import math
import test 

# 24s sampled at 100 Hz 
N = 2400
t = torch.arange(N) * 2 * math.pi / 100
fr, fc = (0.5, 1.)
ranges = ((.25, .6), (.8, 1.2))
x = torch.sin(fr * t) + torch.sin(fc * t)


class TestRates(test.TestCase):

    def test_rates(self):
        result = torch.tensor(rates(x, ranges=ranges))
        expect = torch.tensor([fr, fc])
        self.assertClose(expect, result, tol=1e-2)
