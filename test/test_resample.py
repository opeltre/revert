from revert.transforms import resample

import test
import torch

x = torch.linspace(.1, 1, 10)
y = resample(100)(x)

# cubic polynomial
P = lambda t : t ** 3 -  t ** 2 + t - 1

class TestResample(test.TestCase):

    def test_resample_fixed(self):
        result = y[::11]
        expect = x
        self.assertClose(expect, result)
    
    def test_resample_cubic(self):
        result = resample(100)(P(x))
        expect = P(resample(100)(x))
        self.assertClose(expect, result, tol=1e-3)
