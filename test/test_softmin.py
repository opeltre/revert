from revert.models import SoftMin

import test
import torch

model = SoftMin()
x = torch.randn(6, 32)

class TestSoftMin(test.TestCase):

    # Test if softmin returns positive values
    
    def test_softmin(self):
        y = model.forward(x)
        result = y.abs()
        expect = y
        self.assertClose(expect, result)
