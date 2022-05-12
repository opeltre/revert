import test
import torch

from revert.models import View, ConvNet
N = 64

class TestView(test.TestCase):

    def test_view (self):
        conv = ConvNet([[6, 32], [12, 1], [12]])
        view = View([32])
        model = view @ conv
        x = torch.randn([N, 6, 12])
        result = tuple(model(x).shape)
        expect = (N, 32)
        self.assertEqual(expect, result)
