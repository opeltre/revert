import test
import torch

from revert.models import ConvNet, Pipe

N = 20 
x = torch.randn([N, 6, 32])

f1 = ConvNet([[32, 6, 4],
              [16, 12, 1]])

f2 = ConvNet([[16, 12, 4],
              [8,  24, 1]])

class TestConv(test.TestCase):

    def test_pipe (self): 
        model = Pipe(f1, f2)
        result = tuple(model(x).shape)
        expect = (N, 24, 8)
        self.assertEqual(expect, result)
