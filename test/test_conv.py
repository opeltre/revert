import test
import torch

from revert.models import ConvNet

N = 32

class TestConvNet (test.TestCase):

    def test_shapes(self):
        # Nc = 1 implicit in input
        conv1 = ConvNet([[12, 1, 3],[4, 3, 1]])
        x1 = torch.randn([N, 12])
        result = tuple(conv1(x1).shape)
        expect = (N, 3, 4)
        self.assertEqual(expect, result)
        # Npts = 1 not squeezed on output
        conv2 = ConvNet([[8, 2, 4],[4, 4, 4],[1, 8, 1]])
        x2 = torch.randn([N, 2, 8])
        result = tuple(conv2(x2).shape)
        expect = (N, 8, 1)
        self.assertEqual(expect, result)

        
        
