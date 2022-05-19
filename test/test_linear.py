import test
import torch
import torch.nn as nn

from revert.models import Affine, Linear

class TestAffine(test.TestCase):

    def test_affine(self):
        A = Affine(3, 4)
        # weight and bias attributes
        result = isinstance(A.weight, nn.Parameter)\
             and isinstance(A.bias, nn.Parameter)
        self.assertTrue(result)
        # application for dim = -1:
        x = torch.randn(12, 3)
        y = A(x)
        result = tuple(y.shape)
        expect = (12, 4)
        self.assertEqual(expect, result)
        # application for dim < -1:
        T = Affine(4, 16, dim=-2)
        x = torch.randn(128, 4, 12)
        y = T(x)
        result = tuple(y.shape)
        expect = (128, 16, 12)
        self.assertEqual(expect, result)

class TestLinear(test.TestCase):

    def test_linear(self):
        L = Linear(10, 20, dim=0)
        self.assertTrue(isinstance(L.bias, type(None)))
        x = torch.randn(10, 64)
        y = L(x)
        self.assertEqual(tuple(y.shape), (20, 64))
