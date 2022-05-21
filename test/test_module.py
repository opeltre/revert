import test
import torch

from revert.models import ConvNet, Pipe, Prod, View,\
                          Stack, Cat, Cut

N = 20 
x = torch.randn([N, 6, 32])
y = torch.randn([N, 24, 8])

f1 = ConvNet([[6, 12],  [32, 16], [4]])
f2 = ConvNet([[12, 24], [16, 8],  [4]])


class TestModule(test.TestCase):
    
    def test_matmul(self):
        model = f2 @ f1
        result = tuple(model(x).shape)
        expect = (N, 24, 8)
        self.assertEqual(expect, result)

    def test_pipe(self): 
        model = Pipe(f1, f2)
        result = tuple(model(x).shape)
        expect = (N, 24, 8)
        self.assertEqual(expect, result)

    def test_loss(self):
        f2.loss = lambda out, tgt : ((out - tgt)**2).sum()
        model = Pipe(f1, f2)
        loss = model.loss_on(x, y)
        self.assertTrue(loss.shape == ())

    def test_cat(self):
        cat = Cat(2)
        x = torch.randn(10, 4, 6)
        y = torch.randn(10, 4, 12)
        result = tuple(cat([x, y]).shape)
        expect = (10, 4, 18)
        self.assertEqual(expect, result)
    
    def test_stack(self):
        stack = Stack(1)
        x = torch.randn(10, 4, 8)
        result = tuple(stack([x, x]).shape)
        expect = (10, 2, 4, 8)
        self.assertEqual(expect, result)
    
    def test_cut(self):
        cut = Cut([2, 3], 1)
        x = torch.randn(10, 5, 2)
        y, z = cut(x)
        result = (tuple(y.shape), tuple(z.shape))
        expect = ((10, 2, 2), (10, 3, 2))
        self.assertEqual(expect, result)

    def test_prod(self):
        prod = Prod(View([2, 3, 4]), View([3, 3, 4]), ns=[2, 3], dim=1)
        x = torch.randn(128, 5, 12)
        y = prod(x)
        result = tuple(y.shape)
        expect = (128, 5, 3, 4)
        self.assertEqual(expect, result)
