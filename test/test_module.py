import test
import torch

from revert.models import ConvNet, Pipe

N = 20 
x = torch.randn([N, 6, 32])
y = torch.randn([N, 24, 8])

f1 = ConvNet([[32, 6, 4],
              [16, 12, 1]])

f2 = ConvNet([[16, 12, 4],
              [8,  24, 1]])


class TestConv(test.TestCase):
    
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

        
