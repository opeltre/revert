from revert.models import SoftMin

import test
import torch

model = SoftMin()

def dirac(i, N):
    y = torch.zeros([N])
    y[i] = 1.
    return y.tolist()

class TestSoftMin(test.TestCase):

    # Making sure that the elements of the n-dimensional output Tensor lie in the range [0, 1] and sum to 1
    def test_forward(self):
        Nb, Nc, Npts = 10, 6, 32
        x = torch.randn(Nb, Nc, Npts)
        y = model.forward(x)
        result = torch.ge(y, 0).int()
        expect = torch.ones(Nb, Nc, Npts)
        self.assertClose(expect, result)

        result = y.sum(-1)
        expect = torch.ones(Nb, Nc)
        self.assertClose(expect, result)

    # Test if softmin returns coherent values with dirac
    def test_dirac(self):
        N, Nb = 32, 10
        i = torch.randint(0, N, (Nb,))
        dirac_i = torch.tensor([dirac(val.item(), N) for val in i])
        j = torch.randint(0, N, (Nb,)).unsqueeze(0)
        result = model.loss(dirac_i, j)
        expect = (j - i).abs().sum()
        self.assertClose(expect, result)
        
        dirac_j = torch.tensor([dirac(val.item(), N) for val in j.squeeze_(0)])
        p = (dirac_i + dirac_j) / 2
        k = torch.randint(0, N, (Nb,)).unsqueeze(0)
        result = model.loss(p, k)
        expect = 0.5 * ((k - i).abs().sum() + (k - j).abs().sum())
        self.assertClose(expect, result)
