import test 
import torch

from revert.models import cross_correlation

def xcorr(x, y): 
    x_ = x - x.mean([0])
    y_ = y - y.mean([0])
    C = torch.zeros([x.shape[1], y.shape[1]])
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            dot_ij = (x_[:,i] * y_[:,j]).sum() 
            nx_i = x_[:,i].norm()
            ny_j = y_[:,j].norm()
            C[i, j] = dot_ij / (nx_i * ny_j)
    return C


class TestXCorr(test.TestCase): 

    def test_xcorr(self):
        x, y = torch.randn([2, 20, 5])
        result = cross_correlation(x, y)
        expect = xcorr(x, y)
        self.assertClose(expect, result)
