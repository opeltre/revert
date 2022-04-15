"verif très proche de null, reussir à remettre dans le bon sens avec le y"

from revert.transforms import shift_all

import test
import torch

N = 500
x = torch.arange(32).repeat(6).repeat(N).view([N, 6 ,32])
one, y = shift_all(0.5)(x)

x2 = []
for i in range(N) :  
    x2.append([])
    for j in range(6)  :
        x2[i].append([j for _ in range(32)])
x2 = torch.tensor(x2)
two, y2 = shift_all(0.5)(x2)


def unshift(stdev):
    def run_shift(x, y):
        N = len(x)
        Nc = x.shape[1]
        Npts = x.shape[-1]
        # generate and convert to tensor
        idx = torch.arange(Npts).repeat(Nc*N).view([Nc*N, Npts])
        # generate the guass distribution
        y = -y
        y_index = (y * (Npts / 2)).flatten().long() 
        idx = (idx + y_index[:,None]) % Npts
        idx = (torch.arange(Nc*N)[:,None] * Npts + idx).flatten()
        x_prime = x.flatten()[idx].view([N, Nc, Npts])
        return x_prime, y
    return run_shift

class TestShift(test.TestCase):

    # Test if the sum of the tensor y = 0
    
    def test_const(self):
        result = torch.sum(y)
        expect = torch.linspace(0,1,1)
        self.assertClose(expect, result, tol=1e-4)
    
    # Test with that type of list : [[0...32], [0...32], ... [0...32]]
    
    def test_unshift(self):
        result, _ = unshift(0.5)(one, y)
        expect = x
        self.assertClose(expect, result)
        
    # Test with that type of list : [[0...0], [1...1], ... [5...5]]

    def test_unshift2(self):
        result, _ = unshift(0.5)(two, y2)
        expect = x2
        self.assertClose(expect, result)