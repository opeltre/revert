from revert.transforms import shift_all, shift_one, unshift

import test
import torch

# shift_all()
N = 500
x = torch.arange(32).repeat(6).repeat(N).view([N, 6 ,32])

x2 = []
for i in range(N) :  
    x2.append([])
    for j in range(6)  :
        x2[i].append([j for _ in range(32)])
x2 = torch.tensor(x2)

# shift_one()
x3 = torch.rand(6, 32)
y3 = torch.tensor([0 for _ in range(6)])
shift = torch.randint(-16, 17, (1,)).item()
nb_chan = torch.randint(0, len(x3), (1,))
y3[nb_chan] = shift

class TestShift(test.TestCase):

    # Test if the sum of the tensor y = 0
    
    def test_shift_all_mean(self):
        one, y = shift_all(0.5)(x)
        result = torch.sum(y)
        expect = torch.linspace(0,1,1)
        self.assertClose(expect, result, tol=1e-4)
    
    # Test with that type of list : [[0...32], [0...32], ... [0...32]]
    
    def test_unshift(self):
        one, y = shift_all(0.5)(x)
        result = unshift(one, y)
        expect = x
        self.assertClose(expect, result)
        
    # Test with that type of list : [[0...0], [1...1], ... [5...5]]

    def test_unshift2(self):
        two, y2 = shift_all(0.5)(x2)
        result = unshift(two, y2)
        expect = x2
        self.assertClose(expect, result)

    # Test shift_one()

    def test_shift_one(self):
        result_x, _ = shift_one(x3, y3)
        for i in range(len(y3)):
            with self.subTest(i=i):
                if y3[i].item() == 0:
                    self.assertClose(result_x[i], x3[i])
                else:
                    self.assertClose(result_x[i], x3[i].roll(-y3[i].item(), -1))
