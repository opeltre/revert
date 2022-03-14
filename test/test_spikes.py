import torch
import test

from revert.transforms import find_spikes, filter_spikes

y = torch.linspace(0, 1, 100)

x1 = y + 0.
x1[:20]   = 1000
x1[40:60] = 1000

x2 = y + 0.
x2[40:60] = -1000
x2[80:]   = -1000

class TestSpikes(test.TestCase):
    
    def test_find_spikes(self):
        id1, mask1 = find_spikes(x1)
        # True outside bounds
        result = mask1[:20].prod() * mask1[40:60].prod()
        expect = torch.tensor(1)
        self.assertClose(expect, result)
        # False inside bounds
        result = mask1[20:40].sum() + mask1[60:].sum()
        expect = torch.tensor(0)
        self.assertClose(expect, result)
        # Index of bounds
        result = id1
        expect = torch.tensor([[0, 19], [40, 59]])
        self.assertClose(expect, result)
    
    def test_filter_spikes(self):
        result, mask = filter_spikes(x2)
        expect = y
        self.assertClose(expect, result, tol=1e-2)
