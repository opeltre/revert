from revert.transforms import segment, mask_center

import test
import torch

x    = torch.randn([32])
cuts = torch.tensor([0, 6, 12, 18, 24, 30])

class TestSegment(test.TestCase):

    def test_mask(self):
        mask = torch.zeros([5, 12])
        mask[:,:6] += 1
        _, result = segment(x, cuts, 12)
        expect    = mask
        self.assertClose(expect, result)

    def test_segments(self):
        out, mask = segment(x, cuts, 12)
        result = out[:,0]
        expect = x[cuts[:-1]]
        self.assertClose(expect, result)

    def test_before(self):
        out, mask = segment(x, cuts, 12, before=3)
        result = out[:,3]
        expect = x[cuts[:-1]]
        self.assertClose(expect, result)

class TestMaskCenter (test.TestCase):

    def test_mask_center(self):
        y, mask = segment(x, cuts, 12)
        z = mask_center(y, mask, output=None)
        z0, z1 = z[:,0], z[:,-1]
        self.assertClose(z0, z1)

    def test_linear(self):
        n = torch.arange(6)[:,None]
        t = torch.linspace(0, 1, 12)[None,:]
        y = n * t       
        m = torch.zeros([6, 12])
        m[:,:6] += 1
        result = mask_center(y, m)
        expect = torch.zeros([6, 12])
        self.assertClose(expect, result, tol=1e-4)
