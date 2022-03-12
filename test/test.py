import unittest
import torch

class TestCase (unittest.TestCase):

    def assertClose (self, u, v, tol=1e-6):
        if isinstance(u, torch.LongTensor):
            u = u.float()
        if isinstance(v, torch.LongTensor):
            v = v.float()
        N = torch.tensor(u.shape).prod()
        return self.assertTrue((u - v).norm() < N * tol)

main = unittest.main
