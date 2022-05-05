import unittest
import torch

class TestCase (unittest.TestCase):

    def assertClose (self, u, v, tol=1e-6):
        if isinstance(u, torch.LongTensor):
            u = u.float()
        if isinstance(v, torch.LongTensor):
            v = v.float()
        N = torch.tensor(u.shape).prod()
        dist = (u - v).norm() 
        if dist >= N * tol: 
            print(f"Error {dist / N} >= tolerance {tol}")
        return self.assertTrue(dist < N * tol)

main = unittest.main
