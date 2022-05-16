import unittest
import torch
import os

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

fit  = "REVERT_TEST_FIT" in os.environ and os.environ["REVERT_TEST_FIT"]
main = unittest.main
