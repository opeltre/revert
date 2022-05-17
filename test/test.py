import unittest
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from revert import cli

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

def skipFit (name, testinfo=""):
    def skip(unit):
        def wrap_unit(self):
            print("\n" + "-" * 12 + f" Fitting {name} : {testinfo} " + "-" * 12)
            path = cli.join_envdir("REVERT_LOGS", f"test/{name}")
            unit(self, SummaryWriter(path))
        return unittest.skipUnless(fit, f"optional {name} fit")(wrap_unit)
    return skip

main = unittest.main
