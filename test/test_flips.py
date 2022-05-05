from revert.transforms import vflip_one

import test
import torch

class TestFlips(test.TestCase):
    def test_vflip_one(self):
        x = torch.rand(6, 32)
        y = torch.tensor([0 for _ in range(6)])
        nb_chan = torch.randint(0, len(x), (1,))
        y[nb_chan] = 1

        result_x, _ = vflip_one(x, y)
        for i in range(len(y)):
            with self.subTest(i=i):
                if y[i].item() == 0:
                    self.assertClose(result_x[i], x[i])
                else:
                    self.assertClose(result_x[i], 2*x[i].mean() - x[i])
