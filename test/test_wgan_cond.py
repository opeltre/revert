import test
import torch
import unittest

from revert.models import WGAN, Affine, ConvNet, View

ns = (10, 500)
lr_gen, lr_crit = (1e-2, 5e-3)

N, Nb = 512, 512 
epochs = 5
tag = "critic score"

class TestCWGAN(test.TestCase):

    def test_conditional_wgan(self):
        E = lambda x: x[1]
        G = View([2]) @ ConvNet([[1, 2], [1, 1], [1]])
        D = View([1]) @ ConvNet([[3, 1], [1, 1], [1]])
        cgan = WGAN.conditional(G, D, E)
        self.assertTrue(cgan.encoder == E)
        z = torch.randn([10, 1, 1])
        p_gen = cgan(z)
        x_gen = cgan.gen(z)
        self.assertTrue(tuple(p_gen.shape) == (10, 3))
        self.assertTrue(tuple(x_gen.shape) == (10, 2))
        self.assertClose(p_gen, 
                        torch.cat([z.flatten(1), x_gen.flatten(1)], dim=1))

    @test.skipFit('CWGAN', '2d-embedding of a 1d-gaussian')
    def test_conditional_wgan_fit(self, writer):
        E = lambda x: x[:,1:]
        G = Affine(1, 2)
        D = View([1]) @ ConvNet([[3, 12, 1], [1, 1, 1], [1, 1]]) @ View([3, 1])
        cgan = WGAN.conditional(G, D, E, ns, lr_crit)
        cgan.writer = writer
        #--- 2d-gaussian with (.3, .3) mean and (.05, .1) stdev
        mean = torch.tensor([.3, .3], device="cuda")
        devs = torch.tensor([.05, .1], device="cuda")
        x_true = mean + (torch.randn([N, Nb, 2], device="cuda") * devs)
        #--- 1d-gaussian codes with .3 mean and .1 stdev
        z = mean[1] + (torch.randn([N, Nb, 1], device="cuda") * devs[1])
        #--- fit on dataset of (code, sample) pairs
        dset = [(zi, xi) for zi, xi in zip(z, x_true)]
        cgan.cuda()
        print(f"\n\tn_gen = {cgan.n_gen} \tlr_gen = {lr_gen}")
        print(f"\tn_crit = {cgan.n_crit} \tlr_crit = {cgan.lr_crit}\n")
        cgan.fit(dset, lr=lr_gen, epochs=epochs, progress=True, tag=tag)
        #--- generate
        z = z.view([-1, 1])
        with torch.no_grad():
            x_gen = cgan.gen(z)
        #--- check section consistency
        self.assertClose(z, E(x_gen), tol=.1)
        #--- check mean and support
        print(f"\n\t=> x_gen.mean : {x_gen.mean([0])}")
        print(f"\t   x_gen.std  : {x_gen.std([0])}")
        self.assertClose(z.mean([0]), E(x_true.view([-1, 1])).mean([0]), tol=.1)
        self.assertClose(x_gen.mean([0]), mean, tol=.1)
        self.assertClose(x_gen[:,0].std([0]), 0, tol=.1)
