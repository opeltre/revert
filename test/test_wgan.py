import test
import unittest
import torch
import sys

from revert.models import WGAN, WGANCritic, ConvNet, View, Affine

N, Nb = 200, 256
dx, dz = 6, 3
G = ConvNet([[dz, dx], [1, 1], [1]])
D = ConvNet([[dx, 1],  [1, 1], [1]])

#--- hyperplane 
x_true = torch.randn([N, dx, 1])
x_true += (.2 - x_true.mean())
#--- generated distribution
z = torch.randn([N, dz, 1])
x_gen = G(z)
#--- labels
xs = torch.cat([x_gen, x_true])
ys = (torch.tensor([0, 1])
        .repeat_interleave(N)
        .flatten())

options     = sys.argv[1] if len(sys.argv) > 1 else ''

class TestWGAN(test.TestCase):

    def test_wgan(self):
        gan = WGAN(G, D)
        self.assertTrue(gan.gen is G)
        self.assertClose(gan(z), x_gen)

    def test_wgan_critic(self):
        D1 = WGANCritic(View([1]) @ ConvNet([[dx, 1], [1, 1], [1]]))
        gan1 = WGAN(G, D1)
        #--- Critic
        self.assertTrue(isinstance(gan1.critic, WGANCritic))
        self.assertTrue(gan1.critic is D1)
        self.assertTrue(len(D1.clipped) > 0)
    
    def test_wgan_losses(self):
        gan = WGAN(G, View([1]) @ D)
        #--- critic loss:    E[c(x_true)] - E[c(x_gen)]
        c_loss = gan.critic.loss_on(xs, ys)
        #--- generator loss: E[c(x_true)] - E[gen(z)]
        w_loss = gan.loss_on(z, x_true)
        self.assertTrue(c_loss.dim() == 0)
        self.assertTrue(w_loss.dim() == 0)

    @unittest.skipUnless(test.fit, "optional WGAN fit")
    def test_wgan_fit(self):
        gan = WGAN(G, D)
        gan.n_crit = 100
        #--- input 3d-seeds
        z = torch.randn([N, Nb, 3, 1])
        #--- observed 3d-distribution
        x = .3 + torch.randn([N * Nb, 6])
        P = torch.tensor([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]]).float()
        proj = P.T @ P
        x_true = (proj @ x.T).T.view([N, Nb, 6, 1])
        #--- backprop 
        dset =  [(zi, xi) for zi, xi in zip(z, x_true)]
        gan.fit(dset, lr=1e-3, epochs=1, progress=False)
        #--- generate
        with torch.no_grad():
            x_gen = gan(z.view([-1, 3, 1]))
        #--- test mean ([.3, .3, .3, 0., 0., 0.])
        expect = x_gen.mean()
        result = x_true.mean()
        self.assertClose(expect, result, tol=.1)
        #--- test subspace (proj @ x = x)
        x_gen = x_gen.view([-1, 6])
        expect = proj @ x_gen.T
        result = x_gen.T
        self.assertClose(expect, result, tol=.1)


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

    @unittest.skipUnless(test.fit, 'optional CWGAN fit')
    def test_conditional_wgan_fit(self):
        N, Nb = 200, 256
        E = lambda x: x[:,1:]
        G = Affine(1, 2)
        D = View([1]) @ ConvNet([[3, 3, 1], [1, 1, 1], [1, 1]]) @ View([3, 1])
        cgan = WGAN.conditional(G, D, E)
        #--- 2d-gaussian with (.3, .3) mean and (.05, .1) stdev
        mean = torch.tensor([.3, .3], device="cuda")
        devs = torch.tensor([.05, .1], device="cuda")
        x_true = mean + (torch.randn([N, Nb, 2], device="cuda") * devs)
        #--- 1d-gaussian codes with .3 mean and .1 stdev
        z = .1 + (torch.randn([N, Nb, 1], device="cuda") * .3)
        z0 = z + 0
        #--- fit on dataset of (code, sample) pairs
        dset = [(zi, xi) for zi, xi in zip(z, x_true)]
        cgan.cuda()
        print(f"\nFitting conditional WGAN on {N} batches of size {Nb}:")
        cgan.n_crit  = 100
        cgan.lr_crit = 1e-2
        print(f"\ncritic: \tn_it = {cgan.n_crit} \tcritic lr = {cgan.lr_crit}")
        cgan.fit(dset, lr=1e-3, epochs=5, progress=True)
        #--- generate
        z = z.view([-1, 1])
        with torch.no_grad():
            x_gen = cgan.gen(z)
        #--- check section consistency
        self.assertClose(z, z0.view([-1, 1]))
        self.assertClose(z, E(x_gen), tol=.1)
        #--- check mean and support
        print(f"\nmean  x_gen: {x_gen.mean([0])}")
        print(f"\nstdev x_gen: {x_gen.std([0])}")
        self.assertClose(z.mean([0]), E(x_true.view([-1, 1])).mean([0]), tol=.1)
        self.assertClose(x_gen.mean([0]), mean, tol=.1)
        self.assertClose(x_gen[:,0].std([0]), 0, tol=.1)
