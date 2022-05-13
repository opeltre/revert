import test
import unittest
import torch

from revert.models import WGAN, WGANCritic, ConvNet, View

N = 100
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

    @unittest.skip("optional WGAN fit")
    def test_wgan_fit(self):
        N, Nb = 200, 40
        gan = WGAN(G, D)
        gan.n_critic = 100
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
