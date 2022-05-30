import test
import unittest
import torch
import sys

from revert.models import WGAN, WGANCritic, Clipped,\
                          ConvNet, View, Affine

ns = (5, 100)  # (n_gen, n_crit)
lr_gen, lr_crit = (5e-3, 5e-3)
clip = .5
N, Nb = 512, 256
epochs = 5
tag = "critic score"

dx, dz = 6, 3
G = Affine(3, 6)
D = View([1]) @ ConvNet([[dx, 12, 1],  
                         [1,  1, 1], 
                         [1,  1]])    @ View([dx, 1])

#--- hyperplane 
x_true = torch.randn([N, dx])
x_true += (.2 - x_true.mean())
#--- generated distribution
z = torch.randn([N, dz])
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
        D1 = Clipped(D)
        gan1 = WGAN(G, D1)
        #--- Critic
        self.assertTrue(isinstance(gan1.critic, WGANCritic))
        self.assertTrue(gan1.critic is D1)
        self.assertTrue(len(D1.clipped) > 0)
    
    def test_wgan_losses(self):
        gan = WGAN(G, D)
        #--- critic loss:    E[c(x_true)] - E[c(x_gen)]
        with torch.no_grad():
            c_loss = gan.critic.loss_on(xs, ys)
        #--- generator loss: E[c(x_true)] - E[gen(z)]
        w_loss = gan.loss_on(z, x_true)
        self.assertTrue(c_loss.dim() == 0)
        self.assertTrue(w_loss.dim() == 0)

    @test.skipFit("WGAN", "6d-embedding of a 3d-gaussian")
    def test_wgan_fit(self, writer):
        gan = WGAN(G, D, ns, lr_gen)
        gan.writer = writer
        gan.to('cuda')
        #--- input 3d-seeds
        z = torch.randn([N, Nb, 3]).to('cuda')
        #--- observed 3d-distribution
        x = .3 + (.8 * torch.randn([N * Nb, 6]))
        P = torch.tensor([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]]).float()
        proj = P.T @ P
        x_true = (proj @ x.T).T.view([N, Nb, 6]).to('cuda')
        #--- backprop 
        dset =  [(zi, xi) for zi, xi in zip(z, x_true)]
        print(f"\n\tn_gen: {gan.n_gen} \tlr_gen: {lr_gen}")
        print(f"\tn_crit: {gan.n_crit} \tlr_crit: {gan.lr_crit}")
        gan.fit(dset, lr=lr_gen, epochs=epochs, progress=True, tag=tag)
        #--- generate
        with torch.no_grad():
            x_gen = gan(z.view([-1, 3]))
        print(f"\n\t=> x_gen.mean : {x_gen.mean([0]).flatten().cpu()}")
        print(f"\t   x_gen.std  : {x_gen.std([0]).flatten().cpu()}")
        #--- test mean ([.3, .3, .3, 0., 0., 0.])
        expect = x_gen.mean()
        result = x_true.mean()
        self.assertClose(expect, result, tol=.1)
        #--- test subspace (proj @ x = x)
        x_gen = x_gen.view([-1, 6]).to('cpu')
        expect = proj @ x_gen.T
        result = x_gen.T
        self.assertClose(expect, result, tol=.1)
        gan.to('cpu')
