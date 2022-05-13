import test
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

    def test_wgan_losses(self):
        gan = WGAN(G, View([1]) @ D)
        #--- critic loss:    E[c(x_true)] - E[c(x_gen)]
        c_loss = gan.critic.loss_on(xs, ys)
        #--- generator loss: E[c(x_true)] - E[gen(z)]
        w_loss = gan.loss_on(z, x_true)
        self.assertTrue(c_loss.dim() == 0)
        self.assertTrue(w_loss.dim() == 0)
