import test
import torch

import revert.models as rm

# hyper parameters
lr = (5e-3, 5e-3)
ns = (5, 500)
epochs = 60

# 2d-gaussian seed (+ mix described by 1d-uniform seed)
dz = 2

# 3d-bipartite gaussian mixture 
c = torch.tensor([[-1., 1., 0.], [1., 1., 0.]])
n = torch.randn(1024, 2, 3) * torch.tensor([.1, .1, .001])
x_true = (c + n).view([-1, 3])

# Linear mapping of gaussian mixture input
G = rm.Linear(2, 3)

# Non-linear classifier
D = rm.Pipe(rm.View([3, 1]),
            rm.ConvNet([[3, 9, 1],
                        [1, 1, 1],
                        [1, 1]]),
            rm.View([1]))

class TestWGANMixture (test.TestCase):

    def test_wgan_mix(self):
        # mix = (n_centers, dz)
        gan = rm.WGAN(G, D, mix=(2, dz))
        res = tuple(gan.centers.shape)
        self.assertTrue(res == (2, 2))
        # seed is 2d-normal + 1d-uniform
        seed = gan.seed(128, dz)
        res = tuple(seed.shape)
        self.assertTrue(res == (128, dz + 1))
        # map seed to 3d-mixture
        x_gen = gan(seed)
        res = tuple(x_gen.shape)
        self.assertTrue(res == (128, 3))

    @test.skipFit("WGAN-mix", "3d-embedding of 2d-gaussian mixture")
    def test_wgan_mix_fit(self, writer):
        gan = rm.WGAN(G, D, ns=ns, lr_crit=lr[1], mix=(2, dz))
        gan.writer = writer
        # random seeds
        z_gen = gan.seed(x_true.shape[0], dz)
        idx = torch.randperm(z_gen.shape[0])
        # fit
        dset = (z_gen[idx], x_true[idx])
        gan.fit(dset, lr=lr[0], epochs=epochs, n_batch=256, tag="critic score", mod=5)
        # prototypes
        c_gen = gan.gen(gan.centers)
        res = tuple(c_gen.shape)
        self.assertTrue(res == (2, 3))
        print(c_gen)
        cdist = torch.cdist(c, c_gen)
        idx = cdist.argmin(1)
        self.assertClose(c, c_gen[idx], tol=.1)
