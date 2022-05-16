import torch
import re

from .module import Module


class WGAN (Module):
    """
    Wasserstein GAN.
    """

    @classmethod
    def conditional(cls, G, D, E, n_crit=100):
        """ Conditional WGAN instance. """
        return ConditionalWGAN(G, D, E, n_crit)

    def __init__(self, generator, critic, n_crit=100, lr_crit=1e-2):
        """
        Initialize WGAN from generator G and critic D. 
        """
        super().__init__()
        self.gen    = generator
        self.critic = (critic if isinstance(critic, WGANCritic) 
                             else WGANCritic(critic))
        self.n_crit  = n_crit
        self.lr_crit = lr_crit
    def loss(self, x_gen, x_true):
        """ 
        WGAN loss on returned samples x.

        The Wasserstein distance is estimated by optimising a 
        k-Lipschitz critic on the following classification reward:

            W(x_gen, x_true) = max_f { E[f(x_true)] - E[f(x_gen)] }

        The critic should therefore enforce weight clipping to respect
        the Lipschitz constraint.
        """
        device = x_true.device
        xs = torch.cat([x_gen, x_true])
        ys = torch.cat([torch.zeros(x_gen.shape), torch.ones(x_true.shape)])
        ys = ys.to(device).view([xs.shape[0], -1])
        data = [(xs.detach(), ys)]
        lr, n = self.lr_crit, self.n_crit
        self.critic.fit(data, lr=lr, epochs=n, progress=False)
        return self.critic.loss_on(*data[0])

    def forward(self, z):
        return self.gen(z)


class ConditionalWGAN(WGAN):

    def __init__(self, generator, critic, encoder, n_crit=100):
        super().__init__(generator, critic, n_crit)
        self.encoder = encoder

    def loss(self, p_gen, x_true):
        """
        Loss on generated pairs p_gen = (z, G(z)) and true samples x_true.

        The samples are encoded to pairs p_true = (E(x_true), x_true) before
        optimizing the WGAN critic loss on pairs.
        """
        with torch.no_grad():
            z_true = self.encoder(x_true)
        p_true = torch.cat([z_true.flatten(1), x_true.flatten(1)], dim=1)
        return super().loss(p_gen, p_true)

    def forward(self, z):
        """
        Return the pair (z, G(z)) as a concatenated vector.
        """
        x_gen = self.gen(z)
        return torch.cat([z.flatten(1), x_gen.flatten(1)], dim=1)


class WGANCritic (Module):
    """ 
    Wasserstein critic enforcing Lipschitz constraint by parameter clipping.
    """

    def __init__(self, critic, exclude=(r'.*\.bias$',)):
        """
        Initialize from a model and clip-exclude patterns.
        """
        super().__init__()
        self.model  = critic
        self.max    = .05
        self.exclude = exclude
        #--- register clipped parameters 
        self.clipped = []
        for name, p in self.model.named_parameters():
            if all(not re.match(e, name) for e in self.exclude):
                self.clipped.append(p)
        self.iter_callbacks.append(torch.no_grad()(self.clip))
    
    def clip(self, *xs):
        """
        Clip model weights without constraining biases.
        """
        for p in self.clipped:
            p.clip_(-self.max, self.max)

    def loss(self, fx, y):
        """ 
        Difference of expectations on true and generated data. 
        """
        true, gen   = y, 1 - y
        return ((fx * true).sum() / true.sum())\
             - ((fx * gen).sum() / gen.sum())

    def forward(self, x):
        return self.model(x)
