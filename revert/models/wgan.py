import torch
import re

from .module import Module


class WGAN (Module):
    """
    Wasserstein GAN.
    """

    def __init__(self, generator, critic, n_crit=100):
        """
        Initialize WGAN from generator G and critic D. 
        """
        super().__init__()
        self.gen    = generator
        self.critic = (critic if isinstance(critic, WGANCritic) 
                             else WGANCritic(critic))
        self.n_crit = 100

    def loss(self, x_gen, x_true):
        """ 
        WGAN loss on returned samples x.

        The Wasserstein distance is estimated by optimising a 
        k-Lipschitz critic on the following classification reward:

            W(x_gen, x_true) = max_f { E[f(x_true)] - E[f(x_gen)] }

        The critic should therefore enforce weight clipping to respect
        the Lipschitz constraint.
        """
        xs = torch.cat([x_gen, x_true])
        ys = torch.cat([torch.zeros(x_gen.shape), torch.ones(x_true.shape)])
        ys = ys.view([xs.shape[0], -1])
        data = [(xs.detach(), ys)]
        self.critic.fit(data, lr=1e-3, epochs=self.n_crit, progress=False)
        return self.critic.loss_on(*data[0])

    def forward(self, z):
        return self.gen(z)


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
        self.iter_callbacks.append(self.clip)
    
    def clip(self, *xs):
        """
        Clip model weights without constraining biases.
        """
        for name, p in self.named_parameters():
            bounds = torch.tensor([-self.max, self.max])
            p = torch.min(torch.max(p, bounds[0]), bounds[1])

    def loss(self, fx, y):
        """ 
        Difference of expectations on true and generated data. 
        """
        true, gen   = y, 1 - y
        return ((fx * true).sum() / true.sum())\
             - ((fx * gen).sum() / gen.sum())

    def forward(self, x):
        return self.model(x)
