import torch
import re

from .module import Module


class WGAN (Module):
    """
    Wasserstein GAN.
    """

    def __init__(self, generator, critic, n_crit=100):
        self.gen    = generator
        self.critic = critic
        self.n_crit = 100
        super().__init__()

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
        ys = torch.cat([torch.zeros([x_gen.shape]), torch.ones([x_true])])
        data = [(xs, ys)]
        self.critic.fit(data, lr=1e-3, epochs=self.n_crit)
        return self.critic.loss_on(data)

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
        self.critic = module
        self.max    = .05
        self.exclude = exclude
        #--- register clipped parameters 
        self.clipped = []
        for name, p in self.named_parameter():
            if all(not re.match(e, name) for e in self.exclude):
                self.clipped.append(p)
    
    @self.iter
    def clip(self, *xs):
        """
        Clip model weights without constraining biases.
        """
        for name, p in self.named_parameter():
            for pattern in self.clip_exclude():
                if re.match(pattern, self.clip_exclude)
            bounds = torch.tensor([-self.max, self.max])
            p = torch.min(torch.max(x, bounds[0]), bounds[1])

    def loss(self, fx, y):
        """ 
        Difference of expectations on true and generated data. 
        """
        true, gen   = y, 1 - y
        return ((fx * true).sum() / true.sum())\
             - ((fx * gen).sum() / gen.sum()))

    def forward(self, x):
        return self.critic(x)
