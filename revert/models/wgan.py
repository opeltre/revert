import torch
import re

from .module import Module


class WGAN (Module):
    """
    Wasserstein GAN.
    """

    @classmethod
    def conditional(cls, G, D, E, *args, **kargs):
        """ Conditional WGAN instance. """
        return ConditionalWGAN(G, D, E, *args, **kargs)

    def __init__(self, generator, critic, ns=(10, 1000), lr_crit=5e-3, clip=1):
        """
        Initialize WGAN from generator G and critic D. 
        """
        super().__init__()
        self.gen    = generator

        self.critic = (critic if isinstance(critic, WGANCritic) 
                             else WGANCritic(critic))
        self.critic.clip_value = clip 

        n_gen, n_crit = ns
        self.n_gen   = n_gen 
        self.n_crit  = n_crit
        self.lr_crit = lr_crit
    
    def fit(self, dset, **kws):
        """
        Fit on a dataset of (z_gen, x_true) of seed-sample pairs.
        """
        N, n_gen, n_crit = len(dset), self.n_gen, self.n_crit
        cfit = torch.zeros([N], dtype=torch.long)
        idx = torch.arange(N // n_gen) * n_gen
        cfit[idx] = n_crit
        dset_c = [(*xs, c) for xs, c in zip(dset, cfit)]
        super().fit(dset_c, **kws)
    
    def loss(self, x_gen, x_true, fit_critic=True):
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
        lr = self.lr_crit
        # critic optimization steps
        if fit_critic == True:
            fit_critic = self.n_crit
        #--- maximize expectation difference E[f(x_true)] - E[f(x_gen)] over f
        if fit_critic:
            self.critic.train(True)
            self.critic.fit(data, lr=lr, epochs=fit_critic, progress=False)
            self.critic.train(False)
        #--- maximize expectation E[f(x_gen)] over x_gen
        self.critic.train(False)
        return - self.critic(xs).mean()

    def forward(self, z):
        """ 
        Generate samples from seeds. 
        """
        return self.gen(z)

    def parameters(self):
        """
        Yield generator parameters.
        """
        return self.gen.parameters()

    def named_parameters(self):
        """
        Yield generator parameters.
        """
        return self.gen.named_parameters()


class ConditionalWGAN(WGAN):
    """
    Conditional Wasserstein GAN. 
    """

    def __init__(self, generator, critic, encoder, *args, **kargs):
        super().__init__(generator, critic, *args, **kargs)
        self.encoder = encoder

    def loss(self, p_gen, x_true, fit_critic=True):
        """
        Loss on generated pairs p_gen = (z, G(z)) and true samples x_true.

        The samples are encoded to pairs p_true = (E(x_true), x_true) before
        optimizing the WGAN critic loss on pairs.
        """
        with torch.no_grad():
            z_true = self.encoder(x_true)
        p_true = torch.cat([z_true.flatten(1), x_true.flatten(1)], dim=1)
        return super().loss(p_gen, p_true, fit_critic)

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

    def __init__(self, critic, exclude=(), clip=1):
        """
        Initialize from a model and clip-exclude patterns.

        Input:
            - critic:   nn.Module
            - exclude:  tuple(str)  e.g.(r'.*\.bias$',)
        """
        super().__init__()
        self.model  = critic
        self.exclude = exclude
        self.clip_value = clip
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
        k = self.clip_value
        for p in self.clipped:
            p.clip_(-k, k)

    def loss(self, fx, y):
        """ 
        Difference of expectations on true and generated data. 
        """
        true, gen   = y, 1 - y
        return - ((fx * true).sum() / true.sum())\
               + ((fx * gen).sum() / gen.sum())

    def forward(self, x):
        return self.model(x)
