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

    def __init__(self, generator, critic, ns=(10, 1000), lr_crit=5e-3):
        """
        Initialize WGAN from generator G and critic D.
        """
        super().__init__()
        self.gen    = generator
        self.critic = (critic if isinstance(critic, WGANCritic)
                             else Lipschitz(critic))
        n_gen, n_crit = ns
        self.n_gen   = n_gen
        self.n_crit  = n_crit
        self.lr_crit = lr_crit
        self.n_it = 0

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
        lr = self.lr_crit
        # critic optimization steps
        fit_critic = self.n_crit * (0 == self.n_it % self.n_gen)
        self.n_it += 1
        #--- maximize expectation difference E[f(x_true)] - E[f(x_gen)] over f
        if fit_critic:
            self.critic.unfreeze()
            self.critic.fit(data, lr=lr, epochs=fit_critic, progress=False)
        #--- maximize expectation E[f(x_gen)] over x_gen
        self.critic.freeze()
        with torch.no_grad():
            W = self.critic.wasserstein_on(xs, ys).detach()
        return - self.critic.loss_on(xs, ys)

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
    Wasserstein estimator.

    The critic f tries to maximize the expectation difference:

        E[f(x_true)] - E[f(x_gen)]

    When f is constrained to a Lipschitz-sphere this gives
    an estimator of W(x_gen, x_true).

    N.B: Do not use directly but use a subclass e.g. Lipschitz or Clipped.
    """

    def __init__(self, critic):
        """
        Initialize from a model and clip-exclude patterns.

        Input:
            - critic:   nn.Module
            - exclude:  tuple(str)  e.g.(r'.*\.bias$',)
        """
        super().__init__()
        self.model  = critic

    def label(self, x_gen, x_true, device=None):
        """
        Return label vector y = [0, ..., 1, ... ].

        Provide (x_gen, x_true) as tensor or int (shape[0]).
        """
        if isinstance(x_gen, torch.Tensor):
            device = x_gen.device
        n_gen  = x_gen if isinstance(x_gen, int) else x_gen.shape[0]
        n_true = x_true if isinstance(x_true, int) else x_true.shape[0]
        true = torch.ones([n_true, 1])
        gen  = torch.zeros([n_gen, 1])
        return torch.cat([gen, true]).to(device)

    def energy(self, fx, y):
        """
        Difference of expectations on true and generated data.
        """
        true, gen   = y, 1 - y
        W = ((fx * true).sum() / true.sum())\
          - ((fx * gen).sum() / gen.sum())
        return W

    def wasserstein(self, fx, y):
        """
        Difference of expectations on true and generated data.
        """
        W = self.energy(fx, y)
        return W / self.k if "k" in dir(self) else W

    def wasserstein_on(self, x, y):
        return self.wasserstein(self(x), y)

    def loss(self, fx, y):
        """
        Difference of expectations on true and generated data.
        """
        return - self.wasserstein(fx, y)

    def forward(self, x):
        return self.model(x)


class Lipschitz (WGANCritic):

    def __init__(self, critic, beta=1., temp_k=.1, n_pairs=256):
        super().__init__(critic)
        self.beta = beta
        self.k = 1
        self.n_pairs = n_pairs
        self.temp_k  = temp_k
        self.extrema  = None
        self.roll_idx = 0

    def loss(self, fx, y, x):
        """
        Expectation difference with Lipschitz penalty.

        Estimate the Lipschitz constant k(f) and add its square to the Wasserstein estimate:

                (beta/2) * k(f)^2 + E[f(x_true)] - E[f(x_gen)]

              = (beta/2) * k(f)^2 + E[f(x) * y]

        The minimum of this function at fixed beta computes the Legendre transform
        of k evaluated on the the linear form `(y * p_x) / beta`.

        If `f` is critical then `beta . k . dk = dW` so that `beta . k` is
        a Lagrange multiplier for a Lipschitz sphere constraint `k(f) = cst`.

        Evaluating on both differentials on `(1 + dt) * f`, we have by linearity
        of `W = dW` and homogeneity of the semi-norm k:

                beta . k(f)^2 . dt = W[f] . dt

        By the Kantorovitch-Rubinstein duality, the Wasserstein distance
        `DW(x_gen, x_true)` is then given by `W[f] / k(f)` for optimal f.
        The Lipschitz radius and the Wasserstein distance therefore satisfy:

                k(f) = DW(x_gen, x_true) / beta

        Checking that the ratio `DW / k` is indeed close to `beta` is a good
        way to check whether the critic had a chance to reach the optimal state.

        Remarks:
        --------
        - Because both k(f) and W(f) are fixed by addition of constants, one can add
        a constraint on the mean of f. Fixed to zero the sign of the critic can then be
        meaningful as a prediction.

        - Minimizing k(f)^2 can be viewed as a form of entropic constraint on f,
        and W + (beta/2) * k^2 is a form of free energy associated to the
        internal energy term W = E[f * y].
        """
        W = self.energy(fx, y)
        k = self.lipschitz(fx, x)
        beta = self.beta
        mean = fx.mean().abs()
        return (- W + mean + (beta / 2) * k ** 2)

    def lipschitz(self, fx, x):
        """
        Estimate the Lipschitz constant.

        Maximizers of the dfx/dx ratio are buffered
        to improve the quality of the estimator over time.
        """
        # populate buffer
        n, ns  = self.n_pairs, x.shape[1:]
        if isinstance(self.extrema, type(None)):
            idx = torch.randperm(n, device=x.device)
            self.extrema = torch.stack([x[:n], x[:n][idx]], dim=1)
        # pairwise input and output distances
        s = torch.randperm(x.shape[0])
        with torch.no_grad():
            dx = (x - x[s]).flatten(1).norm(2, [1]) + 1e-5
            dy = (fx - fx[s]).flatten(1).norm(2, [1])
            k_in = dy / dx
        # update buffer
        i = k_in.argmax()
        x_max = self.extrema
        if k_in[i] >= self.k:
            j = self.roll_idx % self.n_pairs
            with torch.no_grad():
               x_max[j, 0] = x[i]
               x_max[j, 1] = x[s[i]]
               self.extrema = x_max.detach()
               self.roll_idx += 1
        # Lipschitz ratio estimate
        y_max = self(x_max.view([2 * n, *ns])).view([n, 2])
        dx_max = (x_max[:,0] - x_max[:,1]).flatten(1).norm(2, [1])
        dy_max = (y_max[:,0] - y_max[:,1]).abs()
        k = dy_max / (dx_max + 1e-6)
        k = (k * torch.softmax(k / self.temp_k, 0)).sum()
        self.k = k.detach()
        return k

    def loss_on (self, x, y):
        fx = self(x)
        return self.loss(fx, y, x)


class Clipped (WGANCritic):

    def __init__(self, critic, exclude=(), clip=1):
        super().__init__(critic)
        self.exclude = exclude
        self.clip_value = clip
        self.k = self.clip_value
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
        k = self.clip_value
        for p in self.clipped:
            if p.requires_grad:
                with torch.no_grad():
                    p.clip_(-k, k)
