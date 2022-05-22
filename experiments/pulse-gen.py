import os
import torch
import torch.nn as nn

import revert.plot as rp
import revert.cli  as cli

from revert.models import WGAN, Lipschitz, Twins, ConvNet, Affine, Linear,\
                          View, Pipe, Module, KMeans,\
                          Sum, Mask, SoftMin, Branch, Prod, Cat

dz = 8              # seed dimension
dx = 64             # pulse dimension
ns = (10,   300)    # (n_gen, n_crit)   : respective number of iterations
lr = (1e-3, 1e-3)   # (lr_gen, lr_crit) : respective learning rates 
n_batch = 256
beta = .5

mask = False
c = 1 if not mask else 2

layers_G = [[8,  32, 16,  c],
            [16, 32, 64, 64],
            [4,  8, 2]]

layers_D = [[c,  32, 16, 1],
            [64, 16, 1, 1],
            [8,  16, 1]]

args = cli.parse_args(dirname   = 'wgan-pulses',
                      datatype  = 'infusion',
                      data      = 'baseline-no_shunt.pt',
                      output    = f'WGAN-2x8:64 | ns:{ns} lr:{lr} beta:{beta}.pt',
                      input     = '../twins/VICReg-64:8:64-may12-1.pt',
                      epochs    = 20,
                      n_batch   = n_batch,
                      beta      = beta,
                      device    = 'cuda:0',
                      std_z     = .1,
                      mask      = mask,
                      layers_G  = layers_G,
                      layers_D  = layers_D)

class MaskMul(Module):
    """ Multiply first channel by masks."""
    def forward(self, x):
        x_mask = x.prod(1).unsqueeze(1)
        return torch.cat([x_mask, x[:,1:]], dim=1)  


def generator(args):

    c = 2 if args.mask else 1

    scale_out = Affine(c, c, dim=-2)
    
    mixt = Pipe(Mask(8),
                SoftMin(0, True),
                Linear(dz, dz * 64),
                View([dz, 64]))

    conv = Pipe(Linear(dz, 64),
                View([8, 8]),
                ConvNet(args.layers_G),
                scale_out)
    
    G = Pipe(Branch(2),
             Prod(mixt, conv),
             Cat(1),
             Sum(1),
             View([c, dx]))
    
    G.centers = mixt[2].weight
    G.conv = conv
    G.mixt = mixt
    return G


def critic(args): 
    c = 2 if args.mask else 1
    # vicreg encoder
    twins   = Twins.load(args.input).freeze()
    encoder = twins.model.module1
    # map input to non-saturating domain
    scale_in = Affine(1, 1, dim=-2)
    with torch.no_grad():
        scale_in.bias = nn.Parameter(torch.tensor([0.01]))
        scale_in.weight = nn.Parameter(torch.tensor([[.2]]))
    """
    D = Pipe(encoder.freeze(), 
             ConvNet([[8, 16, 1],
                      [1,  1, 1],
                      [1, 1]]),
             View([1]))
    """
    D = Pipe(scale_in,
             View([c, 64]),
             ConvNet(args.layers_D),
             View([1]),
             Linear(1, 1))

    D = Lipschitz(D, args.beta)
    D.scale_in = D.model.module0
    return D


def main (args):

    G = generator(args)
    D = critic(args)
    gan = WGAN(G, D, ns=ns, lr_crit=lr[1])
    print(f'\ngan:\n {gan}')

    # serialized pulses and masks
    data  = torch.load(args.data)

    # true pulses
    pulses = (nn.AvgPool1d(2)(data['pulses'])
                .view([-1, 64]))
    masks = (nn.AvgPool1d(2)(data['masks'])
                .view([-1, 64]))

    nb = args.n_batch     
    N  = (pulses.shape[0] // nb) * nb
    idx = torch.randperm(N)

    if args.mask:
        x_true = (torch.stack([pulses[:N], masks[:N]], dim=1) 
                    [idx]
                    .reshape([-1, nb, 2, 64])
                    .to(args.device))
        x_true -= x_true.mean()
        x_true /= x_true.std()

    else:
        x_true = (pulses[:N][idx]
                    .reshape([-1, nb, 1, 64])
                    .to(args.device))
        x_true[:,0] -= x_true[:,0].mean()
        x_true[:,0] /= x_true[:,0].std()

    # seeds
    z_gen = torch.randn(N // nb, nb, dz).to(args.device)
    z_gen *= args.std_z

    dset = [(zi, xi) for zi, xi in zip(z_gen, x_true)]
    
    gan.to(args.device)
    gan.write_to(args.writer)

    def fit_critic(n=2):
        print(f"\n------ Initialize critic on {n} epochs ------\n")
        gan.critic.writer = gan.writer
        with torch.no_grad():
            x_gen  = gan(z_gen.view([-1, dz]))
        x = torch.cat([x_gen, x_true.view(x_gen.shape)])
        y = (torch.tensor([0., 1.]).repeat_interleave(N)
                                    .view([2 * N, 1])
                                    .to(args.device))
        with torch.no_grad():
            fx = gan.critic(x)
            gan.critic.lipschitz(fx, y, x)
        gan.critic.fit((x, y), lr=1e-3, epochs=n, tag="critic initialization")
        with torch.no_grad():
            W = gan.critic.wasserstein_on(x, y)
        print(f'\tWasserstein estimate : {W:.3f}')
        print(f'\tLipschitz estimate   : {gan.critic.k:.3f}')
    
    def fit_linear(samples=10 * nb):
        km = KMeans(dz).to(args.device)
        print(f"\n------ Fitting WGAN on {N} pulses ------\n")
        km.fit((x,), n_batch=1024, tag="Prototypes initialization (KMeans)")
        with torch.no_grad():
            gan.gen.centers = nn.Parameter(km.centers.T.detach())
    
    def fit (n=args.epochs):
        print(f"\n------ Fitting WGAN on {N} pulses ------\n")
        gan.fit(dset, lr=lr[0], epochs=n, tag="critic score")
        gan.critic.episode_callbacks = []
        gan.save(args.output)
    
    # number of critic iterations
    gan.N_crit = 0

    @gan.critic.episode
    def log_critic(tag, data):
        d = data['train']
        x, y = d.dataset[0] if isinstance(d, torch.utils.data.DataLoader) else d[0]
        with torch.no_grad():
            fx = gan.critic(x)
            W = gan.critic.wasserstein_on(x, y)
            mult = gan.critic.model.modules[-1].weight.abs().detach()
        gan.write("1. Wasserstein estimate", W, gan.N_crit)
        gan.write("2. Critic/Lipschitz estimate", gan.critic.k, gan.N_crit)
        # saturation + collapse
        gan.write("2. Critic/saturation", fx.abs().mean() / mult, gan.N_crit)
        gan.write("2. Critic/deviation", fx.std([0]).mean() / mult, gan.N_crit)
        gan.N_crit += 1
    
    @gan.epoch
    def log_gen(tag, data, epoch):
        z = args.std_z * torch.randn(16, dz).to(args.device)
        with torch.no_grad():
            x_gen = gan(z)
            gx = gan.gen.conv(z).cpu()
            fx = gan.critic(x_gen).cpu()
            c = nn.functional.softmin(fx, 0)
        # saturation + collapse
        gan.write("3. Generator/saturation", gx.abs().mean(), epoch)
        gan.write("3. Generator/deviation", gx.std([0]).mean(), epoch)
        # plot pulses
        fig = rp.pulses(x_gen[:,0].cpu(), c=c)
        gan.writer.add_figure(f"4. Generated pulses", fig, global_step=epoch)
        if args.mask:
            fig2 = rp.pulses(x_gen[:,1].cpu(), c=c)
            gan.writer.add_figure(f"5. Generated masks", fig2, global_step=epoch)
    
    fit_critic()
    fit_linear(10)
    fit()
