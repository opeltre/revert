import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import revert.plot as rp
import revert.cli  as cli

from revert.models import WGAN, Lipschitz, Twins, ConvNet, Affine, Linear,\
                          View, Pipe, Module, KMeans,\
                          Sum, Mask, SoftMin, Branch, Prod, Cat, Cut

dz = 16             # seed dimension
dx = 64             # pulse dimension
ns = (5, 200)       # (n_gen, n_crit)   : respective number of iterations
lr = (3e-4, 3e-4)   # (lr_gen, lr_crit) : respective learning rates
n_batch = 512
beta = 3

mask = False
c = 1 if not mask else 2

layers_G = [[16, 64, c],
            [8,  64, 64],
            [4,  4,  4]]

layers_D = [[c,  64, dz],
            [64, 16, 1],
            [8,  8,  16]]

defaults = dict(dirname   = 'cwgan-pulses',
                datatype  = 'infusion',
                data      = 'baseline-no_shunt.pt',
                output    = f'CWGAN-{dz}:64-jul6-1',
                input     = '../twins/VICReg-64:16:64-jun3-1.pt',
                epochs    = 30,
                n_batch   = n_batch,
                beta      = beta,
                device    = 'cuda:0',
                std_z     = .1,
                mask      = mask,
                layers_G  = layers_G,
                layers_D  = layers_D)

def generator(args):

    c = 2 if args.mask else 1

    scale_out = Affine(c, c, dim=-2)

    conv = Pipe(Linear(dz, 128),
                View([16, 8]),
                ConvNet(args.layers_G),
                scale_out)

    return Pipe(conv)

def encoder(args):
    # vicreg encoder
    """
    VICReg(
      (model): Pipe(
        (module0): AvgPool1d(kernel_size=(2,), stride=(2,), padding=(0,))
        (module1): ConvNet([[1, 32, 8], [64, 8, 1], [8, 8]])
        (module2): ConvNet([[8, 32, 64], [1, 1, 1], [1, 1]])
        (module3): View([64])
      )
    )
    """
    twins   = Twins.load(args.input).freeze()
    encoder = View([dz]) @ twins.model[1]
    encoder.to(args.device)
    return encoder

def critic(args):
    c = 2 if args.mask else 1
    # cut codes from pulses
    cut = Cut([dz, c * 64], 1)

    # map input to non-saturating domain
    scale_in = Affine(1, 1, dim=-2)
    with torch.no_grad():
        scale_in.bias = nn.Parameter(torch.tensor([.1]))
        scale_in.weight = nn.Parameter(torch.tensor([[.2]]))

    Dx = Pipe(scale_in,
              View([c, 64]),
              ConvNet(args.layers_D),
              View([dz]),
              Linear(1, 1))
    
    D = Pipe(cut,
             Prod(View([dz]), Dx),
             Cat(1),
             Affine(2 * dz , 1),
             View([1]))

    D = Lipschitz(D, args.beta)
    return D

def dataset (args, encoder):
    """ Return seed, pulse pairs (z_gen, x_true). """

    # serialized pulses and masks
    data  = torch.load(args.data)

    # true pulses and segmentation masks
    pulses = (nn.AvgPool1d(2)(data['pulses'])
                .view([-1, 64]))
    masks = (nn.AvgPool1d(2)(data['masks'])
                .view([-1, 64]))

    # stack masks with pulses if args.mask == True
    x_true = (torch.stack([pulses, masks], dim=1).view([-1, 2, 64])
                if args.mask else pulses.view([-1, 1, 64]))

    N = x_true.shape[0]
    idx = torch.randperm(N)
    x_true = x_true[idx].to(args.device)

    # normalize pulses
    x_true[:,0] -= x_true[:,0].mean()
    x_true[:,0] /= x_true[:,0].std()

    # empiric codes + noise
    dz_gen = .05 * torch.randn(N, dz).to(args.device)
    z_true = encoder(x_true)
    z_gen  = z_true + dz_gen 

    return (z_gen, x_true)

@cli.args(**defaults)
def main (args, runfit=True):
    """
    Train conditional WGAN on ICP baseline pulses. 

    The critic learns to discriminate between pairs:
        (E(x_true), x_true)  -  (z_gen, G(z_gen))
    
    An optimal generator should therefore implement a section of E:
        E(G(z)) = z
    """
    # create gan
    G = generator(args)
    D = critic(args)
    E = encoder(args)
    gan = WGAN.conditional(G, D, E, ns=ns, lr_crit=lr[1])
    gan.to(args.device)

    # load seed, pulse pairs
    dset = dataset(args, E)
    z_gen, x_true = dset

    # writers
    gan.N_crit = 0
    gan.write_to(args.writer)
    gan.critic.writer = gan.writer

    @gan.critic.episode
    def log_critic(tag, data):
        """
        Wasserstein + Lipschitz estimates and optimality ratio.

        The term W[f] / (beta . k(f)^2) should tend to 1 at critic optimality.
        """
        d = data['train']
        if isinstance(d, torch.utils.data.DataLoader):
            print(f"dataloader.dataset[0]: {d.dataset[0]}")
            x, y = d.dataset[:args.n_batch]
        else:
            x, y = d[0]
        with torch.no_grad():
            fx = gan.critic(x)
            W = gan.critic.wasserstein_on(x, y)
            k = gan.critic.k
            #mult = gan.critic.model.modules[-1].weight.abs().detach()
        gan.write("1. Wasserstein estimate", W, gan.N_crit)
        gan.write("2. Critic/Lipschitz estimate", k, gan.N_crit)
        gan.write("2. Critic/Optimality", W / (args.beta * k), gan.N_crit)
        # saturation + collapse
        #gan.write("2. Critic/saturation", fx.abs().mean() / mult, gan.N_crit)
        #gan.write("2. Critic/deviation", fx.std([0]).mean() / mult, gan.N_crit)
        gan.N_crit += 1

    @gan.epoch
    def log_gen(tag, data, epoch):
        """
        Generated pulses, masks and prototypes.
        """
        #mixt, conv = gan.gen[0], gan.gen[1]
        conv = gan.gen[0]
        with torch.no_grad():
            zx_gen = gan(z_gen[12*epoch:12*(epoch+1)])
            z, x_gen = Cut([dz, 64], dim=1)(zx_gen)
            x_gen = x_gen.view([-1, 1, 64])
            #x_ctr = conv(mixt.centers).detach().cpu()
            gx = conv[:-1](z).cpu()
            fx = gan.critic(zx_gen).cpu()
            c = .5 - fx / fx.abs().max()
            #temp = mixt.temp
            #temp = temp.cpu() if isinstance(temp, torch.Tensor) else temp
        # saturation + collapse
        gan.write("3. Generator/saturation", gx.abs().mean(), epoch)
        gan.write("3. Generator/deviation", gx.std([0]).mean(), epoch)
        #gan.write("3. Generator/mix temperature", temp, epoch)
        # plot pulses
        fig = rp.pulses(x_gen[:,0].cpu(), c=c)
        gan.writer.add_figure(f"4. Generated pulses", fig, global_step=epoch)
        # plot prototypes
        """
        fig = rp.pulses(x_ctr.view([-1, 64]))
        gan.writer.add_figure("5. Generator prototypes", fig, global_step=epoch+1)
        """
        if args.mask:
            fig2 = rp.pulses(x_gen[:,1].cpu(), c=c)
            gan.writer.add_figure(f"6. Generated masks", fig2, global_step=epoch)

    # fit CWGAN
    fit_critic(gan, (z_gen, x_true), E, n=20)
    gan.fit((z_gen, x_true), lr=lr[0], n_batch=args.n_batch, epochs=args.epochs)
    # save
    gan.critic.episode_callbacks = []
    gan.save(args.output)
    return gan

def fit_critic(gan, dset, encoder, n=20):
    """ Optimize critic on the initial generator distribution. """

    # generate initial distribution
    z_gen, x_true = dset
    with torch.no_grad():
        zx_gen  = gan(z_gen)
        zx_true = torch.cat([encoder(x_true), x_true.flatten(1)], dim=1)

    zx = torch.cat([zx_gen, zx_true])
    y = gan.critic.label(zx_gen, zx_true)

    # estimate Lipschitz constant
    N = 2096
    idx = torch.randint(zx.shape[0], (N,))
    with torch.no_grad():
        fx = gan.critic(zx[idx])
        gan.critic.lipschitz(fx, zx[idx])

    # fit on initial distribution
    print(f"\n------ Initialize critic for {n} epochs ------\n")
    gan.critic.fit((zx, y), lr=1e-3, epochs=n, tag="critic initialization")

    with torch.no_grad(): W = gan.critic.wasserstein_on(zx[idx], y[idx])
    print(f'\tWasserstein estimate : {W:.3f}')
    print(f'\tLipschitz estimate   : {gan.critic.k:.3f}')
