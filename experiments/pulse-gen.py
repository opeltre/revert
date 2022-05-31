import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import revert.plot as rp
import revert.cli  as cli

from revert.models import WGAN, Lipschitz, Twins, ConvNet, Affine, Linear,\
                          View, Pipe, Module, KMeans,\
                          Sum, Mask, SoftMin, Branch, Prod, Cat

dz = 16             # seed dimension
dx = 64             # pulse dimension
ns = (5, 300)       # (n_gen, n_crit)   : respective number of iterations
lr = (3e-4, 3e-4)   # (lr_gen, lr_crit) : respective learning rates
n_batch = 512
beta = 3

mask = False
c = 1 if not mask else 2

layers_G = [[16, 64, c],
            [8,  64, 64],
            [4,  4,  4]]

layers_D = [[c,  64, 1],
            [64, 16, 1],
            [8,  8,  16]]


defaults = dict(dirname   = 'wgan-pulses',
                datatype  = 'infusion',
                data      = 'baseline-no_shunt.pt',
                output    = f'WGAN-{dz}:64 | ns:{ns} lr:{lr} beta:{beta}.pt',
                input     = '../twins/VICReg-64:8:64-may12-1.pt',
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

def critic(args):
    c = 2 if args.mask else 1
    # vicreg encoder
    twins   = Twins.load(args.input).freeze()
    encoder = twins.model.module1
    # map input to non-saturating domain
    scale_in = Affine(1, 1, dim=-2)
    with torch.no_grad():
        scale_in.bias = nn.Parameter(torch.tensor([.1]))
        scale_in.weight = nn.Parameter(torch.tensor([[.2]]))

    D = Pipe(scale_in,
             View([c, 64]),
             ConvNet(args.layers_D, activation=F.leaky_relu),
             View([1]),
             Linear(1, 1))

    D = Lipschitz(D, args.beta)
    D.scale_in = D.model.module0
    return D

@cli.args(**defaults)
def main (args, runfit=True):

    # create gan
    G = generator(args)
    D = critic(args)
    gan = WGAN(G, D, ns=ns, lr_crit=lr[1])
    gan.to(args.device)

    # load seed, pulse pairs
    dset = dataset(args)

    # writers + number of critic iterations
    gan.N_crit = 0
    gan.write_to(args.writer)
    gan.critic.writer = gan.writer
    gan.write_dict({
        "gen": f"{gan.gen}",
        "critic": f"{gan.critic}",
        "layers_G": f"{args.layers_G}",
        "layers_D": f"{args.layers_D}"
    })

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
            mult = gan.critic.model.modules[-1].weight.abs().detach()
        gan.write("1. Wasserstein estimate", W, gan.N_crit)
        gan.write("2. Critic/Lipschitz estimate", k, gan.N_crit)
        gan.write("2. Critic/Optimality", W / (args.beta * k), gan.N_crit)
        # saturation + collapse
        gan.write("2. Critic/saturation", fx.abs().mean() / mult, gan.N_crit)
        gan.write("2. Critic/deviation", fx.std([0]).mean() / mult, gan.N_crit)
        gan.N_crit += 1

    @gan.epoch
    def log_gen(tag, data, epoch):
        """
        Generated pulses, masks and prototypes.
        """
        z = args.std_z * torch.randn(16, dz).to(args.device)
        #mixt, conv = gan.gen[0], gan.gen[1]
        conv = gan.gen[0]
        with torch.no_grad():
            x_gen = gan(z)
            #x_ctr = conv(mixt.centers).detach().cpu()
            gx = conv[:-1](z).cpu()
            fx = gan.critic(x_gen).cpu()
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

    # fit
    if runfit: fit(gan, dset, args.epochs)

    # save
    gan.critic.episode_callbacks = []
    gan.save(args.output)

    return gan


def fit (gan, dset, epochs):
    #fit_linear(gan, dset)
    fit_critic(gan, dset)
    print(f"\n------ Fitting WGAN on {dset[1].shape[0]} pulses ------\n")
    gan.fit(dset, lr=lr[0], epochs=epochs, tag="critic score")


def fit_critic(gan, dset, n=10):
    """ Optimize critic on the initial generator distribution. """

    # generate initial distribution
    z_gen, x_true = dset
    with torch.no_grad():
        x_gen  = gan(z_gen)
    x = torch.cat([x_gen, x_true])
    y = gan.critic.label(x_gen, x_true)

    # estimate Lipschitz constant
    N = 2096
    idx = torch.randint(x.shape[0], (N,))
    with torch.no_grad():
        fx = gan.critic(x[idx])
        gan.critic.lipschitz(fx, x[idx])

    # fit on initial distribution
    print(f"\n------ Initialize critic for {n} epochs ------\n")
    y = gan.critic.label(x_gen, x_true)
    gan.critic.fit((x, y), lr=1e-3, epochs=n, tag="critic initialization")

    with torch.no_grad(): W = gan.critic.wasserstein_on(x[idx], y[idx])
    print(f'\tWasserstein estimate : {W:.3f}')
    print(f'\tLipschitz estimate   : {gan.critic.k:.3f}')


def fit_linear(gan, dset):
    """ Initialize generator prototypes with K-Means. """

    x_true = dset[1]
    km = KMeans(dz, dx).to(x_true.device)

    # fit kmeans
    print(f"\n------ Initialize prototypes with K-Means ------\n")
    print(x_true.shape)
    km.init(x_true[:1024,0])
    km.fit((x_true[:,0],), epochs=30, n_batch=1024, tag="Prototypes initialization (KMeans)")

    # assign prototypes
    with torch.no_grad():
        gan.gen.centers = nn.Parameter(km.centers.t())

    # plot prototypes
    fig = rp.pulses(km.centers.detach().cpu())
    gan.writer.add_figure("5. Generator prototypes", fig, global_step=0)


def dataset (args):
    """ Return seed, pulse pairs (z_gen, x_true). """

    # serialized pulses and masks
    data  = torch.load(args.data)

    # true pulses and segmentation masks
    pulses = (nn.AvgPool1d(2)(data['pulses'])
                .view([-1, 64]))
    masks = (nn.AvgPool1d(2)(data['masks'])
                .view([-1, 64]))

    x_true = (torch.stack([pulses, masks], dim=1).view([-1, 2, 64])
                if args.mask else pulses.view([-1, 1, 64]))

    N = x_true.shape[0]
    idx = torch.randperm(N)
    x_true = x_true[idx].to(args.device)

    # normalize
    x_true[:,0] -= x_true[:,0].mean()
    x_true[:,0] /= x_true[:,0].std()

    # seeds
    z_gen = torch.randn(N, dz).to(args.device)
    z_gen *= args.std_z

    return (z_gen, x_true)
