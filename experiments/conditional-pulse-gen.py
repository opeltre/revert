import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import revert.plot as rp
import revert.cli  as cli

from revert.models import WGAN, Lipschitz, Twins, ConvNet, Affine, Linear,\
                          View, Pipe, Module, KMeans,\
                          Sum, Mask, SoftMin, Branch, Prod, Cat, Cut

dz = 8              # seed dimension
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

layers_D = [[c,  64, 8],
            [64, 16, 1],
            [8,  8,  16]]

defaults = dict(dirname   = 'cwgan-pulses',
                datatype  = 'infusion',
                data      = 'baseline-no_shunt.pt',
                output    = f'CWGAN-{dz}:64 | ns:{ns} lr:{lr} beta:{beta}.pt',
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
    encoder = twins.model[1]
    return encoder

def critic(args):
    c = 2 if args.mask else 1
    # cut codes from pulses
    cut = Cut([8, c * 64], 1)

    # map input to non-saturating domain
    scale_in = Affine(1, 1, dim=-2)
    with torch.no_grad():
        scale_in.bias = nn.Parameter(torch.tensor([.1]))
        scale_in.weight = nn.Parameter(torch.tensor([[.2]]))

    Dx = Pipe(scale_in,
              View([c, 64]),
              ConvNet(args.layers_D),
              View([8]),
              Linear(1, 1))
    
    D = Pipe(cut,
             Prod(View([8]), Dx),
             Cat(1),
             Affine(16, 1))

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
    z_gen = encoder(x_true) + dz_gen 

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

    # fit CWGAN
    gan.fit((z_gen, x_true), lr=lr[0], n_batch=args.n_batch, epochs=args.epochs)