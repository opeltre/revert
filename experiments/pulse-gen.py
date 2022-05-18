import os
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
import revert.plot as rp
import revert.cli  as cli

from revert.models import ConvNet, Affine, Linear, View,\
                          WGAN, Twins, Pipe, Linear
dz, dx = 8, 64

ns = (20, 500)          # (n_gen, n_crit)   : respective number of iterations
lr = (1e-2, 5e-3)       # (lr_gen, lr_crit) : respective learning rates 
clip = .6               # critic.clip_value : Lipschitz constraint
n_batch = 256

args = cli.parse_args(dirname='pulse-gen',
                      output='WGAN-8:64',
                      datatype='infusion',
                      epochs=50,
                      data='baseline-no_shunt.pt')

def main (args):
    
    data = torch.load(args.data)
    #encoder = Twins.load(args.input).model
    
    # generator
    G = Pipe(Linear(8, 64),
             View([8, 8]),
             ConvNet([[8, 16, 1],
                      [8, 32, 64],
                      [4, 8]]),
             Affine(1, 1),
             View([64]))
    
    # critic
    D = Pipe(View([1, 64]),
             ConvNet([[1,  32, 8, 1],
                      [64, 16, 1, 1],
                      [8,  16, 1]]),
             View([1]))
    
    gan = WGAN(G, D, ns=ns, lr_crit=lr[1], clip=clip)

    # (seed, sample) pairs
    pulses = (nn.AvgPool1d(2)(data['pulses'])
                .view([-1, 64]))
    
    N = (pulses.shape[0] // n_batch) * n_batch
    idx = torch.randperm(N)

    x_true = (pulses[:N][idx]
                .reshape([-1, n_batch, 64])
                .cuda())

    z_gen = torch.randn(N // n_batch, n_batch, 8).cuda()
    
    dset = [(zi, xi) for zi, xi in zip(z_gen, x_true)]
    
    def fit ():
        # Optimize WGAN loss
        print(f"Fitting WGAN on {N} pulses")
        gan.writer = SummaryWriter(args.writer)
        gan.cuda()
        gan.fit(dset, lr=lr[0], epochs=args.epochs, tag="critic score")
        gan.save(args.output)

    @gan.epoch
    def plot_samples(tag, data, epoch):
        if epoch % 10 != 9:
            return "skipped"
        z = torch.randn(12, 8).cuda()
        with torch.no_grad():
            x_gen = gan(z).cpu()
        fig = rp.pulses(x_gen)
        gan.writer.add_figure(f"Samples at epoch {epoch + 1}", fig)

    fit()
