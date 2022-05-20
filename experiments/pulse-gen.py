import os
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
import revert.plot as rp
import revert.cli  as cli

from revert.models import ConvNet, Affine, Linear, View,\
                          WGAN, Twins, Pipe, Linear, Module
dz, dx = 8, 64

ns = (5, 500)           # (n_gen, n_crit)   : respective number of iterations
lr = (5e-3, 5e-3)       # (lr_gen, lr_crit) : respective learning rates 
clip = .2               # critic.clip_value : Lipschitz constraint
n_batch = 512

args = cli.parse_args(dirname='pulse-gen',
                      output=f'WGAN-16:2x64-clip:{clip}-ns:{ns}-lr:{lr}-test.pt',
                      datatype='infusion',
                      epochs=20,
                      device='cuda:0',
                      data='baseline-no_shunt.pt')

class MaskMul(Module):
    """ Multiply first channel by masks."""
    def forward(self, x):
        x_mask = x.prod(1).unsqueeze(1)
        return torch.cat([x_mask, x[:,1:]], dim=1)  


def main (args):
    
    data = torch.load(args.data)
    #encoder = Twins.load(args.input).model
    
    # generator
    G = Pipe(Linear(dz, 64),
             View([8, 8]),
             ConvNet([[8, 32, 2],
                      [8, 32, 64],
                      [4,  8]]),
             Affine(2, 2, dim=-2),
             MaskMul())
    
    # critic
    D = Pipe(Affine(2, 2, dim=-2),
             View([2, 64]),
             ConvNet([[2,  32, 16],
                      [64, 16, 4],
                      [8,  8]]),
             View([64]),
             Linear(64, 1))
    
    gan = WGAN(G, D, ns=ns, lr_crit=lr[1], clip=clip)

    # (seed, sample) pairs
    pulses = (nn.AvgPool1d(2)(data['pulses'])
                .view([-1, 64]))
    masks = (nn.AvgPool1d(2)(data['masks'])
                .view([-1, 64]))
    
    N = (pulses.shape[0] // n_batch) * n_batch
    idx = torch.randperm(N)

    x_true = (torch.stack([pulses[:N], masks[:N]], dim=1)
                [idx]
                .reshape([-1, n_batch, 2, 64])
                .to(args.device))

    z_gen = torch.randn(N // n_batch, n_batch, dz).to(args.device)
    
    # critic iterations
    dset = [(zi, xi) for zi, xi in zip(z_gen, x_true)]
    
    def fit ():
        # Optimize WGAN loss
        print(f"Fitting WGAN on {N} pulses")
        gan.writer = SummaryWriter(args.writer)
        gan.to(args.device)
        l0 = gan.loss_on(z_gen[0], x_true[0], 1000)
        gan.fit(dset, lr=lr[0], epochs=args.epochs, tag="critic score")
        gan.critic.episode_callbacks = []
        gan.save(args.output)
    
    @gan.epoch
    def plot_samples(tag, data, epoch):
        z = torch.randn(16, dz).to(args.device)
        with torch.no_grad():
            x_gen = gan(z).cpu()
        fig = rp.pulses(x_gen[:,0])
        gan.writer.add_figure(f"Samples at epoch {epoch + 1}", fig)
    
    gan.N_crit = 0

    @gan.critic.episode
    def log_wasserstein(tag, data):
        x, y = data['train'][0]
        with torch.no_grad():
            loss = - gan.critic.loss_on(x, y)
        gan.write("Wasserstein estimate", loss, gan.N_crit)
        gan.N_crit += 1

    fit()
