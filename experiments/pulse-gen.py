import os
import torch
import torch.nn as nn

import revert.plot as rp
import revert.cli  as cli

from revert.models import WGAN, Lipschitz, Twins,\
                          ConvNet, Affine, Linear,\
                          View, Pipe, Module
dz, dx = 8, 64

ns = (1, 100)           # (n_gen, n_crit)   : respective number of iterations
lr = (1e-3, 5e-3)       # (lr_gen, lr_crit) : respective learning rates 
n_batch = 512
beta = 2.5

args = cli.parse_args(dirname='wgan-pulses',
                      input='../twins/VICReg-64:8:64-may12-1.pt',
                      output=f'WGAN-8:64 | ns:{ns} lr:{lr} beta:{beta}.pt',
                      datatype='infusion',
                      epochs=20,
                      beta=beta,
                      device='cuda:0',
                      std_z=0.8,
                      mask=False,
                      data='baseline-no_shunt.pt')

class MaskMul(Module):
    """ Multiply first channel by masks."""
    def forward(self, x):
        x_mask = x.prod(1).unsqueeze(1)
        return torch.cat([x_mask, x[:,1:]], dim=1)  

def generator(args):

    c = 2 if args.mask else 1

    # scale output
    scale_out = Affine(c, c, dim=-2)
    with torch.no_grad():
    w = torch.tensor([[2.5, 0.], [0., .5]] if c == 2 else [[2.5]])
        b = torch.tensor([0., 1.5] if c == 2 else [0.01])
        scale_out.weight = nn.Parameter(w)
        scale_out.bias   = nn.Parameter(b)

    G = Pipe(Linear(dz, 64),
             View([8, 8]),
             ConvNet([[8, 16,  c],
                      [8, 32, 64],
                      [4, 8]]),
             scale_out,
             MaskMul())

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
             ConvNet([[c,  32, 16, 1],
                      [64, 16, 1, 1],
                      [8,  16, 1]]),
             View([1]),
             Linear(1, 1))

    D = Lipschitz(D, args.beta)
    D.scale_in = D.model.module0
    return D

def main (args):

    G = generator(args)
    D = critic(args)
    gan = WGAN(G, D, ns=ns, lr_crit=lr[1])

    # serialized pulses and masks
    data  = torch.load(args.data)

    # true pulses
    pulses = (nn.AvgPool1d(2)(data['pulses'])
                .view([-1, 64]))
    masks = (nn.AvgPool1d(2)(data['masks'])
                .view([-1, 64]))
        
    N = (pulses.shape[0] // n_batch) * n_batch
    idx = torch.randperm(N)

    if args.mask:
        x_true = (torch.stack([pulses[:N], masks[:N]], dim=1) 
                    [idx]
                    .reshape([-1, n_batch, 2, 64])
                    .to(args.device))
        x_true -= x_true.mean()
        x_true /= x_true.std()

    else:
        x_true = (pulses[:N][idx]
                    .reshape([-1, n_batch, 1, 64])
                    .to(args.device))
        x_true[:,0] -= x_true[:,0].mean()
        x_true[:,0] /= x_true[:,0].std()

    # seeds
    z_gen = torch.randn(N // n_batch, n_batch, dz).to(args.device)
    z_gen *= args.std_z

    dset = [(zi, xi) for zi, xi in zip(z_gen, x_true)]
    
    def fit ():
        gan.write_to(args.writer)
        # Optimize WGAN loss
        print(f"Fitting WGAN on {N} pulses")
        gan.critic.to(args.device)
        gan.to(args.device)
        l0 = gan.loss_on(z_gen[0], x_true[0], 500)
        gan.fit(dset, lr=lr[0], epochs=args.epochs, tag="critic score")
        gan.critic.episode_callbacks = []
        gan.save(args.output)
    
    # number of critic iterations
    gan.N_crit = 0

    @gan.critic.episode
    def log_critic(tag, data):
        x, y = data['train'][0]
        with torch.no_grad():
            fx = gan.critic(x)
            W = gan.critic.wasserstein_on(x, y)
            mult = gan.critic.model.modules[-1].weight.abs().detach()
        gan.write("Wasserstein estimate", W, gan.N_crit)
        gan.write("Critic/Lipschitz estimate", gan.critic.k, gan.N_crit)
        gan.write("Critic/saturation", fx.abs().mean() / mult, gan.N_crit)
        gan.write("Critic/deviation", fx.std([0]).mean() / mult, gan.N_crit)
        gan.N_crit += 1

    @gan.epoch
    def log_gen(tag, data, epoch):
        z = torch.randn(16, dz).to(args.device)
        with torch.no_grad():
            x_gen = gan(z)
            fx = gan.critic(x_gen).cpu()
            c = (fx - fx.min()) / (fx.max() - fx.min())
        fig = rp.pulses(x_gen[:,0].cpu(), c=c)
        gan.writer.add_figure(f"Generator/epoch {epoch + 1}", fig)
    
    fit()


if __name__ == "__main__":
    print(f'\nWGAN:\n {twins}')
    main(args)
