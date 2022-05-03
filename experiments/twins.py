# TODO
#   - log hparams to tensorboard / jsons

import cli_tools as cli
import torch

from revert.models import BarlowTwins, ConvNet, View, cross_correlation
# augmentations
from revert.transforms import noise, vshift, scale
# dataset 
from revert import infusion

from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

args  = cli.read_args(cli.arg_parser(prefix='twins'))

#--- Model ---

layers  = [[64, 1,  8],
           [8,  16, 1],
           [1,  32, 1]]

model = ConvNet(layers, pool='max') @ nn.AvgPool1d(2)

#--- Expander ---

dim_z = 64
head  = View([dim_z]) @ ConvNet([[1, 32,    1], 
                                 [1, dim_z, 1]])

#--- Twins ---

twins = BarlowTwins(head @ model) 

twins.writer = SummaryWriter(args.writer)

if args.input: 
    twins = twins.load(args.input)

#--- Dataset : to cleanup! ---

def shuffle (dim, tensor):
    sigma = torch.randperm(tensor.shape[dim], device=tensor.device)
    return tensor.index_select(dim, sigma)

full = infusion.Pulses("full").pulses
data = (full[:2500]
            .view([2500, -1, 2, 128])
            .view([-1, 2, 128])
            .transpose(0, 1))

data = shuffle(1, data)
print(f"\nNumber of pulse pairs: {data.shape[1]}")

#==================================================

#--- Main ---

t_comp = noise(0.05) @ vshift(1) @ scale(0.2) 
params = [
        {'transforms': [noise(0.1)],    'epochs': 15, 'lr': 1e-3},
        {'transforms': [vshift(1)],     'epochs': 15, 'lr': 1e-3},
        {'transforms': [scale(0.3)],    'epochs': 15, 'lr': 1e-3},
        {'transforms': [t_comp],        'epochs': 20, 'lr': 1e-3}
]

def main (params, defaults=None):

    twins.cuda()

    defaults = {'epochs':  25,
                'n_batch': 256,
                'lr':      1e-3,
                'gamma':   0.8,
                'n_it':    3750
                } | (defaults if defaults else {})

    for i, p in enumerate(params):
        p = defaults | p 

        #--- data ---
        xs = (pretraining(p['transforms'], p['n_batch'], p['n_it'])
                if 'transforms' in p
                else training(p['n_batch']))
        xs = xs.cuda()

        #--- optimizer ---
        optim = Adam(twins.parameters(), lr=p['lr'])
        lr    = ExponentialLR(optim, gamma=p['gamma'])

        name  = f'pretrain-{i}' if 'transforms' in p else 'train'
        twins.fit(xs, optim, lr, epochs=p['epochs'], w=f"Loss/{name}")
        free(optim, lr)
        
        #--- cross correlation ---
        with torch.no_grad():
            n = f'Cross correlation/{name}'
            xs = xs.transpose(0, 1).view([2, -1, 128])
            xs = shuffle(1, xs)[:,:2500]
            ys = twins(xs)
            C = cross_correlation(*ys)
            twins.writer.add_image(n, (1 + C) / 2, dataformats="HW")
        free(xs, ys, C)

#=== Other helpers === 

#--- Cuda free ---

def free (*xs):
    for x in xs: del x
    torch.cuda.empty_cache()

#--- Batch --- 

def batch (n_batch, xs):
    n_it = xs.shape[1] // n_batch
    return (xs[:,: n_it * n_batch]
            .view([2, n_it, n_batch, 128])
            .transpose(0, 1))

#--- Augmented pairs ---

def pretraining (transforms, n_batch=128, n_it=1250):
    
    x  = shuffle(0, data.view([-1, 128]))[:n_it * n_batch] 
    xs = torch.cat([t.pair(x) for t in transforms], 1)
    return batch(n_batch, xs)

#--- Real pairs --- 

def training (n_batch=128): 
    return batch(n_batch, data)

#--- Synthetic pairs --- 

def noisy_pairs (n_samples = 2 << 13, n_modes = 6):
    ps = torch.randn([n_samples, 2, n_modes])
    x  = ND.map(ps, 128)
    xs = torch.stack([x, x + 0.25 * torch.randn(x.shape)])
    return xs

if __name__ == "__main__":
    print(f'\ntwins:\n {twins}')
    main(params)
    twins.save(args.output)
