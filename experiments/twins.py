# TODO
#   - log hparams to tensorboard / jsons

import torch
import matplotlib.pyplot as plt

from revert.models     import ND, ConvNet, BarlowTwins, cross_correlation, View
from revert.transforms import noise, vshift, scale
from revert            import infusion

def shuffle (dim, tensor):
    sigma = torch.randperm(tensor.shape[dim], device=tensor.device)
    return tensor.index_select(dim, sigma)

import argparse, sys

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.tensorboard import SummaryWriter

#--- Dataset ---

full = infusion.Pulses("full").pulses
data = (full[:2500]
            .view([2500, -1, 2, 128])
            .view([-1, 2, 128])
            .transpose(0, 1))

data = shuffle(1, data)
print(f"Number of pulse pairs: {data.shape[1]}")

#--- Models ---

layers = [[128, 1,   8],
          [16,  16,  8],
          [8,   32,  8],
          [1,   64,  1]]

model = View([-1]) @ ConvNet(layers, pool='max')

twins = BarlowTwins(model)


#===== State dict / logdir as CLI arguments ===== 
#
#   python twins.py -s "model.state" -w "runs/twins-xx"

parser = argparse.ArgumentParser()
parser.add_argument('--state', '-s', help="load state dict", type=str)
parser.add_argument('--writer', '-w', help="tensorboard writer", type=str)
args = parser.parse_args()

# model state 
if args.state: 
    print(f"Loading model state from '{args.state}'")
    st = torch.load(args.state)
    model.load_state_dict(st)

# writer name
log_dir = args.writer if args.writer else None
if log_dir: twins.writer = SummaryWriter(log_dir)

#==================================================

#--- Main ---

t_comp = noise(0.05) @ vshift(1) @ scale(0.2) 
params = [
        {'transforms': [noise(0.1)],    'epochs': 15, 'lr': 1e-3},
        {'transforms': [vshift(1)],     'epochs': 15, 'lr': 1e-3},
        {'transforms': [scale(0.3)],    'epochs': 15, 'lr': 1e-3},
        {'transforms': [t_comp],        'epochs': 20, 'lr': 1e-3}
]

def episodes (params, defaults=None):
    
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


#=== Max activating ===

from math import ceil 

plt.style.use('seaborn')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def argmax_figs (x, P=6):

    y = twins.model(x).T
    val, ids = torch.sort(y, descending=True, dim=1)

    w, h = 4, ceil(y.shape[0] // 4)

    fig = plt.figure(figsize=((w + 1)*2, h + 2))
    for j, idj in enumerate(ids[:,:P]):
        xmax = x.index_select(0, idj)
        ax = plt.subplot(h, w, j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.plot(xmax.T, color=colors[j % len(colors)])

    return fig
    

#--- Synthetic pairs --- 

def noisy_pairs (n_samples = 2 << 13, n_modes = 6):
    ps = torch.randn([n_samples, 2, n_modes])
    x  = ND.map(ps, 128)
    xs = torch.stack([x, x + 0.25 * torch.randn(x.shape)])
    return xs

#--- Plot input pairs ---

def plot_pairs (xs, n=5):
    colors = ["#da3", "#bac", "#8ac", "#32a", "#2b6"]
    for i in range(n):
        plt.plot(xs[0,i], color=colors[i % len(colors)], linewidth=1)
        plt.plot(xs[1,i], color=colors[i % len(colors)], linestyle="dotted", linewidth=1)
    plt.show()

if __name__ == "__main__":
    print(f'\ntwins:\n {twins}')
    pass
