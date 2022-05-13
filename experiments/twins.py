import torch
import json
import os

from revert import cli, infusion
from revert.models      import BarlowTwins, VICReg, ConvNet, View
from revert.transforms  import noise, vshift, scale

from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR

dx  = 128
dy  = 8
dz  = 64
TwinType = VICReg
TwinArgs = [(1, 1, .1)]

args = cli.parse_args(name=f'{TwinType.__name__}-{dx}:{dy}:{dz}',
                      datatype='infusion',
                      data='baseline-full.pt',
                      dirname=f'twins')

#========= Model ===========================

downsample = nn.AvgPool1d(128 // dx)

model_layers = [[1, 64, dy],
               [dx, 8, 1],
               [8,  8]]

head_layers = [[dy, 32, dz],
               [1,  1,  1],
               [1,  1]]

model   = ConvNet(model_layers) @ downsample
head    = View([dz]) @ ConvNet(head_layers)

twins = TwinType(head @ model, *TwinArgs)

#========= Writer ===========================

twins.write_to(args.writer)

if args.input:
    twins = twins.load(args.input)

@twins.epoch
def dump_loss_stdev(tag, data, epoch):
    y = twins(shuffle(0, data['val'])[:2048])
    stdev = y.std(0).mean()
    loss  = twins.loss(y)
    twins.write(f"Loss validation/{tag}", loss, epoch)
    twins.write(f"Stdev/{tag}", stdev, epoch)
    twins.free()

@twins.episode
def dump_xcorr(tag, data):
    n = f'Cross correlation/{tag}'
    y = twins(shuffle(0, data['val'])[:2048])
    C = twins.xcorr(y)
    twins.writer.add_image(n, (1 + C) / 2, dataformats="HW")
    twins.free(C)

twins.write_dict({"model"   : repr(model),
                  "head"    : repr(head),
                  "twins"   : repr(twins),
                  "Layers/model": json.dumps(model_layers),
                  "Layers/head" : json.dumps(head_layers),
                  "Args/twins"  : TwinArgs
                  })

#=========== Main ============================

t_comp = noise(0.05) @ vshift(0.5) @ scale(0.2)
t_comp.__name__ = "compose"
params = [
        {'transforms': [noise(0.1)],    'epochs': 10, 'lr': 1e-4},
        {'transforms': [vshift(0.5)],   'epochs': 10, 'lr': 1e-4},
        {'transforms': [scale(0.3)],    'epochs': 10, 'lr': 1e-4},
        {'transforms': [t_comp],        'epochs': 30, 'lr': 1e-4}
]

def main (params, defaults=None):

    twins.cuda()

    defaults = {'epochs'    : 25,
                'n_batch'   : 128,
                'lr'        : 1e-3,
                'gamma' : 0.8,
                'n_it'  : 3750
                } | (defaults if defaults else {})

    twins.write_dict(defaults)

    data, data_val = getData(args.data)
    print(f"\nFitting over {data.shape[0]} pulse pairs "+\
          f"(validation {data_val.shape[0]})")

    for i, p in enumerate(params):
        print(f'\n[Episode {i}]:')
        p = defaults | p
        #--- data ---
        xs, xs_val = (augmentPairs(data, **p, val=data_val)
                      if 'transforms' in p else
                      realPairs(data, **p, val=data_val))

        #--- optimizer ---
        kws = {
            "tag"   : f'episode {i}' if 'transforms' in p else 'real pairs',
            "epochs": p['epochs'],
            "val"   : xs_val,
            "optim" : Adam(twins.parameters(), lr=p['lr'])
        }
        kws |= {
            "lr"    : ExponentialLR(kws["optim"], gamma=p['gamma'])
        }

        twins.fit(xs, **kws) 

#======= Data loading ======================

def shuffle (dim, tensor):
    sigma = torch.randperm(tensor.shape[dim], device=tensor.device)
    return tensor.index_select(dim, sigma)

def getData (path, ratio=.7, clean=False):
    if clean:
        full = torch.load(path)["pulses"]
    else:
        full = infusion.Pulses("full").pulses
    Npat = int(full.shape[0] * ratio)
    Npts = full.shape[-1]
    data = shuffle(0, full[:Npat].view([-1, 2, Npts]))
    data_val = full[Npat:].view([-1, 2, Npts])
    return data, data_val

#--- Batch ---

def batch (n_batch, xs):
    """ Split in batches along first dimension. """
    n_it = xs.shape[0] // n_batch
    tail = xs.shape[1:]
    return (xs[: n_it * n_batch]
            .view([n_it, n_batch, *tail]))

#--- Augmented pairs ---

def augmentPairs (data, transforms, n_batch=128, n_it=1250, val=None, device='cuda', **kws):
    x  = shuffle(0, data.view([-1, 128]))[:n_it * n_batch]
    xs = torch.cat([t.pair(x, 1) for t in transforms], 1)
    xs = batch(n_batch, xs).to(device)
    if isinstance(val, type(None)):
        return xs
    x_val = augmentPairs(val,  transforms, n_batch, n_it, **kws)
    return xs, x_val.view([-1, 2, 128])
            
#--- Real pairs ---

def realPairs (data, n_batch=128, val=None, device='cuda', **kws):
    xs = batch(n_batch, data).to(device)
    return xs if isinstance(val, type(None)) else xs, val.to(device)

#--- Synthetic pairs ---

def noisy_pairs (n_samples = 2 << 13, n_modes = 6):
    ps = torch.randn([n_samples, 2, n_modes])
    x  = ND.map(ps, 128)
    xs = torch.stack([x, x + 0.25 * torch.randn(x.shape)])
    return xs

if __name__ == "__main__":
    print(f'\ntwins:\n {twins}')
    main(params)
    twins.writer.close()
    print(f'\n> save output to {os.path.basename(args.output)}')
    twins.save(args.output)
