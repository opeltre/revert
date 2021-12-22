import torch
import matplotlib.pyplot as plt

from models     import ND, ConvNet, BarlowTwins, cross_correlation
from infusion   import pulses

import argparse, sys

from torch.utils.tensorboard import SummaryWriter

#--- Dataset ---

full = pulses.Tensor("full").pulses
data = (full[:2500]
            .reshape([2500, 2, -1, 128])
            .transpose(0, 2)
            .reshape([2, -1, 128]))

data = pulses.shuffle(1, data)
print(f"Number of pulse pairs: {data.shape[1]}")

#--- Models ---

layers = [[128, 1, 16],
          [32,  8, 8],
          [8,   8, 8],
          [1,   8, 1]]

model = ConvNet(layers)

twins = BarlowTwins(model)

#--- Main --- 

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
if log_dir:
    twins.writer = SummaryWriter(log_dir)

#--- Synthetic pairs

def noisy_pairs (n_samples = 2 << 13, n_modes = 6):
    ps = torch.randn([n_samples, 2, n_modes])
    x  = ND.map(ps, 128)
    xs = torch.stack([x, x + 0.25 * torch.randn(x.shape)])
    return xs

#--- Plot input pairs 

def plot_pairs (xs):
    colors = ["#da3", "#bac", "#8ac", "#32a", "#2b6"]
    for i in range(5):
        plt.plot(xs[0,i], color=colors[i], linewidth=1)
        plt.plot(xs[1,i], color=colors[i], linestyle="dotted", linewidth=1)
    plt.show()

if __name__ == "__main__":
    print(f'\ntwins:\n {twins}')
    pass
