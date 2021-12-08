from neural_decomposition import ND
from conv import ConvNet
from barlow_twins import BarlowTwins, cross_correlation

import torch
import matplotlib.pyplot as plt

#--- Models ---

model = ConvNet()
twins = BarlowTwins(model)

x = torch.randn([1, 128])
y = model(x)

xs = torch.stack([x, x + 0.01 * torch.randn(x.shape)])
ys = twins(xs)

#--- Synthetic pairs

def noisy_pairs (n_samples = 2 << 13, n_modes = 6):
    ps = torch.randn([n_samples, 2, n_modes])
    x  = ND.map(ps, 128)
    xs = torch.stack([x, x + 0.25 * torch.randn(x.shape)])
    return xs

def train (xs, lr=1e-2, br=1e-3): 
    #--- Print loss before
    ys = twins(xs)
    loss = twins.loss(ys)
    C  = cross_correlation(*ys)
    print(f"Loss: {loss}")
    print(f"Cross-correlation:\n {C}")

    #--- Fit on xs 
    twins.fit(xs, lr, br)
    print(f"\n Fitting on {xs.shape[1]} samples... \n")

    #--- Print loss after
    ys   = twins(xs)
    loss = twins.loss(ys)
    C2 = cross_correlation(*ys)
    print(f"Loss: {loss}")
    print(f"Cross-correlation:\n {C2}")

#--- Plot input pairs 

def plot_pairs (xs):
    colors = ["#da3", "#bac", "#8ac", "#32a", "#2b6"]
    for i in range(5):
        plt.plot(xs[0,i], color=colors[i], linewidth=1)
        plt.plot(xs[1,i], color=colors[i], linestyle="dotted", linewidth=1)
    plt.show()

if __name__ == "__main__":
    xs = noisy_pairs()
    train(xs)
    plot_pairs(xs)
