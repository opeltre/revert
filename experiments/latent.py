import torch
import matplotlib

from models         import tsne, pca, mdse
from infusion.data  import Pulses
from matplotlib     import pyplot as plt

from twins import model

st = torch.load("st/out64/mon3")
model.load_state_dict(st)

db = Pulses('full', 2500)

colors = ['#d3b', '#f35', '#c82', '#fb2',\
          '#5a4', '#5aa', '#38d', '#a2d']

cmap = matplotlib.cm.get_cmap('viridis')

def shunted ():
    field = "Shunt critical ICP [mmHg]"
    res   = db.results()
    S, NS = [], []
    for k, p in zip(db.keys, db.pulses):
        if not k in res:
            continue
        if field in res[k]:
            S += [p]
        else:
            NS += [p]
    return torch.stack(S), torch.stack(NS)

#--- Sort patient pulses by barycenter distance 

def sort (y): 
    d = ((y.mean([1]) - y.mean([0, 1])) ** 2).sum([-1]).sqrt()
    i = list(range(N))
    i.sort(key=lambda j: d[j])
    y = y.index_select(0, torch.tensor(i))
    return y

#--- Scatter plots

def plot2d (y, r=None, c=None, title="Scatter 2d"):
    ax = plt.axes()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.scatter(y[:,0], y[:,1], r, c=c, cmap='viridis')
    plt.title(title)
    plt.show()

def plot3d (y, c=None, title="Scatter 3d"): 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(y[:,0], y[:,1], y[:,2], c=c)
    plt.title(title)
    fig.show()

#--- Reduce output

def main (N=6): 

    #--- Load pulses from N patients 

    full = pulses.Tensor("full")
    ids  = torch.randint(full.pulses.shape[0], [N])
    P    = full.pulses.index_select(0, ids)

    #--- Plot pulses 

    for i, Pi in enumerate(P):
        plt.subplot(3, 2, i+1)
        plt.plot(Pi[:24].T, color=cmap(i / (N - 1)))
    plt.show()

    #--- Model output

    y = model(P.reshape([-1, 128]))

    z = tsne(y, k=2, p=10, N=2000)
    p = pca(y, k=2)
    #m = mdse(y, k=2, e=1e-3)

    color = (torch.linspace(0, N - 1e-4, z.shape[0]) // 1).numpy()
    print(color.reshape([N, 128]))

    plot2d(p, c=color, title="PCA")
    plot2d(z, c=color, title="TSNE")

    #plot2d(m, c=color, title="MDSE")
    #z3 = tsne(y, k=3, p=10, N=2000)
    #plot3d(z3, c=color, title="TSNE")

#--- Pulse plots 

def plotPulses(x, y, save=None):
    c = torch.argmax(y, dim=1)
    for i, xi in enumerate(x):
        plt.plot(xi, color=colors[int(c[i]) % len(colors)])
    plt.title('ICP pulses with max activation')
    if save:
        plt.savefig(save)
    plt.show()

