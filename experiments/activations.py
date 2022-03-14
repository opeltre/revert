import torch

from models     import tsne, pca, mdse
from infusion   import pulses
from matplotlib import pyplot as plt

from twins import model

st = torch.load("st/model")
model.load_state_dict(st)

colors = ['#d3b', '#f35', '#c82', '#fb2',\
          '#5a4', '#5aa', '#38d', '#a2d']


#--- Load P pulses from N patients 

N, P = 500, 6
full = pulses.Tensor("full")
Npat = full.pulses.shape[0]

x    = (full.pulses
            .index_select(0, torch.randint(Npat, [N]))
            .index_select(1, torch.randint(128, [P])))

z = model(x.view([N * P, -1]))
y = (z - z.mean([0])).view([N, P, -1])

#--- Pulse plots 

def plotPulses(x, y, save=None):
    c = torch.argmax(y, dim=1)
    outs = set()
    for i, xi in enumerate(x):
        ci = int(c[i])
        label = (f'out-{ci}' if not ci in outs 
                              else None)
        outs.add(ci)
        plt.plot(xi, color=colors[ci % 8], label=label)
    plt.title('ICP pulses with max activation')
    plt.legend(loc='right')
    if save:
        plt.savefig(save)
    plt.show()

#--- Plot maxima 

def plotArgmax(x, y, i, save=None):
    x = x.view([N * P, -1])
    y = y.view([N * P, -1])
    val, ids = torch.sort(y[:,i], descending=True)
    print(ids)
    xmax = x.index_select(0, ids[:P])
    plt.plot(xmax.T, color=colors[i % 8])
    plt.title(f'out-{i} Max activating')
    if save:
        plt.savefig(save)
    plt.show()

    
