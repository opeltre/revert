import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from math import ceil

def style(name='seaborn'):
    plt.style.use('seaborn')

def colors():
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return c

#--- Multiple plots 

def grid (traces, shape=tuple(), title=None, titles=None, c=None, lw=1):
    c = colors() if not c else c
    w      = shape[0]  if len(shape) else 4 
    w, h   = shape[:2] if len(shape) > 1 else (w, ceil(len(traces) / w))
    sw, sh = shape[2:] if len(shape) > 2 else (2, 2)
    fig = plt.figure(figsize=(w * sw, h * sh))
    if title:
        plt.suptitle(title)
    for i, t in enumerate(traces):
        ax = plt.subplot(h, w, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.plot(t, color=c[i % len(c)], lw=lw)
        if titles:
            plt.title(titles[i], {'fontsize': 8}, y=0.93)
    return fig

#--- KMeans plots 

def kmeans_grid (km, x, y=None, P=12, shape=(4, 4), avg=False, title="Cluster grid"):
    if isinstance(y, type(None)): y = x
    ids = km.nearest(P, y)
    nearest = [x.index_select(0, js).cpu() for js in ids]
    traces  = [xs.T if not avg else xs.mean([0]) for xs in nearest]
    masses  = (km.counts(y) / y.shape[0]).cpu()
    stds    = (km.stdevs(y) / y.shape[-1]).cpu()
    titles  = [f'[{i}]    mass {100 * mi:.1f}% - std {stds[i]:.2f}'\
                    for i, mi in enumerate(masses)]
    fig = grid(traces, shape, title=title, titles=titles)
    return fig

def cluster_grid (x, z, P=64, title='Cluster grid'):
    n_clusters = int(z.max())
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z)
    idx = [(z == i).nonzero().flatten() for i in range(n_clusters)]
    clusters = [x[i] for i in idx]
    traces = [xi[:P].T for xi in clusters]
    shape  = (4, ceil(n_clusters / 4))
    titles = [f'[{i}]    mass {ci.shape[0]}' for i, ci in enumerate(clusters)]
    fig = grid(traces, shape, title=title, titles=titles)
    return fig


def cluster(km, i, x, y=None, P=64, shape=tuple(), **kws):
    if isinstance(y, type(None)): y = x
    c = km.predict(y)
    idx = (c == i).nonzero().flatten()
    xi  = x[idx]
    Nplots = ceil(x.shape[0] / P)
    traces = [x[j:j+P] for j in range(Nplots)]
    grid(traces, **kws)
    plt.plot(xi[:64].T, color=c, lw=.5)
    plt.title(f'cluster {i}')
    plt.show()

#--- Full ICP recordings

def infusion(icp, events, fs=100, size=(20, 10)):
    # full ICP signal
    fig = plt.figure(figsize=size)
    time = [float(i) / fs for i in range(icp.shape[0])]
    plt.plot(time, icp, color="orange")
    bnd = icp.min(), icp.max()
    # event markers
    def plotEvent(name, color):
        if not name in events: return None
        xs = events[name]
        x0 = events["start"]
        for i in (0, 1): plt.plot([xs[i] - x0] * 2, bnd, color=color)
    names  = ["Baseline", "Infusion", "Plateau"]
    colors = ["blue", "red", "green"]
    for n, c in zip(names, colors): plotEvent(n, c)
    return fig

def pulses(xs, size=(10, 4), c=None):
    c = np.linspace(0, 1, xs.shape[0]) if isinstance(c, type(None)) else c
    color=cm.coolwarm_r(c)
    fig = plt.figure(figsize=size)
    for xi, ci in zip(xs, color):
        plt.plot(xi, color=ci)
    plt.title("Centered pulses")
    return fig
