import torch

from sklearn.cluster import KMeans

from twins import model, full
from matplotlib import pyplot as plt

st = torch.load("st/out64/mon2")
model.load_state_dict(st)

K = 64
P = 16
width = 8

def dist (a, b):
    return torch.sqrt(torch.sum((a - b)**2, dim=[-1]))

with torch.no_grad():
    x = full[:2500,:32].reshape([-1, 128])
    y = model(x)

def kmeans (K=K):
    return KMeans(n_clusters=K).fit(y)

def near_centers (km, P=P):
    c = km.cluster_centers_
    c = torch.tensor(c)
    ds = torch.stack([dist(ci, y) for ci in c])
    _, ids = torch.sort(ds, dim=-1)
    return ids[:,:P]

#--- plot centers --- 

from math import ceil 

plt.style.use('seaborn')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_clusters (km, P=P):

    ids = near_centers(km, P)
    
    w, h = width, ceil(K // width)
    fig = plt.figure(figsize=(w * 4, h * 4))
    plt.suptitle('K-Means')

    for i, js in enumerate(ids): 
        ax = plt.subplot(h, w, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.plot(x.index_select(0, js).T, color=colors[i % len(colors)], lw=1)
        plt.title(i,  {'fontsize': 8}, y=0.93)
    return fig

def plot_cluster(xi, i='n', c='blue'):
    plt.plot(xi[:64].T, color=c, lw=.5)
    plt.title(f'cluster {i}')
    plt.show()

def classes (km):
    labels = torch.tensor(km.predict(y))
    C = []
    for j in range(km.n_clusters):
        in_j = labels == j
        id_j = torch.nonzero(in_j)[:,0]
        x_j = x.index_select(0, id_j)
        C += [x_j]
    return labels, C

