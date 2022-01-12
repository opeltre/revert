import torch

from sklearn.cluster import KMeans

from twins import model, full
from matplotlib import pyplot as plt

st = torch.load("st/model")
model.load_state_dict(st)

K = 32

def dist (a, b):
    return torch.sqrt(torch.sum((a - b)**2, dim=[-1]))

with torch.no_grad():
    x = full[:2500,:32].reshape([-1, 128])
    y = model(x)

kmeans = KMeans(n_clusters=K).fit(y)
c = kmeans.cluster_centers_
c = torch.tensor(c)

ds = torch.stack([dist(ci, y) for ci in c])
_, ids = torch.sort(ds, dim=-1)


#--- plot centers --- 

from math import ceil 

plt.style.use('seaborn')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_clusters (ids):
    
    w, h = 4, ceil(K // 4)
    fig = plt.figure(figsize=(w * 4, h * 2))
    plt.suptitle('K-Means')

    for i, js in enumerate(ids): 
        ax = plt.subplot(h, w, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.plot(x.index_select(0, js[:12]).T, color=colors[i % len(colors)], lw=1)
    return fig
