import torch

from twins import model, full
from matplotlib import pyplot as plt
from infusion.data import Pulses

from models.kmeans import KMeans

K = 64
P = 16
width = 8

km = KMeans.load("_kmeans/centers.pt")

st = torch.load("st/out64/mon2")
model.load_state_dict(st)

db = Pulses("full")

with torch.no_grad():
    x = full[:2500,:32].reshape([-1, 128])
    y = model(x)

#--- plot centers --- 

from math import ceil 

def classes (km, x, y):
    labels = km.predict(y)
    C = []
    for j in range(km.k):
        in_j = labels == j
        id_j = torch.nonzero(in_j)[:,0]
        x_j = x.index_select(0, id_j)
        C += [x_j]
    return C

#--- from saved data ---

def sorted_cluster (Ci, mi):
    d = torch.cdist(model(Ci), mi[None,:])[:,0]
    _, ids = torch.sort(d)
    xi = Ci.index_select(0, ids)
    return xi

def cluster_table (km, classes, i, w=6, p=64, c='blue'):
    x = sorted_cluster(classes[i], km.centers[i])
    N = ceil(x.shape[0] / p)
    h = ceil(N / w)
    print(h)
    fig = plt.figure(figsize=(w * 4, h * 4))
    plt.suptitle(f'Cluster {i}  -  {x.shape[0]} pulses')
    for j in range(N):
        ax = plt.subplot(h, w, j+1)
        ax.set_xticks([])
        ax.set_yticks([])
        xj = x[j*p: (j+1)*p]
        plt.plot(xj.T, color=c, lw=.5)
    return fig

def predict (data, x):
    m = data['means']
    d = torch.cdist(x, m)
    _, ids = torch.sort(d, -1)
    return ids[:,0]

def patient_clusters (data, x):
    c = predict(data, x)
