import torch
from torch.fft import fft, ifft
torch.set_printoptions(2)

import matplotlib.pyplot as plt

from loaders import pcmri, pcmri_exams
from sig import normalise, heat, resample, jet

from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA

from math import floor

CC = 1        # cardiac cycle duration (time units)
N_POINTS=64     # time points / C.C.
HEAT=1          # heat kernel

COLORS = ["cyan", "blue", "green", "orange", "red"]

def age (path): 
    return pcmri(path).age()

def age_groups (): 
    groups = [[] for i in range(5)]
    for e in pcmri_exams():
        group = min(floor(age(e) / 20), 4)
        if group >= 0:
            groups[group] += [e]
    return groups

def flux (path, level="cervi"): 
    """ Load a 3-channel debit curve from exam key """
    exam = pcmri(path)
    curve = exam.fluxes(level)
    curve = curve.fmap(resample(N_POINTS))
    curve = curve.fmap(heat(HEAT))
    alpha = curve['art'].sum() / curve['art'].sum()
    curve['ven'] = alpha * curve['ven']
    channels = ['art', 'ven', 'csf']
    curve = torch.stack([curve[k] for k in channels])
    return curve

def jet2 (path):
    """ Load 2-jet of debit curve from exam key """
    curve = flux(path)
    j = jet(2, step = CC / N_POINTS)(curve)
    j = j.reshape((j.shape[0] * j.shape[1], j.shape[2]))
    return j

def flux_cs (path): 
    f = flux(path)
    alpha = f[0].sum() / f[1].sum()
    jcs = f[0] - alpha * f[1]
    js = f[2] - f[2].mean()
    return jcs, js

def plot_cs (path):
    jcs, js = flux_cs(path)
    plt.plot(js, color="green", label="spinal")
    plt.plot(jcs, color="pink", label="cerebrospinal")
    plt.legend(loc="lower right")
    plt.show()

def transfer_cs (path):
    jcs, js = flux_cs(path)
    H = fft(js) / fft(jcs)
    return H

def transfer_av (path):
    f = flux(path)
    fa, fv = f[0], f[1]
    H = fft(fv) / fft(fa)
    return H 

def cloud_H1 (H1_func=transfer_av): 

    def H1 (e): 
        try:
            H = transfer_cs(e)[1]
            return torch.tensor([torch.real(H), torch.imag(H)])
        except:
            return torch.tensor([0., 0.])

    groups = age_groups()

    cloud = [
        torch.stack([ H1(e) for e in exams ]) \
        for exams in groups
    ]

    return cloud

def cloud_av ():
    def H1 (e):
        H = transfer_av(e)[1]
        return torch.tensor([torch.real(H), torch.imag(H)])

    groups = age_groups()
    return [
        torch.stack([ H1(e) for e in exams])\
        for exams in groups
    ]

def plot_cloud (groups):
    for (i, points) in enumerate(groups):
        plt.scatter(points[:,0], points[:,1], color=COLORS[i])
    plt.show()
