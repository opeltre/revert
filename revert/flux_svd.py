import torch
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

def age (path): 
    return pcmri(path).age()

def flux (path): 
    """ Load a 3-channel debit curve from exam key """
    exam = pcmri(path)
    curve = exam.fluxes("cervi")
    curve = curve.fmap(resample(N_POINTS))
    curve = curve.fmap(heat(HEAT))
    channels = ['art', 'ven', 'csf']
    f = torch.stack([curve[k] for k in channels])
    alpha = f[0].sum() / f[1].sum() 
    f[1] *= alpha
    return f

def jet2 (path):
    """ Load 2-jet of debit curve from exam key """
    curve = flux(path)
    j = jet(2, step = CC / N_POINTS)(curve)
    j = j.reshape((j.shape[0] * j.shape[1], j.shape[2]))
    return j

def jet2_md (path): 
    j = jet2(path)
    mean = j.mean(dim=1)
    jc = j - mean[:,None]
    dev = torch.sqrt(jc.var(dim=1))
    return jc / dev[:,None], mean, dev

def jet2_eqs (path): 
    """ Extract 2nd order ODEs from exam key """
    j = jet2(path)
    u,s,v = torch.svd(j)
    if s[3] / s[0] > 0.01:
        print(f"{path}: corank seems higher than 3")
        return None
    
    eqs = u.t()[3:]
    projector = eqs.t() @ eqs
    return projector

def projectors (): 
    """ Stack projectors for all exam files """
    ps = [jet2_eqs(e) for e in pcmri_exams()]
    return torch.stack([p for p in ps if isinstance(p, torch.Tensor)])

def tsne (x, k=2, p=30, N=4000): 
    mytsne = TSNE(n_components=k, init='pca', perplexity=p,
                  n_iter=N)
    return mytsne.fit_transform(x)

def pca (x, k=2):
    return PCA(n_components=k).fit_transform(x)

def mdse (x, k=2, e=1e-3):
    return MDS(n_components=k, eps=e).fit_transform(x)

def plot3d (y): 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(y[:,0], y[:,1], y[:,2])
    fig.show()

def plot2d (y):
    plt.scatter(y[:,0], y[:,1])
    plt.show()

def plot_flux(f, title=""):
    t = torch.linspace(0, 1, f.shape[-1])
    plt.plot(t, f[0], color="red", label="art")
    plt.plot(t, f[1], color="blue", label="ven")
    plt.plot(t, f[2], color="green", label="csf")
    plt.title(f"Flux: {title}")
    plt.legend(loc="right")
    plt.show()

def histogram (y, N=10):
    y0, y1 = torch.min(y), torch.max(y)
    x = torch.linspace(y0, y1, N)
    h = torch.tensor([float(len(si)) for si in split(y, N)])
    return (h, x)

def split (y, N=10):
    y0, y1 = torch.min(y), torch.max(y)
    x = torch.linspace(y0, y1, N)
    h = [[] for i in range(N)] 
    for j, yj in enumerate(y):
        i = floor((yj - y0) * (N - 1) / (y1 - y0))
        h[i] += [j]
    return h

def covar (x):
    return (x.t() @ x) * (1 / x.shape[0])
    
def covmetric (x): 
    C = covar(x)
    def met(dx, dy):
        return dx.t() @ (C @ dy)
    return met

def covnorm (x):
    m = covmetric(x)
    return lambda dx: torch.sqrt(m(dx, dx))

#P = projectors() 
#torch.save(P, "projectors.pt")
