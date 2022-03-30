import os
import json
import torch
import matplotlib.pyplot as plt
from math import pi
from scipy.signal import find_peaks as peaks
from torch.fft import rfft

dset_path   = os.environ["INFUSION_DATASETS"]
tags_path   = os.path.join(dset_path, "tags_2016.json")
pulses_path = os.path.join(dset_path, "pulses_2016")

with open(tags_path, "r") as f:
    tags = json.load(f)

good = [k for k, tk in tags.items() if tk == 0]

def load_pulses (p):
    seg = torch.load(os.path.join(pulses_path, p))
    lengths = seg[0]
    pulses  = seg[1:]
    return lengths, pulses.t()

def ND_out (params): 
    a, phi, w = params
    def out (t):
        #--- w `otimes` t
        wt     = w[:, None] @ t[None, :]
        modes  = a[:, None] * torch.sin(wt + phi[:, None])
        return modes.sum(dim=[0])
    return out

def ND (x, p0 = None, nit=1000, rate=0.001, pdim=20): 
    #--- time domain
    t = torch.linspace(0, 2 * pi, x.shape[0])
    #--- params
    if p0 == None:
        p = init_fft(x, pdim)
    else: 
        p = p0
    p.requires_grad_(True)
    #--- loss
    l2 = lambda u : torch.sqrt((u**2).sum())
    #--- loop
    for i in range(nit):
        loss = l2(ND_out(p)(t) - x)
        print(f"loss: {loss}")
        loss.backward()
        dp = rate * p.grad
        p.requires_grad_(False)
        p.grad.zero_()
        p -= dp
        p.requires_grad_(True)
    return p

def init_fft(x, pdim=3):
    Fx  = rfft(x)
    a   = 0.5 * torch.randn([pdim])
    _, js = torch.sort(Fx.abs(), descending=True)
    phi = torch.index_select(Fx, 0, js).angle()[:pdim]
    w   = js[:pdim]
    return torch.stack([a, phi, w])

params = torch.tensor([
    [1., 1., 1.],
    [0., 0., 0.],
    [1.1, 2.2, 3.3]
])

t = torch.linspace(0, 2*pi, 50)
x = ND_out(params)(t)
