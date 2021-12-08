import torch
import math
from scipy.signal import find_peaks
from .filter import heat as Heat

def peaks (t, heat=()):
    if not len(heat):
        return find_peaks(t)[0]
    N = t.shape[0]
    h0, *heat = heat
    ps = find_peaks(Heat(h0)(t))[0]
    for h in heat:
        th = Heat(h)(t)
        r = math.floor(2 * h0)
        slices = [th[max(0, p - r):min(N, p + r)] for p in ps]
        ps = [max(0, p - r) + argmax(s) for s, p in zip(slices, ps)]
        h0 = h
    return ps

def argmax (t):
    return int(torch.argmax(t))
    
def segments (t, heat=(), sign=-1):
    ps = peaks(-t, heat)
    return [t[p:ps[i+1]] for i, p in enumerate(ps[:-1])]
