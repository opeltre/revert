import torch
import math
from scipy.signal import find_peaks
from sig.filter import Heat

def segments (icp, heat=6, dist=10):
    peaks = find_peaks(Heat(heat)(- icp))[0]
    segments = [icp[p:peaks[i+1]] for i, p in enumerate(peaks[:-1])]
    return segments
