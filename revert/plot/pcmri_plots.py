import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from math import ceil

STYLES = {
    "art > cervi": {"color": "#f64"},
    "art > cereb": {"color": "#f64", "linestyle": "dotted"},
    "ven > cervi": {"color": "#53a"},
    "ven > cereb": {"color": "#53a", "linestyle": "dotted"},
    "c2-c3"      : {"color": "#4a2"},
    "aqueduc"    : {"color": "#4a2", "linestyle": "dotted"}
}
CHANNELS = [
    "art > cervi",
    "art > cereb",
    "ven > cervi",
    "ven > cereb",
    "c2-c3"      ,
    "aqueduc"    
]

def flows (x, channels=CHANNELS, repeat=2, legend=True, shape=tuple(), title="PCMRI Flows"):
    shape = shape if len(shape) >= 2 else (20, 10)
    x = torch.cat([x] * repeat, dim=-1)
    fig = plt.figure(figsize=shape)
    for (xi, ci) in zip(x, channels):
        label = ci if legend else None
        plt.plot(xi, **STYLES[ci], label=label)
    if legend:
        plt.legend(loc="lower right")
    plt.title(title)
    return fig
