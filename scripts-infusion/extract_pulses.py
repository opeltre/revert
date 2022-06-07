import torch
import os
import json
import tqdm
import sys

from revert import infusion
from revert.models     import ConvNet
from revert.transforms import filter_spikes, bandpass, Troughs, diff
from revert.transforms import segment, mask_center

if len(sys.argv) > 1:
    dbname = sys.argv[1]
    label  = sys.argv[2] if len(sys.argv) > 2 else "Baseline"
    dest   = f'{label.lower()}-{os.path.basename(dbname)}.pt'

else:
    print("Usage : python extract_pulses.py dirname [label]")
    sys.exit(1)

fs = 100

if "INFUSION_DATASETS" in os.environ and not os.path.isabs(dest):
    dbpath = os.environ["INFUSION_DATASETS"]
    assert os.path.isdir(dbpath)
    dest = os.path.join(dbpath, dest)

db = infusion.Dataset(dbname)

def main (model_state=None, Npulses=64, minutes=6):
    print(f"--- Extracting '{label}' pulses")

    # model variation losses
    print(f"loading model from '{model_state}'" if model_state else "model = Id")
    model = (ConvNet.load(model_state) if model_state else lambda x: x)
    losses = [mean_loss(model), diff_loss(model)]
    loss = mixed_loss(losses)

    # files with {label}
    print(f"filtering files with '{label}' timestamps")
    keys = db.filter(lambda f: db.periods[f.key][label])

    # filters and peak detection
    print(f"extracting pulses from {len(keys)} recordings")
    Npts    = minutes * 6000
    bp      = bandpass(.6, 12, fs)
    # multiscale peak detection
    argmin  = Troughs(Npts, 50)

    #--- Main loop

    out, good = [], []
    bad = {"y_quant": [], "amp": [], "errors": []}
    for k in tqdm.tqdm(keys):
        keep = True
        evts = db.periods[k]
        file = db.get(k)
        try:
            # load ICP slice
            i0 = int(100 * (evts[label][0] - evts["start"]))
            icp = file.icp(i0, Npts)
            icp = filter_spikes(icp)[0]
            # check slice length
            if icp.shape[0] != Npts:
                raise RuntimeError("not enough points")
            # find troughs from filtered ICP
            troughs = argmin(bp(icp).float())
            # filter high Y-quantization
            if quantization_y(icp) >= .099:
                bad["y_quant"].append(k)
                keep = False
                continue
            # extract segments from raw ICP
            segments = select_pulses(icp, troughs, Npulses, loss)
            # filter low amplitudes
            if amplitude_avg(segments[1]) <= 1:
                bad["amp"].append(k)
                keep = False
            if keep:
                out.append(segments)
                good.append(k)
        except Exception as e:
            error = f'{k}:\n{e.args}'
            bad["errors"].append(error)
        file.close()

    #--- Save output

    print(f"saving output as '{dest}'")
    xs    = [torch.stack([x[i] for x in out]) for i in range(4)]
    names = ["masks", "pulses", "means", "slopes"]

    data  = {ni: xi for xi, ni in zip(xs, names)}
    data["keys"] = good
    data |= bad

    for n in names:
        print(f"  + {n}\t: {list(data[n].shape)} tensor")
    print(f"  + keys\t: {xs[0].shape[0]} list string")

    torch.save(data, f'{dest}')

    print(f"extracted {Npulses} pulses from {len(data['keys'])} recordings")
    print(f"  - {len(bad['y_quant'])} bad Y-quantizations encountered")
    print(f"  - {len(bad['amp'])} low amplitudes encountered")
    print(f"  - {len(bad['errors'])} errors encountered")
    return data

#--- Pulse selection

def select_pulses (icp, troughs, Npulses, loss):
    Npts = icp.shape[-1]
    # segment and center pulses
    pulses, mask = segment(icp, troughs, 128)
    x, x_mean, x_slope = mask_center(pulses, mask, output='slopes')
    # sort by loss value
    z = loss(x)
    idx = z.sort().indices[:Npulses]
    # return best pulses
    return mask[idx], x[idx], x_mean[idx], x_slope[idx]

def mean_loss(model=None):
    def loss(x):
        with torch.no_grad():
            y = (model(x) if not isinstance(model, type(None))
                          else x)
        y_mean = y - y.mean([0])[None,:]
        return y_mean.norm(dim=[-1])
    return loss

def diff_loss(model=None):
    def loss(x):
        with torch.no_grad():
            y = (model(x) if not isinstance(model, type(None))
                          else x)
        dy = diff(y.T).T
        return dy.norm(dim=[-1])
    return loss

def mixed_loss(losses, weights=None):
    if isinstance(weights, type(None)):
        weights = torch.ones([len(losses)])
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights)
    def loss(x):
        z = torch.stack([l(x) for l in losses])
        return (z * weights[:,None]).sum([0]) / weights.sum()
    return loss

#--- Bad recordings ---

def quantization_y(x):
    """ Return Y-quantization, should be below .1 """
    dx = torch.diff(x).abs()
    nz = dx.nonzero().flatten()
    return dx[nz].min()

def amplitude_avg(x):
    """ Return average amplitude of segments, should be above 1. """
    return (x.max(1).values - x.min(1).values).mean()

if __name__ == '__main__':
    main(model_state = None)
