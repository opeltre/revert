import torch
import os
import json
import tqdm
import sys

from .dataset import Dataset

from revert import models
from revert import transforms

from revert.transforms import filter_spikes, bandpass, Troughs, diff
from revert.transforms import segment, mask_center


class ExtractPulses (models.Module):

    def __init__(self, Npulses=64, minutes=4, timestamp=None, fs=100, model=None):
        # model variation losses
        if model is not None:
            print(f"loading model from '{model}'")
            self.model = (models.Module.load(model) if model else lambda x: x)
        losses = [mean_loss(model), diff_loss(model)]
        self.loss_fn = mixed_loss(losses)
        # I/O shapes
        self.Npulses  = Npulses
        self.Npts     = minutes * 60 * fs
        # filters and peak detection 
        self.bandpass = transforms.bandpass(.6, 12, fs)
        self.argmin   = transforms.Troughs(self.Npts, 50)
        # time parameters (move elsewhere)
        self.timestamp = timestamp
        self.minutes = minutes

    def loss(self, x):
        if self.loss_fn is not None:
            return self.loss_fn(x)
        return torch.randn(x.shape[:-1])
    
    def extract_pulses(self, icp):
        icp = transforms.filter_spikes(icp)[0]
        icp_filtered = self.bandpass(icp)
        troughs = self.argmin(icp_filtered)
        pulses, masks = transforms.segment(icp_filtered, troughs, 128)
        return pulses, masks
    
    def center_pulses(self, pulses, masks):
        x, x_mean, x_slope = transforms.mask_center(pulses, masks, output='slopes')
        return x, x_mean, x_slope

    def select_pulses(self, pulses, masks):
        x, x_mean, x_slope = self.center_pulses(pulses, masks)
        # sort by loss value
        z = self.loss(x)
        idx = z.sort().indices[:self.Npulses]
        return masks[idx], x[idx], x_mean[idx], x_slope[idx] 

    def run(self, dbname, save=None, verbose=True):
        """ 
        Run the pulse extraction pipeline to produce a pulse dataset. 
        """
        dataset = Dataset(dbname, self.timestamp, self.minutes)

        #--- Accumulators
        out, good, bad = [], [], {}
        for k in ['y_quant', 'amp', 'errors']:
            bad[k] = [] 

        #--- Main loop 
        for key, icp in tqdm.tqdm(dataset):
            keep = True
            try: 
                pulses, masks = self.extract_pulses(icp)
                if self.icp_quantization_y(icp) >= .099: 
                    bad["y_quant"].append(key)
                    keep = False
                    continue
                segments = self.select_pulses(pulses, masks)
                if self.pulse_amplitude_avg(segments[1]) <= 1:
                    bad["amp"].append(key)
                    keep = False
                if keep:
                    out.append(segments)
                    good.append(key)
            except Exception as e:
                bad["errors"].append(key)

        #--- Aggregate data dictionary
        xs    = [torch.stack([x[i] for x in out]) for i in range(4)]
        names = ["masks", "pulses", "means", "slopes"]
        data  = {ni: xi for xi, ni in zip(xs, names)}
        data["keys"] = good
        data |= bad

        #--- Output 
        if verbose: 
            self.print(data)
        if save is not None:
            print(f"saving output as '{save}'")
            torch.save(data, f'{save}')        

    def print(self, data):
        """ 
        Log information about a run. 
        """
        names = ["masks", "pulses", "means", "slopes"]
        for n in names: 
            print(f"  + {n}\t: {list(data[n].shape)} tensor")
        print(f"  + keys\t: {xs[0].shape[0]} list string")
        print(f"extracted {self.Npulses} pulses from {len(data['keys'])} recordings")
        print(f"  - {len(data['y_quant'])} bad Y-quantizations encountered")
        print(f"  - {len(data['amp'])} low amplitudes encountered")
        print(f"  - {len(data['errors'])} errors encountered")
            
    #--- Bad recordings --- 

    def icp_quantization_y(self, icp): 
        """ 
        Return Y-quantization, should be below .1 
        """
        dx = torch.diff(icp).abs()
        nz = dx.nonzero().flatten()
        return dx[nz].min()

    def pulse_amplitude_avg(self, x):
        """ 
        Return average amplitude of segments, should be above 1. 
        """
        return (x.max(1).values - x.min(1).values).mean()


#--- Pulse score losses --- 

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
        dy = transforms.diff(y.T).T
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