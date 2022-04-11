import re
import csv
import json
import os
import torch
from glob import glob

from .dataset import CHANNELS
from revert.transforms import resample

def channel (name): 
    return (name.replace(".txt", "")
                .replace("_", "-")
                .lower())

class File:
    """ Parse flow files of an exam path. 
    
            File :: { str : Flux }

        Usage:
        --------
        >>> f = File('samples/abc123', 'sinus*', 'aqueduc')
    """
    def __init__(self, exam_path):
        self.path = exam_path
        self.id   = os.path.basename(exam_path)

        self.channels = []
        for name in os.listdir(self.path):
            key = channel(name)
            if key in CHANNELS and CHANNELS[key]["type"] != "x":
                self.channels += [name.replace(".txt", "")]
    
    def getChannels (self, *patterns):
        patterns = [''] if not len(patterns) else patterns
        channels = []
        for pat in patterns: 
            p = channel(pat) if type(pat) == str else pat
            channels += [c for c in self.channels
                           if re.search(p, channel(c))]
        return channels 

    def flows (self, fmt='torch', normalise=True, aqueduc=True):
        """
        Return intracranial flows as a (6, 32) tensor. 

        The returned tensor has 5 to 6 channels, depending 
        the `aqueduc` optional parameter: 

            - arterial cervical
            - arterial cerebral
            - venous cervical
            - venous cerebral
            - csf cervical
            - csf aqueduc
        
        If `normalise=True`, the mean venous flows will be made 
        equal to the mean arterial flows, level-wise. 
        """
        types = ['art > cervi', 'art > cereb', 
                 'ven > cervi', 'ven > cereb']
        flows = [self.sumAll(t, fmt=fmt) for t in types]
        flows += [self.read('c2-c3', fmt=fmt)[0]]
        if aqueduc:
             flows += [self.read('aqueduc', fmt=fmt)[0]]
        if normalise:
            flows[0] *= torch.sign(flows[0]).mean()
            flows[1] *= torch.sign(flows[1]).mean()
            flows[2] *= flows[0].mean() / flows[2].mean()
            flows[3] *= flows[1].mean() / flows[3].mean()
        return torch.stack(flows)

    def sumAll (self, fluxtype, fields=['debit'], fmt='torch'):
        patterns = [c for c in self.channels 
                      if CHANNELS[channel(c)]['type'] == fluxtype] 
        return self.readAll(*patterns, fields=fields, fmt=fmt).sum([0])
    
    def readAll (self, *patterns, fields=['debit'], fmt='torch'):
        files = self.getChannels(*patterns)
        fluxes = {}
        for f in files:
            key   = channel(f)
            flux  = self.read(f, fields, fmt)
            if len(flux) > 0 and CHANNELS[key] != 'x':
                fluxes[key] = self.read(f, fields, fmt)
        if fmt == 'dict': 
            return fluxes
        if fmt == 'json':
            return json.dumps(fluxes)
        if fmt[:5] == 'torch': 
            return torch.cat([fk for fk in fluxes.values()])
        return fluxes

    def read (self, name, fields=['debit'], fmt='torch', pad=128):
        """ Read fields from a flux file. 
            
            Output formats: 
                torch)      Tensor N x T   
                *)          {fields: List T Num}
        """
        #  fmt='torch-128'
        if fmt[:6] == 'torch-':  
            pad = int(fmt[6:])
            f = self.read(name, fields + ['time'], pad=0)
            t = f[-1,-1] / (pad * 10)
            print (f[-1, -1], pad, pad * t)
            return torch.stack([
                torch.cat([
                    resample(int(t * pad))(f), 
                    f[0] * torch.ones([int((1 - t) * pad)])
                ]) for f in f[:-1]
            ]) 

        fields = ['debit', 'surface', 'volume', 'time'] \
               if fields == '*' else fields
        curves = { k: [] for k in fields }
        
        filename = self.getChannels(name)[0] + ".txt"
        with open(os.path.join(self.path, filename), "r") as f:
            env = None
            for line in f:
                num = re.search(r'-+\s*(-?\d+\.?\d*)', line)
                if  env in fields and num != None: 
                    curves[env] += [float(num.group(1))]
                elif re.search('DEBIT_mm3/sec', line):
                    env = 'debit'
                elif re.search('SURFACE_mm2', line):
                    env = 'surface'
                elif re.search('VOLUME_DPL mm3', line):
                    env = 'volume'
                elif re.search('Axe des Temps', line):
                    env = 'time'
        if fmt == 'torch':
            t = torch.stack([torch.tensor(curves[k]) for k in fields])
            return t if t.shape[-1] > 0 else torch.tensor([])
        elif fmt == 'json':
            return json.dumps(curves)
        return curves

    def age (self): 
        """ Look for age of patient. """
        try:
            def year (line): 
                match = re.search(r'\d{4}', line)
                return int(match.group(0)) if match else 0
            
            info = "general-informations.txt"
            info_path = os.path.join(self.path, info)
            
            with open(info_path, "r") as f:
                l = f.readlines()
                return year(l[3]) - year(l[4])
        except:
            return -1

    def __repr__(self): 
        return "PCMRI File:\n  " + "\n  ".join(
             [channel(ck) for ck in self.channels])  

