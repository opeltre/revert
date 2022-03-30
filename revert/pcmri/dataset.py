import torch
import re
import os
import csv
from glob import glob

import revert.pcmri.file as file

PCMRI_DATASETS = (os.environ["PCMRI_DATASETS"] 
        if "PCMRI_DATASETS" in os.environ 
        else os.getcwd())

def mri_path (d=""): 
    """ Path relative to $PCMRI_DATASETS directory. """
    return os.path.join(PCMRI_DATASETS, d) \
        if d[0] != "/" else d

CHANNELS = {} 
with open(mri_path("channels.csv")) as f:
    for level in csv.DictReader(f):
        def strip(s): return re.search(r'\s*(.*?)\s*$', s).group(1) 
        d = {k: strip(s) for k, s in level.items()}
        key = d.pop('key')
        CHANNELS[key] = d

class Dataset :

    channels = CHANNELS

    def __init__(self, path):
        self.path = mri_path(path)

    def glob (self, pattern='*'):
        """ Expand glob in dataset directory. """
        return glob(f'{self.path}/{pattern}')
    
    def ls (self, pattern='*'):
        return [os.path.basename(p) for p in self.glob(pattern)]

    def get (self, pattern):
        if type(pattern) == int:
            return file.File(self.glob()[pattern])
        return file.File(self.glob(pattern)[0])
    
    def getAll (pattern):
        return [file.File(f) for f in self.glob(pattern)]

    def __repr__(self):
        return f"PCMRI Dataset ({len(self.ls())} files)" 
