import torch
import os
import json

from torch.utils.data import Dataset, DataLoader, TensorDataset


PREFIX = (os.environ["INFUSION_DATASETS"] 
            if "INFUSION_DATASETS" in os.environ
            else os.getcwd())

def shuffle (dim, tensor):
    return tensor.index_select(dim, torch.randperm(tensor.shape[dim]))

class PulseDir (Dataset):
    """ Interface to a directory of segmented pulses of the form:

            INF_xxx_INF.pt -> Tensor [Npulses, Npts] 

        Use dataset.aggregate("dataset.pt", 256) to randomly
        select 256 pulses per patients and stack them to a 
        large N x 256 x Npts tensor. 
    """ 

    def __init__(self, path):
        super().__init__()
        self.path = (os.path.join(PREFIX, path) if path[0] != "/"
                                                else path)
        self.keys = [k.replace(".pt", "") for k in os.listdir(self.path)]

    def __getitem__(self, key):
        key = key if isinstance(key, str) else self.keys[key]
        return torch.load(os.path.join(self.path, f"{key}.pt"))
    
    def __len__(self):
        return len(self.keys)
    
    def select (self, keys, max_per_patient=256):
        acc, N = [], max_per_patient
        for k in keys:
            pulses = self[k]
            idx = torch.randint(pulses.shape[0], [N])
            acc += [pulses.index_select(0, idx)]
        return torch.stack(acc)

    def aggregate(self, dest=None, max_per_patient=256):
        dest = dest if dest else f"{self.path}.pt"
        out = self.key_select(self.keys)
        torch.save(out, dest)


class Pulses (Dataset):
    """ Interface to stacked Npatients x Npulses x Npts pulse tensors."""

    def __init__(self, path, ids=None):

        super().__init__()
        name = f'pulses_{path}.pt'
        self.path = (path if path[0] == "/"
                          else os.path.join(PREFIX, name))
        db = torch.load(self.path)

        #--- Subsets ---
        if type(ids) == type(None):
            ids = len(db['keys'])
        if type(ids) == int: 
            ids = torch.arange(ids) 

        pulses = db['pulses'].index_select(0, ids) 

        #--- Shape --- 
        self.Npatients = pulses.shape[0]
        self.Npulses   = pulses.shape[1]
        self.Npts      = pulses.shape[2]
        #--- Exam keys ---
        self.keys = [db['keys'][i] for i in ids]
        #--- Tensors ---
        self._singles  = pulses.view([-1, self.Npts])
        self._pairs    = (pulses.view([-1, self.Npts])
                           .view([-1, 2, self.Npts]))
        self.pulses = pulses
    
    def pairs (self, n_batch=256, shuffle=True):
        """ DataLoader instance for pulse pairs. """
        dset    = TensorDataset(self._pairs)
        collate = lambda ps: torch.stack(ps, 1)
        return DataLoader(dset, n_batch, shuffle, 
                          collate_fn=collate, 
                          drop_last=True)
   
    def singles (self, n_batch=256, shuffle=True):
        """ DataLoader instance for single pulses. """
        dset    = TensorDataset(self._singles)
        return DataLoader(dset, n_batch, shuffle, drop_last=True)
    
    def results (self, path='results-full.json'):
        path = (os.path.join(PREFIX, path) if path[0] != '/' else path)
        with open(path, "r") as f:
            res = json.load(f)
        return res
    
    def targets (self, fields, path='results-full.json'): 
        res = self.results(path)
        pulses, tgts = [], []
        for k, ps in zip(self.keys, self.pulses):
            if k in res:
                if not sum(not f in res[k] for f in fields):
                    pulses  += [ps]
                    tgts    += [torch.tensor([res[k][f] for f in fields])]
        return torch.stack(pulses), torch.stack(tgts)
    
    def items (self, n_batch=256, shuffle=True):
        """ DataLoader instance for (key, pulse) tuples. """ 
        return DataLoader(self, n_batch, shuffle, drop_last=True)
    
    def __getitem__(self, idx):
        return self.keys[idx // self.Npulses], self._singles[idx] 

    def __len__(self): 
        return self._singles.shape[0]
