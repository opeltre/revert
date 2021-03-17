import h5py
import os
import torch

def path (s=''): 
    return os.path.join(os.path.dirname(__file__), s)

def ls (d='data'): 
    return os.listdir(path(d))

class File (h5py.File): 

    def __init__(self, key): 
        if isinstance(key, int): 
            key = ls('data')[key]
        super().__init__(path(f'data/{key}'), 'r')

    def __repr__(self): 
        def gstr(group): 
            s = ''
            for k in group.keys():
                if isinstance(group[k], h5py.Dataset): 
                    s += f'\n{k}: {group[k].shape}'
                else:
                    sk = gstr(group[k]).replace('\n', '\n. ')
                    s += f'\n{k}: {sk}'
            return s 
        return '<Hdf5>' + gstr(self).replace('\n', '\n  ')

    def icp (self, N=200): 
        arr = self['waves']['icp'][0:N]
        return torch.tensor(arr)

