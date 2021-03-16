import h5py
import os
import torch

def ls (): 
    return os.listdir('data')

class File (h5py.File): 

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
