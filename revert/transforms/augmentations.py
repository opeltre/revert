import torch
import torch.utils.data as data

class AugmentLoader (data.DataLoader):

    def __init__(self, dset, transforms, Nbatch=1, n=2, device='cpu', **kws):
        self.n_augment = n
        self.mix = None
        self.transforms = transforms
        self.device = 'cpu'
        kws['drop_last'] = True
        super().__init__(dset, Nbatch, collate_fn=self.augment, **kws)
    
    def to(self, device):
        self.device = device
        return self
    
    def augment(self, x):
        # First two dimensions
        Nbatch, n = len(x), self.n_augment
        x = torch.stack(x)
        # Mix transforms
        T = self.transforms
        mix = torch.ones(len(T)) if self.mix is None else self.mix
        mix = (Nbatch * mix / mix.sum()).long()
        idx = [0, *mix.cumsum(0).tolist()]
        idx[-1] = Nbatch
        x_t = []
        for t, i, j in zip(T, idx[:-1], idx[1:]):
            xi = x[i:j]
            xi_t = torch.stack([xi, *[t(xi) for i in range(n - 1)]], 1)
            x_t.append(xi_t)
        # Concatenate and Shuffle
        x_t = torch.cat(x_t, 0)
        return x_t
    
    def shuffle(self, x):
        idx = torch.randperm(x.shape[0])
        return x[idx]
    
class Transform : 

    def __init__(self, f=None):
        self.callable = f

    def __call__(self, arg):
        return self.callable(arg)

    def __matmul__(self, other):
        return Transform(lambda x: self(other(x)))
    
    def slice(self, dim, Ndims):
        slc = [None] * Ndims
        slc[dim] = slice(None)
        return tuple(slc)

    def pair(self, arg, dim=0):
        return torch.stack([arg, self(arg)], dim=dim)

class Shuffle (Transform): 
    
    def __init__(self, dim=0):
        self.dim = dim 

    def __call__(self, x):
        idx = torch.randperm(x.shape[self.dim])
        return x.index_select(self.dim, idx)
    
class RandomTransform(Transform):

    def __init__(self, amp):
        self.amp = amp

class VShift(RandomTransform):

    def __call__(self, x):
        slc = self.slice(0, x.dim())
        return x + self.amp * torch.randn(x.shape[0])[slc]    
    
class VScale(RandomTransform):

    def __call__(self, x):
        slc = self.slice(0, x.dim())
        return x * (1 + self.amp * (torch.rand(x.shape[0]) - .5)[slc])

class Noise(RandomTransform):

    def __call__(self, x):
        return x + self.amp * torch.randn(x.shape)


#--- Obsolete on batches  --- 

def vshift (amp):
    Transform(
        lambda x: (x + amp * 
                   torch.randn(x.shape[0])[(slice(None), 
                                                *([None] * x.dim() - 1))]))

def noise (amp):
    return Transform(
        lambda x: x + amp * torch.randn(x.shape))

def scale (amp):
    return Transform(
        lambda x: x * (1 + amp * (torch.rand(x.shape[0]) - .5))[:,None])

def Pair (transform):
    return lambda x: torch.stack([x, transform(x)])

