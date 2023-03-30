import torch
import torch.nn as nn

from .module import Module, Slice, Prod, Branch

class Diff(Module):
    """ Centered Differentials. """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, index=None):
        dim = self.dim
        d_x = []
        unsqueeze = False
        if index is None: 
            index = range(1, dim + 1)
        if isinstance(index, int):
            index = [index]
            unsqueeze = True
        for j in index:
            dj = torch.diff(x, dim=-j)
            dj_0 = dj.select(-j, 0).unsqueeze(-j)
            dj_1 = dj.select(-j, -1).unsqueeze(-j)
            dj_x = (torch.cat([dj_0, dj], -j) 
                 +  torch.cat([dj, dj_1], -j)) / 2
            d_x.append(dj_x)
        if unsqueeze:
            return d_x[0]
        return torch.stack(d_x, -1-dim)


class Jet(Module):
    """ Jets returned as list of differentials. """
    
    def __init__(self, dim, rank):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.diff = Diff(self.dim)
        self.indices = self.compute_indices(self.dim, self.rank)

    def graded(self):
        begin = self.begin
        slices = [slice(i, j) for i, j in zip(begin[:-1], begin[1:])]
        return (Prod([Slice(slc, -1 -self.dim) for slc in slices])
                @ Branch(len(slices))
                @ self)
    
    
    @classmethod
    def compute_indices(cls, dim, rank):
        """ 
        Ordered list of integer vectors summing to r for r <= rank.  
        """
        if not isinstance(dim, int) or dim < 1:
            raise RuntimeError('dim must be >= 1.')
        #--- K[1, r] = {[r]} ---
        if dim == 1:
            return [torch.tensor([[r]]) for r in range(rank + 1)]
        
        #--- K[d-1, r+1] -> K[d, r+1] ---
        Kd_1 = cls.compute_indices(dim - 1, rank)
        def embed (k): 
            zero = torch.zeros([*k.shape[:-1], 1], dtype=torch.long)
            return torch.cat([zero, k], -1)
        
        #--- K[d, r] -> K[d, r+1] ---
        e1 = torch.zeros([dim], dtype=torch.long)
        e1[0] = 1
        def increment(k): 
            return e1 + k
        
        #--- K[d, r+1] =~ embed(K[d-1, r+1]) 
        #---           +  increment(K[d, r])
        Kd = [embed(Kd_1[0])]
        for r in range(1, rank + 1):
            K0 = embed(Kd_1[r])
            K1 = increment(Kd[r-1])
            Kd_r = torch.cat([K0, K1])
            Kd.append(Kd_r)
        
        return Kd
    
    @classmethod
    def compute(cls, x, dim, rank):
        diff = Diff(dim)
        if dim == 1:
            Jx = [x.unsqueeze(0)]
            for r in range(rank):
                Jx.append(diff(Jx[-1], 1))
            return Jx
        Jd_1 = cls.compute(x, dim - 1, rank)
        Jd = [Jd_1[0]]

        for r in range(1, rank + 1):
            J0 = Jd_1[r]
            J1 = diff(Jd[r-1], dim)
            Jd.append(torch.cat([J0, J1]))
        
        return Jd
    
        
    def forward(self, x, degree=None):
        Jx = self.compute(x, self.dim, self.rank)
        if degree is None:
            return torch.cat(Jx)
        return Jx[degree]
            



    



        
        
        

