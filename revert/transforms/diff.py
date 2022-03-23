import torch
from .bounds import mirror 

def diff (t, step=1) :
    """ Centered difference operator on (batched) signals. """
    dim = t.dim() - 1
    dt = torch.diff(t)
    dt0 = dt.select(dim,  0).unsqueeze(dim)
    dt1 = dt.select(dim, -1).unsqueeze(dim)
    return (1 / (2 * step))\
        *  (torch.cat([dt0, dt], dim) + torch.cat([dt, dt1], dim))

def jet (order, step=1):
    def runJet(t):
        js = [t]
        for i in range(order):
            js += [diff(js[-1], step)]
        return torch.stack(js)
    return runJet

def laplacian (N, width=1):
    """
    Graph laplacian as a sparse (N, N)-matrix.
       
    For width=1, returns the usual discrete laplacian.
    For width=n, returns the laplacian of the graph where each index i
    is linked to indices j != i such that |j - i| <= width.
    """
    # diagonal
    i = torch.arange(N)
    # neighbours and degree
    js_r = torch.cat([i + x for x in range(1, width + 1)])
    js_l = torch.cat([i - x for x in range(1, width + 1)])
    j = torch.cat([js_r, js_l])
    mask = (j >= 0) * (j < N)
    deg  = mask.view([2 * width, -1]).long().sum([0]).repeat(2 * width)
    # indices
    ij_diag = torch.stack([i, i])
    ij_off  = torch.stack([i.repeat(2 * width), j])[:,mask]
    # values
    val_diag = torch.ones([N])
    val_off  = (1 / deg)[mask]
    # sparse matrix
    ij  = torch.cat([ij_diag, ij_off], 1)
    val = torch.cat([val_diag, -val_off])
    return torch.sparse_coo_tensor(ij, val, size=[N, N])
