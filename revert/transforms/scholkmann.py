import torch
from .filter import bandpass

def diff_scalogram (smax, N, forward=1):
    """ 
    Difference operator scalogram with strides up to smax, as a sparse matrix.
    """
    sign = 1 if forward == 1 else -1
    #--- Scale indices
    s = torch.arange(smax)[:,None].repeat(1, N).flatten()
    #--- Diagonal indices ---
    ij = torch.arange(N)[None,:].repeat(2, smax)
    sij_diag = torch.stack([ij[0] + N * s, ij[1]])
    #--- Offdiagonal indices ---
    bds = torch.tensor([0, N - 1])
    ij[1] += sign * (1 + s) 
    ij = torch.min(torch.max(ij, bds[0]), bds[1])
    sij_off = torch.stack([ij[0] + N * s, ij[1]])
    #--- Values
    one = torch.ones(ij.shape[1])
    values = -sign * torch.cat([one, -one])
    #--- Sparse matrix
    indices = torch.cat([sij_diag, sij_off], 1)
    return torch.sparse_coo_tensor(indices, values, size=[smax * N, N])


def scholkmann(smax, N):
    """ 
    Return the Scholkmann algorithm up to scale smax on N data points.

    The returned function computes the 'local maxima scalogram' 
    used in the Scholkmann algorithm. Difference operators
    are cached as sparse matrices for memory efficiency and 
    performance. 
    Truncate the scalogram to discard boundary artifacts.

    > Bishop and Ercole, 2018: Multi-Scale Peak and Trough Detection
      Optimised for Periodic and Quasi-Periodic Neuroscience Data.
    """
    mm = torch.sparse.mm
    Df = diff_scalogram(smax, N, 1)
    Db = diff_scalogram(smax, N, -1)
    def LMS (x, peak=1):
        """ Local Extrema Scalogram. """
        Dfx = torch.sign(Df @ x)
        Dbx = torch.sign(Db @ x)
        extrema = (Dfx * Dbx) == -1
        slope   = (Dbx == (1 if peak == 1 else -1))
        mask    = extrema * slope
        return mask.view([smax, N])
    return LMS


class Troughs:
    
    def __init__(self, Npts, smax=40):
       self.LMS   = scholkmann(smax, Npts)
       self.Npts  = Npts
       self.scale = smax
    
    def filtered(self, icp, fs=100):
        #--- Find respiratory and cardiac Rates
        fr, fc = rates(icp, fs=fs)
        #--- Filtered ICP
        icpf = bandpass(0.65, 15, 100, self.Npts // 2 + 1)(icp)
        #--- Trough width 
        width = int(fs / (2 * fc))
        #--- Diastoles
        is_trough = self.LMS(icpf, -1)[:width,:].prod(dim=0) == 1
        troughs = torch.nonzero(is_trough).reshape([-1])
        return troughs

    def __call__(self, icp, fs=100, width=None):
        width = width if width else self.scale
        #--- Diastoles
        is_trough = self.LMS(icp, -1)[:width,:].prod(dim=0) == 1
        troughs = torch.nonzero(is_trough).reshape([-1])
        return troughs
