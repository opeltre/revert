import torch

def segment(val, cuts, N=128, before=0):
    """ 
    Return a (segment, mask) pair of shape (n_cuts, N). 
    
    The number of interior cuts returned is `n_cuts = cuts.shape[0] - 1`.
    """
    Npts = before + val.shape[0] + N
    idx  = cuts[:-1][:,None] + torch.arange(N)[None,:]
    idx  = idx.view([-1])
    #--- segments of shape (n_cuts, N)
    padded = torch.cat([val[0] * torch.ones([before]),
                        val, 
                        val[-1] * torch.ones([N])])
    out    = padded[idx].view([-1, N]) 
    #--- segment masks
    n_cut = (torch.bucketize(idx, cuts, right=True)
                  .view([-1, N]))
    n_tgt = 1 + torch.arange(0, n_cut.shape[0])[:,None]
    mask  = (n_cut == n_tgt).long()
    #--- return segment, mask 
    return out, mask

def mask_center(x, mask, output=None):
    """ 
    Center a batch of segments up to order 1 in the last dimension.

    If output=True, return a (centered, means, slopes) triplet.
    """
    N = x.shape[-1]
    # means
    means = (x * mask).sum([1]) / mask.sum([1])
    # boundary values
    dmask = torch.diff(mask)
    bndry = (dmask == -1).long()
    x0, x1 = x[:,0], (x[:,:-1] * bndry).sum([1])
    # slopes
    t = mask.sum([1]) - 1
    v  = (x1 - x0) / t
    vt = v[:,None] * torch.arange(N)[None,:]
    vm = (x1 - x0) / 2
    out = ((x - vt) * mask 
        +  x0[:,None] * (1 - mask))
    out += vm[:,None] - means[:,None]
    if not isinstance(output, str):
        return out
    if output == 'means':
        return out, means
    if output == 'slopes':
        return out, means, v
