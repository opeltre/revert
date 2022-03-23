import torch
from .bounds import bound

def find_spikes(x, bounds=[-50, 150]):
    """
    Return NaN intervals and a mask vanishing outside NaNs.
   
    Returns:
    --------
        spikes: (N, 2) torch.LongTensor
        mask:   torch.BoolTensor
    """
    # spike indices
    mask  = (x < bounds[0]) + (x > bounds[1])
    idx   = mask.nonzero().flatten()
    if not len(idx): return torch.tensor([]), mask
    # get boundaries
    d_idx = torch.diff(idx) > 1
    true  = torch.tensor([True])
    d1 = torch.cat([true, d_idx])
    d0 = torch.cat([d_idx, true])
    # spike intervals
    spikes = torch.stack([idx[d1], idx[d0]]).T
    return spikes, mask

def filter_spikes(x, bounds=[-50, 150]):
    """
    Return interpolated signal and mask vanishing outside NaNs.
    """
    # find spikes
    spikes, mask = find_spikes(x, bounds)
    if not len(spikes): return x, mask
    # interval boundary neighbours
    neighb = spikes + torch.tensor([-2, 2])
    if mask[0]:  neighb[0, 0]   = neighb[0, 1]
    if mask[-1]: neighb[-1, -1] = neighb[-1, 0]
    # map domain to segment ends
    N = x.shape[0]
    idx = torch.arange(N)
    buckets = torch.cat([torch.tensor([0]), neighb.reshape([-1]), torch.tensor([N])])
    interval = torch.bucketize(idx, neighb.reshape([-1]))
    i0 = bound(buckets[interval], [0, N - 1])
    i1 = bound(buckets[interval + 1], [0, N - 1])
    # interpolation
    x0 = x[torch.max(i0, neighb[0, 0])]
    x1 = x[torch.min(i1, neighb[-1, 1])]
    slope  = (x1 - x0) / (i1 - i0)
    offset = x0 - slope * i0
    x_int  = mask * (slope * idx + offset)
    x_int += ~ mask * x
    return x_int, mask
