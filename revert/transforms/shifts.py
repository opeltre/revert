import torch

def shift_all(stdev):
    """
        Inputs :
            - stdev : standard deviation
            - x : list of channels
        Output :
            - x_prime : the same list of channels (with all of them shifted)
            - y : list of all size of shift for each channel
    """
    def run_shift(x):
        N = len(x)
        Nc = x.shape[1]
        Npts = x.shape[-1]
        
        # generate and convert to tensor
        idx = torch.arange(Npts).repeat(Nc*N).view([Nc*N, Npts])
        # generate the guass distribution
        y = torch.randn([N, Nc]) * stdev
        y = mod(y, 1)
        y = (y - y.mean([1])[:,None])

        y_index = (y * (Npts / 2)).flatten().long() 

        idx = (idx + y_index[:,None]) % Npts
        idx = (torch.arange(Nc*N)[:,None] * Npts + idx).flatten()
        x_prime = x.flatten()[idx].view([N, Nc, Npts])
        return x_prime, y
    return run_shift

def shift_one(x):
    """
        Input:
            - x corresponds to a list of channels
        Outputs:
            - x corresponds to the same list of channels (with one of them shifted)
            - y corresponds to the list of shifts
    """
    if x.dim() == 2:
        y = torch.tensor([0 for i in range(len(x))])
        nb_chan = torch.randint(0, len(x), (1,))
        half = x.shape[-1] // 2
        shift = torch.randint(-half, half+1, (1,)).item()
        x[nb_chan] = x[nb_chan].roll(-shift, -1)
        y[nb_chan] = shift
        return x, y
    elif x.dim() == 3:
        return torch.stack([shift_one(xi) for xi in x])
    else:
        raise TypeError("x must be of shape (N, Nc, Npts) or (Nc, Npts)")
        
def mod(x, m) :
    return torch.sign(x) * ((torch.sign(x)*x) % m)