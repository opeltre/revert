import torch

def unshift(x, y):
    """
    Action of channel-wise translations. 

        Inputs: 
            - x : (N, Nc, Npts) tensor
            - y : (N, Nc) tensor in the range [-1, 1]
        Output:
            - x_prime : x shifted by -y, channel wise
    """
    N = len(x)
    Nc = x.shape[1]
    Npts = x.shape[-1]
    # generate and convert to tensor
    idx = torch.arange(Npts).repeat(Nc*N).view([Nc*N, Npts])
    y = -y
    y_index = (y * (Npts / 2)).flatten().long() 
    idx = (idx + y_index[:,None]) % Npts
    idx = (torch.arange(Nc*N)[:,None] * Npts + idx).flatten()
    x_prime = x.flatten()[idx].view([N, Nc, Npts])
    return x_prime

def shift_all(stdev):
    """
    Shift all channels by gaussian random offsets. 

        Inputs :
            - stdev : standard deviation
            - x : list of channels
            - y (optional) : list of all shift size for each channel to be applied
        Output :
            - x_prime : the same list of channels (with all of them shifted)
            - y : list of all shift size for each channel
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


def shift_one(x, y=None):
    """
    Shift one channel by a uniform random offset. 

        Input:
            - x corresponds to a list of channels
            - y (optional) corresponds to the list of shifts to be applied (only one channel should be shifted)
        Outputs:
            - x corresponds to the same list of channels (with one of them shifted)
            - y corresponds to the list of shifts applied (only one channel should be shifted)
    """
    if x.dim() == 2:
        x_prime = x.detach().clone()
        if y is None:
            y = torch.tensor([0 for _ in range(len(x_prime))])
            nb_chan = torch.randint(0, len(x_prime), (1,))
            half = x_prime.shape[-1] // 2
            shift = torch.randint(-half, half+1, (1,))
            y[nb_chan] = shift.item()
        else:
            for nb_chan, shift in enumerate(y):
                if shift.item() != 0:
                    break
        x_prime[nb_chan] = x_prime[nb_chan].roll(-shift.item(), -1)
        return x_prime, y
    elif x.dim() == 3:
        return torch.stack([shift_one(xi, y) for xi in x])
    else:
        raise TypeError("x must be of shape (N, Nc, Npts) or (Nc, Npts)")
        
def mod(x, m) :
    return torch.sign(x) * ((torch.sign(x)*x) % m)
