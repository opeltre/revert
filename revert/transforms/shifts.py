import random
import numpy as np
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
    dim = len(x.shape)
    if dim == 2 :
        x.unsqueeze_(0)
    
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
    
    if dim == 2 : 
        return x_prime.squeeze(0)    
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
        dim = len(x.shape)
        if dim == 2 :
            x.unsqueeze_(0)
            
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
        
        if dim == 2 : 
            return x_prime.squeeze(0), y
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
        x_list = []
        y_list = []
        for i, xi in enumerate(x):
            if y != None :
                x_prime, y_prime = shift_one(xi, y[i])
            else :
                x_prime, y_prime = shift_one(xi, y)
            x_list.append(x_prime)
            y_list.append(y_prime)
        return torch.stack(x_list), torch.stack(y_list)
    else:
        raise TypeError("x must be of shape (N, Nc, Npts) or (Nc, Npts)")

def shift_discret(x, ind=1):
    """
    Shift one channels by a 33 shifts possible 

        Inputs :
            - x : list of channels
            - ind : the index of the channel to shift
        Output :
            - x_prime : the same list of channels (the one with the defined index shifted)
            - y : label of shift type applied
    """

    y = np.linspace(-32,32, 33, dtype=np.int)
    if (ind > len(x)) :
        raise ValueError("Index out of range")
    dim = len(x.shape)
    if dim == 2 :
        x.unsqueeze_(0)
            
    N = len(x)
    
    # generate the labels
    y_lab = torch.tensor([random.choice(y) for _ in range(N)])
    labels = (y_lab + 32 ) / 2
    labels = labels.long()

    # generate the shift to be applied
    y_0 = torch.tensor([0]*N)
    a = [y_0]*ind
    a.append(y_lab)
    t = tuple(a)
    y = torch.stack(t, dim=-1)

    x_prime, _ = shift_one(x, y)
    if dim == 2 : 
        return x_prime.squeeze(0), labels
    return x_prime, labels
    
    
       
def mod(x, m) :
    return torch.sign(x) * ((torch.sign(x)*x) % m)
