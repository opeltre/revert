import torch

def unshuffle(x, y) :
    """
    Take all the data and unshuffled them.
    
        Inputs :
            - x : (N, Nc, Npts) tensor (can also take (Nc, Npts) tensor)
            - y : (N, Nc, Npts) tensor with all new index of each channel (can also take (Nc, Npts) tensor)
        Output :
            - x : x unshuffled by y 
    """
    dim = len(x.shape)

    if (dim == 2) :
        x.unsqueeze_(0)
        y.unsqueeze_(0) 

    N = x.shape[0]
    Nc = x.shape[1]
    Npts = x.shape[-1]

    y = y.sort().indices
    idx = (torch.arange(N)[:,None] * 6 + y).flatten()
    x = x.view([N * Nc, Npts])
    x_prime = x.index_select(0,idx)
    x_prime = x_prime.view([N, Nc, Npts])

    if (dim == 2) :
        return x_prime.squeeze(0)
    return x_prime

def shuffle_all(x) :
    """
    Take all the data and shuffle each channel
    
        Inputs :
            - x : (N, Nc, Npts) tensor (can also take (Nc, Npts) tensor)
        Output :
            - x : same list but all channel shuffle on each row
            - y : each new index for each channel
    """
    dim = len(x.shape)

    if (dim == 2) :
        x.unsqueeze_(0)
    N = x.shape[0]
    Nc = x.shape[1]
    Npts = x.shape[-1]

    y = torch.randint( torch.iinfo (torch.int64).max, (N,Nc)).argsort()
    idx = (torch.arange(N)[:,None] * 6 + y).flatten()
    x = x.view([N * Nc, Npts])
    x_prime = x.index_select(0,idx)
    x_prime = x_prime.view([N, Nc, Npts])

    if (dim == 2) : 
        return x_prime.squeeze(0), y
    return x_prime, y
        
def shuffle_two(x) :
    """
    Take all the data and take two random channel and swap them
    
        Inputs :
            - x : (N, Nc, Npts) tensor (can also take (Nc, Npts) tensor)
        Output :
            - x : same list but two channel swap on each row
            - y : each new index for each channel
    """
    dim = len(x.shape)
    
    if (dim == 2) :
        x.unsqueeze_(0)
    N = x.shape[0]
    Nc = x.shape[1]
    Npts = x.shape[-1]

    y =  torch.arange(Nc).repeat(N).view([N,Nc])
    for i, xi in enumerate(x) :
        rand1 = torch.randint(Nc, (1,1))
        rand2 = torch.randint(Nc, (1,1))
        y[i][rand1] = rand2
        y[i][rand2] = rand1

    idx = (torch.arange(N)[:,None] * 6 + y).flatten()
    x = x.view([N * Nc, Npts])
    x_prime = x.index_select(0,idx)
    x_prime = x_prime.view([N, Nc, Npts])

    if (dim == 2) : 
        return x_prime.squeeze(0), y
    return x_prime, y