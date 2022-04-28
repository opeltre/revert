import torch

def unshuffle(x, y) :
    """
    Unshuffle a batch of channels, applying the inverse of y.
    
        Inputs :
            - x  : (N, Nc, Npts) or (Nc, Npts) tensor 
            - y  : (N, Nc, Npts) or (Nc, Npts) long tensor, representing permutations.
        Output :
            - x' : action on x of the inverse of y. 
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
    Shuffle channels by random permutations. 
    
        Inputs :
            - x : (N, Nc, Npts) or (Nc, Npts) tensor
        Output :
            - x' : like x with randomly shuffled channels
            - y  : applied permutations as (N, Nc, Npts) or (Nc, Npts) long tensor
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
    Transpose two random channels. 
    
        Inputs :
            - x : (N, Nc, Npts) or (Nc, Npts) tensor
        Output :
            - x' : like x with two swapped channels 
            - y  : applied permutations as (N, Nc, Npts) or (Nc, Npts) long tensor
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
