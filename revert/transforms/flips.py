import torch

def vflip_one(x, y=None):
    """
    Flip one channel vertically.

        Input:
            - x corresponds to a list of channels
            - y (optional) corresponds to a list with the channel on which the flip should be applied
        Outputs:
            - x corresponds to the same list of channels (with one of them flipped)
            - y corresponds to the list with the channel on which the flip has been applied (only one channel should be flipped)
    """
    if x.dim() == 2:
        x_prime = x.detach().clone()
        if y is None:
            y = torch.tensor([0 for _ in range(len(x_prime))])
            nb_chan = torch.randint(0, len(x_prime), (1,))
            y[nb_chan] = 1
        else:
            for nb_chan, flip in enumerate(y):
                if flip.item() != 0:
                    break
        x_prime[nb_chan] = 2*x_prime[nb_chan].mean() - x_prime[nb_chan]
        return x_prime, y
    elif x.dim() == 3:
        return torch.stack([vflip_one(xi, y) for xi in x])
    else:
        raise TypeError("x must be of shape (N, Nc, Npts) or (Nc, Npts)")
