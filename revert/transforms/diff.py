import torch

def diff (t, step=1) :
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
