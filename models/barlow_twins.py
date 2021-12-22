import torch
import torch.nn as nn

def norm2 (t, dim=None):
    """ L2-norm on specified dimensions """
    return torch.sqrt((t ** 2).sum(dim)) 

def cross_correlation (ya, yb):
    """ Cross-correlation of N_batch x N """
    ya, yb = ya - ya.mean([0]), yb - yb.mean([0])
    yab = ya[:,:,None] @ yb[:,None,:]
    return yab.sum(dim=[0]) / (norm2(ya, [0]) * norm2(yb, [0]))

class BarlowTwins (nn.Module):

    def __init__(self, model, offdiag=0.5):
        """ Create twins from a model. """
        super().__init__()
        self.model   = model
        self.offdiag = offdiag 
        self.writer  = False

    def forward (self, x):
        """ Apply twins to 2 x N_batch x N tensor. """
        xa, xb = x
        ya, yb = self.model(xa), self.model(xb)
        return torch.stack([ya, yb])

    def loss (self, y): 
        """ Return Barlow twin loss of N_batch x N output. """
        n_out = y.shape[-1]
        C = cross_correlation(*y) 
        I = torch.eye(n_out)
        w = self.offdiag
        loss_mask = w * torch.ones(C.shape) + (1 - w) * I
        return torch.sum(((C - I) * loss_mask) ** 2) / (2 * n_out)

    def cross_corr (self, x):
        """ Cross correlation matrix of twin outputs. l"""
        return cross_correlation(*self(x))

    def loss_on (self, x):
        """ Barlow twin loss on input """
        return self.loss(self(x))

    def fit (self, x, lr=1e-2, br=1e-3, n_batch=128, w="Loss/fit"):
        """ Fit on a 2 x N_samples x N tensor. """
        n_it = x.shape[1] // n_batch
        print(f"Fitting on {n_it} * {n_batch} pairs...")
        for s in range(n_it):
            y = self.forward(x[:,s:s + n_batch])
            loss = self.loss(y)
            loss.backward()
            if w: 
                self.write(w, loss, nit=s)
            with torch.no_grad(): 
                for p in self.parameters(): 
                    p -= p.grad * lr
                    p -= br * torch.randn(p.shape)
                self.zero_grad()
        return self

    def loop (self, xs, lr=1e-2, br=1e-3, n_batch=128, w="Loss/fit", epochs=1):
        lr = lr if isinstance(lr, list) else [lr] * epochs
        br = br if isinstance(br, list) else [br] * epochs
        lr = lr + [lr[-1]] * (epochs - len(lr))
        br = br + [br[-1]] * (epochs - len(br))
        for e in range(epochs):
            ys   = self(xs)
            l0   = self.loss(ys)
            print(f"\nLoss {e}: {float(l0):.4f}")
            self.fit(xs, lr[e], br[e], n_batch, w=f'{w}{e}')
        ys   = self(xs)
        l1   = self.loss(ys)
        print(f"\nLoss {e}: {float(l1):.4f}")
        return self



    def write(self, name, data, nit):
        if self.writer:
            self.writer.add_scalar(name, data, global_step=nit)






        
         

