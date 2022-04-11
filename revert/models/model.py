import torch
import torch.nn as nn

class Model (nn.Module):
    
    def loss_on (self, x):
        """ Model loss on input """
        try: 
            loss = getattr(self, 'loss')
        except:
            raise RuntimeError("'model.loss' is not defined")
        return loss(self.forward(x))
    
    def optimize (self, xs, optimizer, scheduler=None, epochs=1, w=None, nw=10):
        """ Fit on a N_it x 2 x N_batch x N tensor. """
        N_it = xs.shape[0]
        for e in range(epochs):
            l = 0
            for nit, x in enumerate(xs): 
                optimizer.zero_grad()
                loss = self.loss_on(x)
                loss.backward()
                optimizer.step()
                if w and nit % nw == 0:
                    l = (l + loss.detach()) / nw
                    self.write(w, l / nw, nit + e * N_it) 
                else: 
                    l += loss.detach()
            if scheduler:
                scheduler.step()
        return self

    def write(self, name, data, nit):
        """ Write a scalar to tensorboard."""
        if self.writer:
            self.writer.add_scalar(name, data, global_step=nit)
