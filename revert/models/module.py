import torch
import torch.nn as nn

class Module (nn.Module):
    """ Module subclass for writing to tensorboard during training. """
    
    def loss_on (self, x, *ys):
        """ Model loss on input """
        try: 
            loss = getattr(self, 'loss')
        except:
            raise RuntimeError("'model.loss' is not defined")
        return self.loss(self.forward(x), *ys)
    
    def fit (self, xs, optimizer=None, scheduler=None, epochs=1, w=None, nw=10):
        """ Fit on a N_it x 2 x N_batch x N tensor. 
            
            The iterable 'xs' should yield either tensor / tuple of tensor batches,
            see torch.utils.data.TensorDataset for instance. 
        """
        N_it = len(xs)
        for e in range(epochs):
            l = 0
            for nit, x in enumerate(xs): 
                optimizer.zero_grad()
                loss = (self.loss_on(x) if isinstance(x, torch.Tensor)
                        else self.loss_on(*x))
                loss.backward()
                optimizer.step()
                l += loss.detach()
                if w and nit % nw == 0 and nit > 0:
                    self.write(w, l / nw, nit + e * N_it) 
                    l = 0
            if scheduler:
                scheduler.step()
        return self

    def write(self, name, data, nit):
        """ Write a scalar to tensorboard."""
        if self.writer:
            self.writer.add_scalar(name, data, global_step=nit)