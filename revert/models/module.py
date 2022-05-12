import torch
import torch.nn as nn

import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

class Module (nn.Module):
    """
    Module subclass.

    Module objects f, g implement composition as:

        f @ g = Pipe(g, f)

    Any module object f implementing a loss method of the form

        f.loss : (y, *tgts) -> float

    will inherit a working fit method that backpropagates its gradient
    on a dataset d = (x, *tgts).

    The methods f.epoch and f.episode can be used as decorators to
    register writing callbacks that will be called during training, e.g.

        @f.epoch
        def write_stdev(tag, epoch):
            f.write(tag, f(data_val).dev(0).mean(), epoch)

    Adding the decorator will register `write_stdev` in the
    module's `.epoch_callbacks` attribute.
    """
    def __init__(self) :
        super().__init__()
        self.writer = {}
        self.epoch_callbacks   = []
        self.episode_callbacks = []

    def loss_on (self, x, *ys):
        """ Model loss on input """
        if not "loss" in self.__dir__():
            raise RuntimeError("'model.loss' is not defined")
        return self.loss(self.forward(x), *ys)

    def fit (self, xs, optim=None, lr=None, epochs=1, tag=None, val=None, *kws):
        """ 
        Fit on a N_it x 2 x N_batch x N tensor.

        The iterable 'xs' should yield either tensor / tuple of tensor batches,
        see torch.utils.data.TensorDataset for instance.
        """
        #--- number of steps between calls to writer
        mod = kws["mod"] if "mod" in kws else 10
        #--- optimizer and scheduler
        if optim and lr:
            scheduler = lr
        elif isinstance(lr, float):
            optim = torch.optim.Adam(lr)
            scheduler = None
        N_it = len(xs)
        #--- loop over epochs
        for e in tqdm(range(epochs), position=0, desc='epoch', colour="green"):
            l, ntot = 0, len(xs)
            #--- loop over batches
            for nit, x in enumerate(xs):
                #--- backprop
                optim.zero_grad()
                loss = (self.loss_on(x) if isinstance(x, torch.Tensor)
                        else self.loss_on(*x))
                #--- write callback
                if tag: 
                    l += loss.detach()
                if tag and nit % mod == 0 and nit > 0:
                    self.write(f"Loss/{tag}", l / mod, nit + e * N_it)
                    l = 0
                #--- backprop
                loss.backward()
                optim.step()
            #--- learning rate decay
            if scheduler:
                scheduler.step()
            #--- epoch callback
            data = {'train': xs, 'val': val}
            for cb in self.epoch_callbacks: cb(tag, data, e)
        #--- episode callback
        for cb in self.episode_callbacks:   cb(tag, data)
        #--- 
        self.free(optim, scheduler)
        self.free(data)
        return self
    
    def epoch(self, callback):
        """ 
        Run callback("tag", data, epoch) after each epoch. 

        The data argument will be a dict with 'train' and 'val' values.
        """
        self.epoch_callbacks.append(torch.no_grad()(callback))

    def episode(self, callback):
        """ 
        Run callback("tag", data) after each episode. 

        The data argument will be a dict with 'train' and 'val' values.
        """
        self.episode_callbacks.append(torch.no_grad()(callback))

    def write(self, name, data, nit):
        """ Write a scalar to tensorboard / module.writer dict. """
        if isinstance(self.writer, dict):
            if name in self.writer :
                self.writer[name]["Val"].append(data.item())
                self.writer[name]["Step"].append(nit)
            else :
                self.writer[name] = {"Val" : [], "Step" : []}
                self.write(name, data, nit)
        elif isinstance(self.writer, SummaryWriter):
            self.writer.add_scalar(name, data, global_step=nit)

    def write_dict(self, data):
        """ Write key value pairs to tensorboard / module.writer dict. """
        writer = self.writer
        if isinstance(writer, dict) :
            # save the hyper parameter to writer dict
            for key, value in data.items():
                writer[key] = str(value)
        elif isinstance(writer, SummaryWriter) :
            # save the hyper parameter to tensorboard
            for key, value in data.items():
                writer.add_text(key , str(value))

    def __matmul__ (self, other):
        """ Composition of modules. """
        if isinstance(other, Pipe) and isinstance(self, Pipe):
            return Pipe(*other.modules, *self.modules)
        if isinstance(self, Pipe):
            return Pipe(other, *self.modules)
        if isinstance(other, Pipe):
            return Pipe(*other.modules, self)
        return Pipe(other, self)
 
    def write_to(self, path=None):
        if isinstance(path, str):
            self.writer = SummaryWriter(path)

    @classmethod
    def load (cls, path, env="REVERT_MODELS"):
        """
        Load module state, checking class name.

        If the environment variable `env` is defined, then
        relative paths will be understood from it.

        """
        if not os.path.isabs(path) and env in os.environ:
            path = os.path.join(os.environ[env], path)
        data = torch.load(path)
        if not isinstance(data, cls):
            raise TypeError(f'Loaded data is not of type {cls}')
        return data

    def save (self, path, env="REVERT_MODELS"):
        """
        Save module state.

        If the environment variable `env` is defined, then
        relative paths will be understood from it.
        """
        if isinstance(self.writer, SummaryWriter):
            self.writer.close()
        if not os.path.isabs(path) and env in os.environ:
            path = os.path.join(os.environ[env], path)
        torch.save(self, path)
    
    def free (self, *xs):
        for x in xs: del x
        torch.cuda.empty_cache()


class Pipe (Module):
    """ Composition of models, analogous to torch.nn.Sequential. """

    def __init__(self, *modules):
        """
        Pipe modules in sequential order.

        The last model should have a loss function attached for fitting.
        """
        super().__init__()
        self.modules = modules

        for i, mi in enumerate(self.modules):
            setattr(self, f'module{i}', mi)

    def forward (self, x):
        xs = [x]
        for f in self.modules:
            xs.append(f(xs[-1]))
        y = xs[-1]
        del xs
        torch.cuda.empty_cache()
        return y

    def loss(self, y, *ys):
        return self.modules[-1].loss(y, *ys)
