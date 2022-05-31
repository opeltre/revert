import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

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
        self.iter_callbacks    = []

    def loss_on (self, x, *ys, **ks):
        """ Model loss on input """
        if not "loss" in self.__dir__():
            raise RuntimeError("'model.loss' is not defined")
        return self.loss(self.forward(x), *ys, **ks)

    def fit (self, dset, optim=None, lr=None, epochs=1, tag=None, val=None, **kws):
        """ 
        Fit on a N_it x 2 x N_batch x N tensor.

        The iterable 'dset' should yield either tensor / tuple of tensor batches,
        see torch.utils.data.TensorDataset for instance.
        """
        if isinstance(dset, tuple):
            xs = TensorDataset(*dset)
            nb = kws["n_batch"] if "n_batch" in kws else 256
            dset = DataLoader(xs, shuffle=True, batch_size=nb)
        #--- number of steps between calls to writer
        mod = kws["mod"] if "mod" in kws else 10
        #--- hide progress bar
        progress = True if "progress" not in kws else kws["progress"]
        #--- optimizer and scheduler
        if optim and lr:
            scheduler = lr
        elif isinstance(lr, float):
            optim = torch.optim.Adam(self.parameters(), lr)
            scheduler = None
        else:
            scheduler = None
        N_it = len(dset)
        #--- loop over epochs
        for e in (range(epochs) if not progress else 
                  tqdm(range(epochs), position=0, desc='epoch', colour="green")):
            l = 0.
            #--- loop over batches
            for nit, x in enumerate(dset):
                #--- backprop
                optim.zero_grad()
                loss = (self.loss_on(x) if isinstance(x, torch.Tensor)
                        else self.loss_on(*x))
                #--- write callback
                if tag: 
                    l += loss.detach()
                if tag and nit % mod == mod - 1:
                    self.write(f"Loss/{tag}", l / mod, nit + e * N_it)
                    l = 0.
                #--- backprop
                loss.backward()
                optim.step()
                #--- iter callback
                for cb in self.iter_callbacks: cb(tag, x, nit)
            #--- learning rate decay
            if scheduler:
                scheduler.step()
            #--- epoch callback
            data = {'train': dset, 'val': val}
            for cb in self.epoch_callbacks: cb(tag, data, e)
        #--- episode callback
        for cb in self.episode_callbacks:   cb(tag, data)
        #--- 
        optim.zero_grad()
        self.free(optim, scheduler)
        self.free(data)
        return self
    
    def iter(self, callback):
        """ 
        Run callback("tag", data, nit) after each batch iteration. 

        The data argument will be the current batch.
        """
        self.iter_callbacks.append(torch.no_grad()(callback))

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

    def write_to(self, path=None):
        """ Initialize tensorboard writer. """
        if isinstance(path, str):
            self.writer = SummaryWriter(path)

    def freeze (self): 
        """ Freeze parameters. """
        for p in self.parameters():
            p.requires_grad = False
        return self

    def unfreeze (self):
        """ Unfreeze parameters. """
        for p in self.parameters():
            p.requires_grad = True
        return self

    def __matmul__ (self, other):
        """ Composition of modules. """
        if isinstance(other, Pipe) and isinstance(self, Pipe):
            return Pipe(*other.modules, *self.modules)
        if isinstance(self, Pipe):
            return Pipe(other, *self.modules)
        if isinstance(other, Pipe):
            return Pipe(*other.modules, self)
        return Pipe(other, self) 

    def __or__(self, other):
        """ 
        Cartesian product (parallel application).
        """
        if isinstance(other, Prod) and isinstance(self, Prod):
            return Prod(*other.modules, *self.modules)
        if isinstance(self, Prod):
            return Prod(other, *self.modules)
        if isinstance(other, Prod):
            return Prod(*other.modules, self)
        return Prod(other, self) 
        
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
        # close writer
        if isinstance(self.writer, SummaryWriter):
            self.writer.close()
        # unregister callbacks
        self.epoch_callbacks = []
        self.episode_callbacks = []
        # save to $env/path
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
    
    def index_of(self, g):
        """ Index of module g in the pipe. """
        i = None
        for j, gj in enumerate(self.modules):
            if gj == g: 
                i = j
        if isinstance(i, int): 
            return i
        raise RuntimeError(f"Could not find module {g} in pipe.")
        
    def until(self, g, strict=True):
        """ Return pipe section until module g. """
        if isinstance(g, int):
            i = g
        elif isinstance(g, Module):
            i = self.index_of(g)
        x = 0 if strict else 1
        return Pipe(*self.modules[:i+x])
    
    def since(self, g, strict=False):
        """ Pipe section from module g. """
        if isinstance(g, int):
            i = g
        elif isinstance(g, Module):
            i = self.index_of(g)
        x = 1 if strict else 0
        return Pipe(*self.modules[i+x:])

    def __getitem__(self, slc):
        """ Pipe sections. """
        if isinstance(slc, int):
            return self.modules[slc]
        if isinstance(slc, slice):
            i0, i1 = slc.start, slc.stop
            if isinstance(i0, Module):
                i0 = self.index_of(i0)
            if isinstance(i1, Module):
                i1 = self.index_of(i1)
        return Pipe(*self.modules[i0:i1])

        
class Prod (Module):
    """ 
    Cartesian product of modules (parallel application).
    """

    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        for i in range(len(modules)):
            setattr(self, f'module{i}', modules[i])
    
    def forward(self, x):
        return [mi(xi) for mi, xi in zip(self.modules, x)]
        

class Skip (Module):
    """ 
    Apply a module to the (input, output) pair of an other module.class Cat (Module).
    """
    
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, xs):
        d = self.dim
        return self.cat([x.flatten(d) for x in xs], dim=d)


class Branch(Module):

    def __init__(self, n):
        super().__init__()
        self.n = n
    
    def forward(self, x):
        return self.n * [x]
    

class Cut (Module):
    """
    Cut input into specified shapes. 
    """
    def __init__(self, ns=None, dim=1, shapes=None):
        super().__init__()
        self.dim = dim
        if ns:
            self.sizes  = ns
            self.shapes = None
        elif shapes:
            prod = lambda n, *ns: (n * prod(ns) if ns else 1)
            self.sizes  = sum(prod(ni for ni in s) for s in shapes)
            self.shapes = shapes
    
    def forward(self, x):
        xs, begin = [], 0
        shapes = self.shapes
        slc = [slice(None)] * self.dim
        for k, Nk in enumerate(self.sizes):
            xk = x[slc + [slice(begin,begin + Nk)]]
            xs.append(xk.reshape(shapes[k]) if shapes else xk)
            begin += Nk
        return xs


class Cat(Module):
    """ 
    Concatenate inputs along specified dim.
    """

    def __init__(self, dim=1, flatten=False):
        super().__init__()
        self.dim = dim
        self.flatten = flatten

    def forward(self, xs):
        d = self.dim
        return (torch.cat(xs, dim=d) if not self.flatten else
                torch.cat([x.flatten(d) for x in xs], dim=d))


class Stack(Module):
    """ 
    Stack inputs along specified dim.
    """

    def __init__(self, dim=1, flatten=False):
        super().__init__()
        self.dim = dim
        self.flatten = flatten

    def forward(self, xs):
        d = self.dim
        return (torch.stack(xs, dim=d) if not self.flatten else
                torch.stack([x.flatten(d) for x in xs], dim=d))


class Mask(Module):

    def __init__(self, *shapes, stdev=.01):
        super().__init__()
        mask  = 1 + stdev * torch.randn(shapes)
        self.register_parameter('mask', nn.Parameter(mask))
        self.shape = self.mask.shape

    def forward(self, x):
        return self.mask * x


class Sum(Module):
    """
    Sum along a dimension.
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.sum(self.dim)
