import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt

from revert import infusion, cli
from revert.models import ConvNet, Module, Pipe, Affine, View

dx = 64
dy = 16

layers_head = [[dy, 1],
               [1 , 1],
               [1]]

@cli.args(dirname   = "compliance",
          datatype  = "infusion",
          data      = "baseline-no_shunt.pt",
          dbname    = "no_shunt",
          input     = "VICReg-64:16:64-jun3-1.pt",
          output    = "conv-64:1-jun3-6.pt",
          device    = "cuda:0",
          Cmax      = 5,
          layers_head = layers_head)

def main(args):
    # load datasets of pulse, compliance pairs 
    dset, val = getData(args, Cmax=args.Cmax)
    # pipe pretrained network with projection head
    model   = getModel(args)
    encoder = model[:2]
    # freeze backbone during first episode
    encoder.freeze()
    model.loss = F.mse_loss
    model.fit(dset, lr=1e-5, epochs=60, n_batch=256, tag="episode 0", val=val)
    encoder.unfreeze()
    # optimize full model during second episode
    model.fit(dset, lr=1e-5, epochs=60, n_batch=256, tag="episode 1", val=val)
    model.save(args.output)

def getModel(args):
    """ Return pretrained twin network's backbone. """

    # load pretrained twin networks
    twins   = Module.load(args.input)
    encoder = twins.model[:2]
    # projection head
    head = Pipe(Conv([[dy, 8, 1],
                      [1,  1, 1],
                      [1,  1]]),
                View([1]),
                Affine(1, 1))
    # full model 
    model = head @ encoder
    
    @model.epoch
    def log_epoch(tag, data, epoch):
        x, y = data['val']
        loss = model.loss_on(x, y).detach().cpu()
        model.write(f'Loss validation/{tag}', loss, epoch)
    
    @model.episode 
    def log_episode(tag, data): 
        x, y = data['val']
        fx = model(x)
        fig = plt.figure()
        plt.scatter(y.cpu(), fx.cpu(), .5)
        plt.plot([0, args.Cmax], [0, args.Cmax], lw=.5, color='purple')
        n = int(tag[-1])
        model.writer.add_figure(f'Prediction v. truth', fig, n)

    model.write_to(args.writer)
    model.to(args.device)
    return model
 
def getData(args, Cmax=10):
    """ 
    Return train/validation datasets of (pulses, compliance) pairs. 

    Output: - dset : (x_t, y_t)  :: tuple(Tensor)
            - val  : (x_v, y_v)  :: tuple(Tensor)

    x and y have shapes (N, 128) and (N, 1) respectively, 
    where N = Npatients * 64. 
    """
    # load segmented pulses
    data = torch.load(args.data)
    # hdf5 dataset interface
    db   = infusion.Dataset(args.dbname) 
    # filter dataset based on analysis results
    N = len(data['keys'])
    E, P0, ICP = torch.zeros([3, N])
    keep = torch.zeros([N])
    dic_P0  = db.map(lambda f: f.pss())
    dic_E   = db.map(lambda f: f.elastance())
    dic_ICP = db.map(lambda f: f.baselineICP())
    for i, k in enumerate(data['keys']):
        if k in dic_P0 and k in dic_E and k in dic_ICP:
            E[i]    = dic_E[k]
            P0[i]   = dic_P0[k]
            ICP[i]  = dic_ICP[k]
            keep[i] = 1
    nz = keep.nonzero().flatten()
    # compute compliance
    pulses = data['pulses'][nz]
    C = 1 / (E[nz] * (ICP[nz] - P0[nz]))
    # filter anomalous values
    bad = C.isinf() + C.isnan() + (C < 0) + (C > Cmax)
    nz2 = (~ bad).nonzero().flatten()
    x, y = pulses[nz2].to(args.device), C[nz2].to(args.device)
    # repeat compliance along patient dimension
    y = y.repeat_interleave(x.shape[1]).view([*x.shape[:2], 1])
    # split dataset and shuffle pulses (flatten patient dimensions)
    dset = x[:896].flatten(0, 1), y[:896].flatten(0, 1)
    val  = x[896:].flatten(0, 1), y[896:].flatten(0, 1)
    # shuffle pulses (flatten patient dimension)
    print(dset[0].shape)
    return dset, val

def histogram(x, buckets):
    idx = torch.bucketize(x, buckets)
    acc = torch.zeros([buckets.shape[0] + 2])
    acc.scatter_add_(0, idx, torch.ones(idx.shape))
    return acc
