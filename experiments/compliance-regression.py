import torch
from revert import infusion, cli
from revert.models import ConvNet, Module

@cli.args(datatype="infusion",
          data="baseline-no_shunt.pt",
          dbname="no_shunt",
          input="VICReg-64:8:64.pt",
          output="conv-64:1.pt")
def main(args):
    data = torch.load(args.data)
    db = infusion.Dataset(args.dbname) 
    x, y = dataset(data, db)
    return x, y

def dataset(data, db):
    N = len(data['keys'])
    E, P0, ICP = torch.zeros([3, N])
    keep  = torch.zeros([N])
    dic_P0 = db.map(lambda f: f.pss())
    dic_E  = db.map(lambda f: f.elastance())
    dic_ICP = db.map(lambda f: f.baselineICP())
    for i, k in enumerate(data['keys']):
        if k in dic_P0 and k in dic_E and k in dic_ICP:
            E[i] = dic_E[k]
            P0[i] = dic_P0[k]
            ICP[i] = dic_ICP[k]
            keep[i] = 1
    nz = keep.nonzero().flatten()
    pulses = data['pulses'][nz]
    C = 1 / (E[nz] * (ICP[nz] - P0[nz]))
    bad = C.isinf() + C.isnan() + (C < 0) + (C > 100)
    nz2 = (~ bad).nonzero().flatten()
    return pulses[nz2], C[nz2]

def histogram(x, buckets):
    idx = torch.bucketize(x, buckets)
    acc = torch.zeros([buckets.shape[0] + 2])
    acc.scatter_add_(0, idx, torch.ones(idx.shape))
    return acc

