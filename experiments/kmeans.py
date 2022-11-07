import os
import torch
from torch.utils.tensorboard import SummaryWriter

import revert.plot as rp

from revert.models import KMeans, Module, Pipe, View, cross_correlation
from revert        import cli

parser = cli.arg_parser()
parser.add_argument('--k', '-k', type=int, metavar='number of clusters')
parser.add_argument('--col', '-C', type=int, metavar='grid columns')

defaults = dict(input='../twins/VICReg-64:16-nov3-1.pt',
                output='KMeans:{k} @ {input}', 
                datatype='infusion', 
                data='baseline-full.pt',
                k=64,
                col=8,
                dy=16,
                dirname='kmeans')

rp.plt.style.use('seaborn')

def setArgs(args):
    name = os.path.basename(args.input).replace('.pt', '')
    args.output = args.output.replace('{k}', str(args.k)).replace('{input}', str(name))
    args.writer = args.writer.replace('{k}', str(args.k)).replace('{input}', str(name))
    args.shape = (args.col, args.k // args.col)
    return args
    
args = setArgs(cli.read_args(parser, **defaults))

#--- KMeans instance
km = KMeans(args.k)
km.writer = SummaryWriter(args.writer)

#--- load model state
twins = Module.load(args.input)
model = twins.model
model.cuda()

#--- load pulses
d = torch.load(args.data)
x = d['pulses'].view([-1, 128]).cuda()

@km.episode
def cluster_grid(tag, data, P=6):
    y = model(x)
    fig = rp.kmeans_grid(km, x, y, P=P, shape=args.shape)
    km.writer.add_figure("Clusters", fig)

@km.episode
def cluster_xcorr(tag, data):
    ys = model(x).view([len(d['pulses']), -1, args.dy])
    p = torch.stack([km.counts(yi) / yi.shape[0] for yi in ys]).cpu()
    print(p.shape)
    C = cross_correlation(p, p)
    print(C)
    km.writer.add_image("Cluster correlation", (1 + C) / 2, dataformats="HW")

def main(): 
    #--- image of pulse dataset
    with torch.no_grad(): 
        y = model(x).detach()

    #--- K-Means loop
    km.fit([y.cuda()] * 10, epochs=100, mod=5, tag="average cluster variance")
    
    km.save(args.output)

if __name__ == '__main__':
    main()
