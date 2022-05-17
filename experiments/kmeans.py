import torch
import revert.plot as rp

from revert.models import KMeans, Module, Pipe, View
from revert.cli    import join_envdir

#--- KMeans instance
km    = KMeans(32)
#--- load model state
twins = Module.load("twins/VICReg-64:8:64-may12-1.pt")
model = Pipe(*twins.model.modules[:2], View([8]))
model.cuda()

def main(): 
    #--- image of pulse dataset
    d = torch.load(join_envdir("INFUSION_DATASETS", "baseline-no_shunt.pt"))
    x = d['pulses'].view([-1, 128]).cuda()

    with torch.no_grad(): 
        y = model(x).detach()

    #--- K-Means loop
    km.fit(y.cuda()).cpu()
