import torch
import revert.plot as rp

from revert.models import ConvNet, KMeans

model = ConvNet.load("apr12-1.pt")
km    = KMeans(64)

d = torch.load("../scripts-infusion/baseline.pt")
x = d['pulses'].view([-1, 128])
y = model(x).detach()

# K-Means loop
km.fit(y.cuda()).cpu()

# Subset of pulses
x1 = d['pulses'][:,:6].reshape([-1, 128])
y1 = model(x1)

