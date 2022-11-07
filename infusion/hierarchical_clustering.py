import hdbscan
import revert
import revert.cli as cli
from revert.models import Module, View

import torch
from torch import nn

import matplotlib.pyplot as plt
import revert.plot as rp 

device = 'cpu'

#--- Encoder --- 

dy = 16
encoder = View([dy]) @\
          (Module
           .load("twins/VICReg-64:16-nov3-1.pt")
           .model)
        

encoder.to(device)

#--- Pulses (N x 64 x 128) ---

dataset = torch.load(cli.join_envdir("INFUSION_DATASETS", "baseline-full.pt"))

print(dataset['pulses'].shape)
x = (dataset['pulses'][:,:30]
     .reshape([-1, 128])
     .to(device))

#--- HDBSCAN --- 

clusterer = hdbscan.HDBSCAN(min_cluster_size=30,
                            min_samples=20)

with torch.no_grad():
     y = encoder(x)

z = (clusterer.fit(x)
              .labels_)

print(f'{tuple(y.shape)}: latent point cloud shape')
print(f'{int(z.max())} clusters')

#--- Cluster grid --- 

fig = rp.cluster_grid(x, z, 16)
plt.show()

clusterer.condensed_tree_.plot(select_clusters=True)
plt.show()
