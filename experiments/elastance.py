import torch
import os
import kmeans

from twins         import model
from infusion.data import Pulses
from torch.utils.data import TensorDataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#--- Load --- 

# Pulses, E, PVI
prefix = os.environ["INFUSION_DATASETS"]
data = torch.load(f'{prefix}/pulses_E_PVI.pt')
# Clustering
km = torch.load("_kmeans/km.pt")
# Model
st = torch.load('st/out64/mon3')
model.load_state_dict(st)
model.train(False)

#--- Distribution of E within cluster --- 

x = data["pulses"].view([-1, 128])
y = model(x)
c = kmeans.predict(km, y)

E = (data["E"]
        .unsqueeze(1)
        .repeat(1, 256)
        .view([-1]))

df = pd.DataFrame({"E": E.numpy(), "C": c.numpy()})
pivot = df.pivot(columns="C", values="E")

def seriesE (i):
    return pivot[i][pivot[i].notnull()]

ax = sns.catplot(y="E", col="C", col_wrap=8, kind="violin", 
                 data=df, bw=0.05, scale='area', linewidth=0.5, 
                 height=3, legend=False)
plt.show()

