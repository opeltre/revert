import os
import torch
import pandas as pd

from infusion.data import Pulses, PulseDir, Results

fields = {
    "E":    "Elastance [1/ml]",
    "PVI":  "PVI [ml]"
}

inf_datasets = os.environ["INFUSION_DATASETS"]

dir = PulseDir("pulses_full")
res = Results("results-full.json")

#--- Filter keys ---

df = res.filtered()
ks = list(set(dir.keys) & set(df.index))
df = df.loc[ks]
pulses = dir.select(ks)

#--- Output dict ---

out = {"pulses": pulses}
dest = inf_datasets + "/pulses_filtered"

for k, fk in fields.items():
    out[k] = torch.tensor(df[fk].to_numpy())
    dest += f"_{k}"
dest += ".pt"

torch.save(out, dest)
