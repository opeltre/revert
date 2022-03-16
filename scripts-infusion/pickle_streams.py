""" Write raw ICP signal to .pt format. 

    The torch pickled format is ~3000 x faster to read. 

        $ python script.py src dest
"""

import sys
import os
import torch

from tqdm import tqdm
from revert import infusion

src  = sys.argv[1] if len(sys.argv) > 1 else 'full'
dest = sys.argv[2] if len(sys.argv) > 2 else f'{src}_pt' 

db = infusion.Dataset(src)

# path relative to $INFUSION_DATASETS directory
if not dest[0] == '/':
    dbdir = os.path.split(db.path)[0]
    dest = os.path.join(dbdir, dest)

# make dest dir
if not os.path.exists(dest): os.mkdir(dest)

# error if dest is a file
if not os.path.isdir(dest): raise RuntimeError(f"{dest} is not a directory")

print(f"Writing .pt signals to {dest}")

# Save .pt streams 
for k in tqdm(db.ls()):
    f = db.get(k)
    try:
        icp = f.icp()
        path = os.path.join(dest, f'{f.key}.pt')
        torch.save(icp, path)
    except: 
        print(f"Error looking for waves/icp in {k}")
    f.close()
