import torch
import os
import json
import tqdm
import sys

from revert import infusion
from revert import models

if len(sys.argv) > 1:
    dbname = sys.argv[1]
    label  = sys.argv[2] if len(sys.argv) > 2 else "Baseline"
    dest   = f'{label.lower()}-{os.path.basename(dbname)}.pt'

else:
    print("Usage : python extract_pulses.py [label]")
    sys.exit(1)

fs = 100

if "INFUSION_DATASETS" in os.environ and not os.path.isabs(dest):
    dbpath = os.environ["INFUSION_DATASETS"]
    assert os.path.isdir(dbpath)
    dest = os.path.join(dbpath, dest)
   
db = infusion.Dataset(dbname)

if __name__ == '__main__':
    extract_pulses = infusion.ExtractPulses(64)
    extract_pulses.run(dbname, save=dest)