import argparse
import json
import os
from revert import pcmri
import sys
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-j', help="Map PCMRI files to JSON files", action='store_true')
parser.add_argument('--torch', '-t', help="Map PCMRI files to tensor (.pt)", action='store_true')
args = parser.parse_args()

if not args.json and not args.torch:
    parser.print_help(sys.stderr)
    sys.exit(1)

if args.json:
    os.makedirs('patients_json', exist_ok=True)

db = pcmri.Dataset("full")
pt_flows = []
valid_keys = []
error_keys = []

for patient in tqdm(db.getAll("*")):
    try:
        if args.json:
            data = patient.readAll(fields=["debit", "time"], fmt="dict")
            with open("patients_json/" + patient.id + ".json", 'w') as outfile:
                json.dump(data, outfile, indent=4)
        if args.torch:
            pt_flows.append(patient.flows().tolist())
        valid_keys.append(patient.id)
    except:
        error_keys.append(patient.id)

if args.torch:
    torch.save({"flows": torch.tensor(pt_flows), "keys": valid_keys, "errors": error_keys}, "pcmri_tensor.pt")

print("Valid:", len(valid_keys))
print("Error:", len(error_keys))
