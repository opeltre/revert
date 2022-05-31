# Look for results in aux/ICM+/icmtests
import json
import csv

from revert.infusion import Dataset
from sys import argv

# Input dataset $1
dbname = argv[1] if len(argv) > 1 else 'full'
db = Dataset(dbname)

# Dump to $2 or "datasets/results/results-$1.json"
# Dump as CSV if $2 has ".csv" extension
dest   = (argv[2] if len(argv) > 2 
        else db.path.replace(dbname, f"results-{dbname}.json"))

print(argv)

# Dict of infusion results read from 'aux/ICM+/icmtests'
results = db.map(lambda f: f.results())

# Harmonise fields
keymap = {
    'Elasticity [1/ml]' : 'Elastance [1/ml]',
    'Elastance []'      : 'Elastance [1/ml]',
    'pss []'            : 'Pss [mmHg]',
    'pss [mmHg]'        : 'Pss [mmHg]'
}

def relabel (row, keymap=keymap):
    label = lambda k : keymap[k] if k in keymap else k
    return {label(k): rk for k, rk in row.items()}

results = {key: relabel(row) for key, row in results.items()}


# JSON
if dest[-5:] == ".json":
    with open(dest, "w") as out:
        json.dump(results, out, indent=2)

# CSV
elif dest[-4:] == ".csv":
    with open(dest, "w") as out:
        fields = {"key": "x"}
        for k, r in results.items(): 
            fields |= r 
            r["key"] = k
        w = csv.DictWriter(out, list(fields.keys()))
        w.writeheader()
        w.writerows(results.values())

print(f"--- Looking for results in icmtests:\n"
    + f"mapped {len(results)} files "\
    + f"({len(results) * 100 / len(db.ls()):.1f}%)\n"
    + f"wrote output to {dest}")
