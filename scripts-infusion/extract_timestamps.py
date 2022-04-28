""" Extract timestamps from a dataset: 

        $ python ./periods.py 2016 periods-2016.json 
"""
import json
import sys 

from revert.infusion import Dataset

#--- CLI arguments ---

dbname  = sys.argv[1] if len(sys.argv) > 1 else '2016'
out     = sys.argv[2] if len(sys.argv) > 2 else f'./periods-{dbname}.json'
db      = Dataset(dbname)
Ntot    = len(db.ls())

print(f"--- Database '{dbname}': {Ntot} files")

#--- Recording timestamps --- 

start  = db.map(lambda f: f.start())
samp   = db.map(lambda f: f.fs())
Nindex = min(len(start), len(samp))
print(f"\n--- Looking through icp.index for 'starttime':\n"\
    + f"\t{Nindex} files ({100 * len(samp)/Ntot:.1f}%)")

#--- Events ---

print(f"\n--- Looking for infusion events: ")

sources = ["annotations", "icmtests", "icmevents"]
events  = {si: db.map(lambda f: f.intervals(src=si)) for si in sources}

def join(*evts):
    out = {}
    for e in evts: 
        for k, ek in e.items():
            if k in out: out[k] |= ek
            else:        out[k] = ek
    return out

evt_join = join(*[events[si] for si in sources])

files = list(evt_join.keys())
Nfilt = len(files)

#--- Periods data ---

data = {
    f : {
        "samp"      : samp[f] if f in samp else "NA",
        "start"     : start[f] if f in start else "NA"
    } | evt_join[f]
    for f in files
}

#--- Print ratios --- 

Ns = [len(events[si]) for si in sources]
print(f"\t{len(files)} files ({(100 * Nfilt / Ntot):.1f}%)\n")
print(f"\t| annotations |   icmtests  |  icmevents  |")
print("\t" + f"".join([f"     {100*Ni/Ntot:.1f}%    " for Ni in Ns]))

#--- Dump data ---

print(f"\n--- Writing output to {out}")
with open(f"{out}", "w") as out:
    json.dump(data, out, indent=4)

#--- Check conflicts --- 

print(f"\n--- Checking intersections and conflicts\n")
keys  = {s: set(e.keys()) for s, e in events.items()}
pairs = [[s1, s2] for i, s1 in enumerate(sources) for s2 in sources[:i]]

for s1, s2 in pairs:
    Ncup = len(keys[s1] | keys[s2])
    Ncap = len(keys[s1] & keys[s2])
    print(f"\t{s1} U {s2}:\t{Ncup} files ({100 * Ncup/Ntot:.1f}%)")
    print(f"\t{s1} ^ {s2}:\t{Ncap} files ({100 * Ncap/Ntot:.1f}%)")
    Ndiff = 0
    for k in keys[s1] & keys[s2]: 
        if events[s1][k] != events[s2][k]:
            Ndiff += 1
    print(f"\t> {Ndiff} conflicts\n")

for k, ek in events.items():
    names = set()
    for f, ekf in ek.items():
        names |= set(ekf.keys())
    print(f"--- Keys encountered in {k}:")
    print(json.dumps(list(names), indent=2))
