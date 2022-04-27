from revert import infusion
import os
import shutil

try: 
    path = os.environ["INFUSION_DATASETS"] 
except:
    raise RuntimeError("Please define $INFUSION_DATASETS environment variable")

srcdir  = os.path.join(path, "full")
destdir = os.path.join(path, "no_shunt")

def is_shunted(file): 
    res = file.results()
    return "Shunt critical ICP [mmHg]" in res

def main(): 
    # create $INFUSION_DATASETS/no_shunt
    if os.path.exists(destdir): 
        raise RuntimeError("$INFUSION_DATASETS/no_shunt exists")
    os.mkdir(destdir)
    # filter $INFUSION_DATASETS/full
    db = infusion.Dataset("full")
    shunted = db.map(is_shunted)
    noshunt = [k for k, v in shunted.items() if not v]
    # create symlinks
    for key in noshunt:
        src  = os.path.join(srcdir, f"{key}.hdf5")
        dest = os.path.join(destdir, f"{key}.hdf5")
        os.link(src, dest)
    print(f"Created {len(noshunt)} symlinks in {destdir}")

if __name__ == "__main__":
    main()
