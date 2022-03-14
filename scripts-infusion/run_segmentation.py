from infusion import Dataset
from infusion import bandpass, Troughs
import torch

db = Dataset('2016')
dest = "segmentation_2016"

N       = 10 * 6000
Npts    = 2000
stride  = 100
Nfreqs  = stride // 2  + 1
smax    = 40
snr_min = 1.5

segmentation = Troughs(Npts, smax)

bp = bandpass(0.5, 5, 100, Nfreqs)
hp = bandpass(5, 40, 100, Nfreqs)
lp = bandpass(0, 25, 100, Npts // 2 + 1)

def norm2 (t): return torch.sqrt(torch.sum(t ** 2))

def main ():
    fs = 100
    for key in list(db.periods.keys())[:10]:
        file = db.get(key)
        icp_raw = file.icp(N, infusion=-20)
        assert(fs - file.fs() < 1)
        slices, kept = clean_raw(icp_raw)
        print(f"{key}:\t {100 * len(slices) / (N // Npts):.0f} % clean")
        print(kept)
        try:
            data = segment(slices)
            if data: torch.save(data, f"{dest}/{key}.seg")
        except Exception as e:
            print(e)
            print(f"{key}: ERROR")
        file.close()


def segment(slices):
    with torch.no_grad():
        d = {
            "Npts"   : Npts,
            "troughs": [n + segmentation(icp) for icp, n in slices],
            "icp"    : torch.stack([icp for icp, n in slices]),
            "starts" : [n for _, n in slices]
        } if len(slices) else None 
    return d

def clean_raw (icp_raw):
    with torch.no_grad():
        slices = []
        kept   = []
        for k in range(N // Npts):
            if (k + 1) * Npts  > icp_raw.shape[0]:
                break
            N0  = k * Npts
            icp = lp(icp_raw[N0:N0 + Npts])
            keep = True
            if torch.min(icp) < 0 or torch.max(icp) > 50:
                keep = False 
                kept += [f"range ({min(icp):.0f},{max(icp):.0f})"]
                continue
            for s in range(Npts // stride):
                i = stride * s
                icp_s = icp[i:i+stride]
                snr   = norm2(bp(icp_s)) / norm2(hp(icp_s))
                if snr < snr_min:
                    keep = False
                    kept += [f"snr: {snr}"]
                    break
            if keep: 
                kept += ["+"]
                slices += [(icp, k * Npts)]
    return slices, kept

if __name__ == '__main__':
    main()
