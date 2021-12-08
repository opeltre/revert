import icp 
import sig
import notebooks.plot as plot
import matplotlib.pyplot as plt
import torch

""" Load ICP file """
f = icp.file(0)
icp = sig.heat(1.5)(f.icp(2000))

""" Segment Pulses """
segs = sig.segments(icp, heat=(8, 4, 1))

""" Center up to order 2 """
ctr = sig.center(3)
segs = [ctr(s) for s in segs]

fig = plt.figure()
ax = plot.ax3(fig)
for s in segs:
    plot.jet3(s, ax, color="#56a", linewidth=0.3)
plot.show()

import matplotlib.animation as anim 

def rotate(angle): 
    ax.view_init(azim=angle)

print("animating...")
out = anim.FuncAnimation(fig, rotate, frames=torch.arange(0, 362, 2), interval=100)
out.save('phase3.gif', dpi=80, writer="imagemagick")
