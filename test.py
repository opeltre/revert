from icp.loader import File
from sig.filter import Heat
import notebooks.plot as plot

f = File(0)
icp = Heat(4)(f.icp(400))

ax = plot.ax3()
plot.jet3(icp, ax)
plot.show()
