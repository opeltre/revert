from icp.loader import File
from sig.filter import Heat
import notebooks.plot as plot

f = File(0)
icp = Heat(2)(f.icp(400))

plot.jet3(icp)
plot.show()
