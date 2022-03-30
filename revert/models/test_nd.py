from neural_decomposition import ND
from math import pi
import torch
import matplotlib.pyplot as plt

t = torch.linspace(0,  2 * pi, 40)
x = torch.sin(t)

nd = ND(5).init(x)

y = []

for i in range(40):
    nd.fit(x, lr=0.002)
    y += [nd(t).detach()]

for yi in y:
    plt.plot(yi)

plt.plot(x, color="black", linestyle="dashed")
plt.show()
