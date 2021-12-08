from flux_svd import flux, plot_flux, plt, torch

path = "/home/oli/revert/assets/slides/kickoff/"

f = flux("d3")
t = torch.linspace(0, 1, 64)
c = ["red", "blue", "green"]
labels = ["art", "ven", "csf"]

T = [0, 18, -9]
Tf = torch.stack([f[i].roll(T[i]) for i in range(3)])

Tstyle = { "linestyle":"dashed"}

plt.plot(t, f[0], c[0])
plt.plot(t, f[1], c[1])
plt.plot(t, f[2], c[2])

plt.savefig(path + 'flux3.plt.svg')

"""
for i in range(3):
    plt.plot(t, f[i], color=c[i])
    plt.plot(t, Tf[i], color=c[i], linestyle="dashed")

print(t)
plt.show()
"""
