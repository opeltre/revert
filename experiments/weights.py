from twins import model
import torch
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt

st = torch.load("trained.state")
model.load_state_dict(st)

w = model.conv0.weight.detach()

for wi in w:
    plt.plot(wi.view([-1]))

plt.show()
