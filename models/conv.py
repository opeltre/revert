import torch
import torch.nn as nn

class Conv1d(nn.Conv1d):
    """ Periodic Convolution. """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, padding='same', padding_mode='circular', **kwargs)


class ConvNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1d(1, 16, 8)      # 128 x 16
        self.pool1 = nn.MaxPool1d(4)       # 32  x 16 
        self.conv2 = Conv1d(16, 16, 8)     # 32  x 16
        self.pool2 = nn.MaxPool1d(2)       # 16  x 16
        self.conv3 = Conv1d(16, 8, 8)      # 16  x 8
        self.pool3 = nn.MaxPool1d(4)       # 4   x 8 
        self.relu  = nn.ReLU()

    def forward(self, x):
        n_b, n = x.shape
        x0 = x.view([n_b, 1, n])
        x1 = self.pool1(self.relu(self.conv1(x0)))
        x2 = self.pool2(self.relu(self.conv2(x1)))
        x3 = self.pool3(self.relu(self.conv3(x2)))
        return x3.reshape([n_b, -1])
