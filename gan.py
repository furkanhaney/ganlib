import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Dense

class Discriminator(nn.Module):
    def __init__(self, bn=False, dropout=False, width=1):
        super().__init__()
        self.seq = nn.Sequential(
            Dense(1024, int(1024 * width), bn=bn, dropout=dropout),
            Dense(int(1024 * width), int(512 * width), bn=bn, dropout=dropout),
            Dense(int(512 * width), int(256 * width), bn=bn, dropout=dropout),
            Dense(int(256 * width), 1, act="sigmoid", bn=False, dropout=False),
        )

    def forward(self, images):
        x = images.reshape(images.shape[0], 1024)
        return self.seq(x)

    def __str__(self):
        num_params = sum(p.numel() for p in self.parameters())
        return super().__str__() + "\nTotal Parameters: {:,}\n".format(num_params)

class Generator(nn.Module):
    def __init__(self, noise_dim=100, bn=False, dropout=False, width=1):
        super().__init__()
        self.seq = nn.Sequential(
            Dense(noise_dim, int(256 * width), bn=bn, dropout=dropout),
            Dense(int(256 * width), int(512 * width), bn=bn, dropout=dropout),
            Dense(int(512 * width), int(1024 * width), bn=bn, dropout=dropout),
            Dense(int(1024 * width), 1024, act="tanh", bn=False, dropout=False),
        )

    def forward(self, noise):
        y = self.seq(noise)
        y = y.reshape(y.shape[0], 1, 32, 32)
        return y

    def __str__(self):
        num_params = sum(p.numel() for p in self.parameters())
        return super().__str__() + "\nTotal Parameters: {:,}\n".format(num_params)
