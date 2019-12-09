import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Dense

class Discriminator(nn.Module):
    def __init__(self, bn=False, dropout=False):
        super().__init__()
        self.seq = nn.Sequential(
            Dense(1024, 1024, bn=bn, dropout=dropout),
            Dense(1024, 512, bn=bn, dropout=dropout),
            Dense(512, 256, bn=bn, dropout=dropout),
            Dense(256, 1, act="sigmoid", bn=False, dropout=False),
        )

    def forward(self, images):
        x = images.reshape(images.shape[0], 1024)
        return self.seq(x)

    def __str__(self):
        num_params = sum(p.numel() for p in self.parameters())
        return super().__str__() + "\nTotal Parameters: {:,}".format(num_params)

class Generator(nn.Module):
    def __init__(self, noise_dim=100, bn=False, dropout=False):
        super().__init__()
        self.seq = nn.Sequential(
            Dense(noise_dim, 256, bn=bn, dropout=dropout),
            Dense(256, 512, bn=bn, dropout=dropout),
            Dense(512, 1024, bn=bn, dropout=dropout),
            Dense(1024, 1024, act="tanh", bn=False, dropout=False),
        )

    def forward(self, noise):
        y = self.seq(noise)
        y = y.reshape(y.shape[0], 1, 32, 32)
        return y

    def __str__(self):
        num_params = sum(p.numel() for p in self.parameters())
        return super().__str__() + "\nTotal Parameters: {:,}".format(num_params)
