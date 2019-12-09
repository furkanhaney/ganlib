import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Dense, Conv2D, Upscale, Downscale, GlobalAvgPooling

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            Conv2D(1, 16),
            Conv2D(16, 32),
            Conv2D(32, 32),
            Downscale(),
            Conv2D(32, 64),
            Conv2D(64, 64),
            Downscale(),
            Conv2D(64, 128),
            Conv2D(128, 128),
            Downscale(),
            Conv2D(128, 256),
            Conv2D(256, 256),
            GlobalAvgPooling(),
            Dense(256, 1, bn=False, act=torch.sigmoid)
        )

    def forward(self, images):
        return self.seq(images)

    def __str__(self):
        num_params = sum(p.numel() for p in self.parameters())
        return super().__str__() + "\nTotal Parameters: {:,}".format(num_params)

class Generator(nn.Module):
    def __init__(self, noise_dim=100, start_dim=4):
        super().__init__()
        self.start_dim = start_dim
        self.dense = Dense(noise_dim, start_dim * start_dim * 256)
        self.seq = nn.Sequential(
            Conv2D(256, 256),
            Conv2D(256, 128),
            Upscale(),
            Conv2D(128, 128),
            Conv2D(128, 64),
            Upscale(),
            Conv2D(64, 64),
            Conv2D(64, 32),
            Upscale(),
            Conv2D(32, 32),
            Conv2D(32, 16),
            Conv2D(16, 1, bn=False, act=torch.tanh)
        )

    def forward(self, noise):
        features = self.dense(noise)
        features = features.reshape(noise.shape[0], 256, self.start_dim, self.start_dim)
        images = self.seq(features)
        return images

    def __str__(self):
        num_params = sum(p.numel() for p in self.parameters())
        return super().__str__() + "\nTotal Parameters: {:,}".format(num_params)
