import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgPooling(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.shape[2])[:, :, 0, 0]


class Upscale(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2)


class Downscale(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class Dense(nn.Module):
    def __init__(self, in_units, out_units, bn=True, act=F.relu):
        super().__init__()
        self.lin = nn.Linear(in_units, out_units)
        self.has_bn = bn
        if bn:
            self.bn = nn.BatchNorm1d(out_units)
        self.act = act

    def forward(self, x):
        x = self.lin(x)
        x = self.act(x)
        if self.has_bn:
            x = self.bn(x)
        return x


class Conv2D(nn.Module):
    def __init__(self, in_units, out_units, bn=True, act=F.leaky_relu):
        super().__init__()
        self.conv = nn.Conv2d(in_units, out_units, kernel_size=3, padding=1)
        self.has_bn = bn
        if bn:
            self.bn = nn.BatchNorm2d(out_units)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        if self.has_bn:
            x = self.bn(x)
        return x
