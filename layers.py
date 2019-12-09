import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_activation(name):
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    return None


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
    def __init__(self, in_units, out_units, bn=True, act="leaky_relu", dropout=False):
        super().__init__()
        layers = [nn.Linear(in_units, out_units)]
        if bn:
            layers.append(nn.BatchNorm1d(out_units))
        if dropout:
            layers.append(nn.Dropout(p=0.5))
        if act != "linear":
            layers.append(parse_activation(act))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


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
