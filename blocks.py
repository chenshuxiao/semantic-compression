import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from typing import *

class VanillaBlockEnc(nn.Module):
    """ Resnet Block without skip connections and batch normalization
    inspired by: https://github.com/julianstastny/VAE-ResNet18-PyTorch

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self, in_planes: int, stride: int=1):
        super().__init__()

        self.planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, self.planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x: Tensor):
        out = nn.ReLU(self.conv1(x))
        out = nn.ReLU(self.conv2(out))
        return out

class ResBlockEnc(nn.Module):

    def __init__(self, in_planes: int, stride: int=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: Tensor):
        out = nn.ReLU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU(out)
        return out

class ResBlockDec(nn.Module):

    def __init__(self, in_planes: int, stride: int=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: Tensor):
        out = nn.LeakyReLU(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = nn.LeakyReLU(out)
        return out