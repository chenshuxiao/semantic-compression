import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from typing import *

# TODO: make blocks of regular, resnet, iso, R-iso
# TODO: we will make regular and resnet first, then iso and R-iso as flags?

# Won't really use these two
def get_regular_enc(in_channels, h_dim, stride):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels=h_dim,
                    kernel_size= 3, stride= stride, padding= 1),
            nn.BatchNorm2d(h_dim),
            nn.ReLU()
        )
    
def get_regular_dec(h_dim_i, h_dim_i_1, stride):
    return nn.Sequential(
            nn.ConvTranspose2d(h_dim_i,
                                h_dim_i_1,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                output_padding=1),
            nn.BatchNorm2d(hidden_dims[i + 1]),
            nn.LeakyReLU()
        )

class SReLU(nn.Module):
    """Shifted ReLU"""

    def __init__(self, nc):
        super().__init__()
        self.srelu_bias = nn.Parameter(torch.Tensor(1, nc, 1, 1))
        self.srelu_relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.srelu_bias, -1.0)

    def forward(self, x):
        return self.srelu_relu(x - self.srelu_bias) + self.srelu_bias

class BasicTransform(nn.Module):
    """Basic transformation: 3x3, 3x3"""

    def __init__(self, w_in, w_out, stride, w_b=None, num_gs=1, **kwargs):
        assert w_b is None and num_gs == 1, \
            'Basic transform does not support w_b and num_gs options'
        super().__init__()
        self._construct(w_in, w_out, stride, **kwargs)

    def _construct(self, w_in, w_out, stride, **kwargs):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_out, kernel_size=3,
            stride=stride, padding=1, bias=not kwargs['HAS_BN'] and not kwargs['SReLU']
        )
        if kwargs['HAS_BN']:
            self.a_bn = nn.BatchNorm2d(w_out)
        self.a_relu = nn.ReLU(inplace=True) if not kwargs['SReLU'] else SReLU(w_out)
        # 3x3, BN
        self.b = nn.Conv2d(
            w_out, w_out, kernel_size=3,
            stride=1, padding=1, bias=not kwargs['HAS_BN'] and not kwargs['SReLU']
        )
        if kwargs['HAS_BN']:
            self.b_bn = nn.BatchNorm2d(w_out)
            self.b_bn.final_bn = True

        # if C.ISON.HAS_RES_MULTIPLIER:
        #     self.shared_scalar = SharedScale()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResBlock(nn.Module):
    """Residual block: x + F(x)"""

    def __init__(
        self, w_in, w_out, stride, trans_fun, w_b=None, num_gs=1, skip_relu=False, **kwargs
    ):
        """

        Parameters
        ----------
        w_b :
        num_gs : 
            ^ all of these above don't matter rn
        skip_relu:
            currently only used in decode, just so we can get the pixel shuffle, then
            afterwards the relu
        """
        super(ResBlock, self).__init__()
        self.kwargs = kwargs
        self.skip_relu = skip_relu
        self._construct(w_in, w_out, stride, trans_fun, w_b, num_gs, **kwargs)

    def _add_skip_proj(self, w_in, w_out, stride, **kwargs):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1,
            stride=stride, padding=0, bias=not kwargs['HAS_BN'] and not kwargs['SReLU']
        )
        if kwargs['HAS_BN']:
            self.bn = nn.BatchNorm2d(w_out)

    def _construct(self, w_in, w_out, stride, trans_fun, w_b, num_gs, **kwargs):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block and kwargs['HAS_ST']:
            self._add_skip_proj(w_in, w_out, stride, **kwargs)
        self.f = trans_fun(w_in, w_out, stride, w_b, num_gs, **kwargs)
        if not self.skip_relu:
            self.relu = nn.ReLU(True) if not kwargs['SReLU'] else SReLU(w_out)

    def forward(self, x):
        if self.proj_block:
            if self.kwargs['HAS_BN'] and self.kwargs['HAS_ST']:
                x = self.bn(self.proj(x)) + self.f(x)
            elif not self.kwargs['HAS_BN'] and self.kwargs['HAS_ST']:
                x = self.proj(x) + self.f(x)
            else:
                x = self.f(x)
        else:
            if self.kwargs['HAS_ST']:
                x = x + self.f(x)
            else:
                x = self.f(x)
        if not self.skip_relu:
            x = self.relu(x)
        return x

class ResStem(nn.Module):
    """Stem of resnet (start of resnet)."""
    def __init__(self, w_in: int, w_out: int, stride: int=1, **kwargs):
        """[summary]

        Parameters
        ----------
        w_in : int
            numbers of dim in
        w_out : int
            numbers of dim out
        stride : int, optional, by default 1
        cifar : bool, optional
            cifar images are 32x32, if working with larger images, set to False, by default True
            if True: kernel_size=7, padding=3; else k=3, padding=1
        """
        super().__init__()
        if kwargs['CIFAR']:
            self._construct_cifar(w_in, w_out, **kwargs)
        else:
            self._construct_imagenet(w_in, w_out, **kwargs)

    # Biases are in the BN layers that follow.
    def _construct_cifar(self, w_in, w_out, **kwargs):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=3,
            stride=1, padding=1, bias=not kwargs['HAS_BN'] and not kwargs['SReLU']
        )
        self.relu = nn.ReLU(True)
        if kwargs['HAS_BN']:
            self.bn = nn.BatchNorm2d(w_out)
        self.relu = nn.ReLU(True) if not kwargs['SReLU'] else SReLU(w_out)

    def _construct_imagenet(self, w_in, w_out, **kwargs):
        # 7x7, BN, ReLU, maxpool
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=7,
            stride=2, padding=3, bias=not kwargs['HAS_BN'] and not kwargs['SReLU']
        )
        if kwargs['HAS_BN']:
            self.bn = nn.BatchNorm2d(w_out)
        self.relu = nn.ReLU(True) if not kwargs['SReLU'] else SReLU(w_out)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1, **kwargs):
        super().__init__()
        self._construct(w_in, w_out, stride, d, w_b, num_gs, **kwargs)

    def _construct(self, w_in, w_out, stride, d, w_b, num_gs, **kwargs):
        # Construct the blocks
        for i in range(d):
            # Stride and w_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Retrieve the transformation function
            trans_fun = BasicTransform
            # Construct the block
            res_block = ResBlock(
                b_w_in, w_out, b_stride, trans_fun, w_b, num_gs, **kwargs
            )
            self.add_module('b{}'.format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

