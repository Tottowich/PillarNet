import torch.nn as nn
import numpy as np
import torch

from .base import spconv


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, bias=False, norm_cfg=None):
    result = spconv.SparseSequential()
    result.add_module(
        'conv',
        spconv.SubMConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            # dilation=dilation,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        ))
    result.add_module('bn', build_norm_layer(norm_cfg, out_channels)[1])
    return result


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None,
                 stride=1, activation=nn.ReLU(), indice_key=None
                 ):

        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.padding_mode = padding_mode
        self.se_block = se_block

        self.nonlinearity = activation

        self.fused = False
        bias = norm_cfg is not None

        self.rbr_identity = build_norm_layer(norm_cfg, in_channels)[1]
        self.rbr_dense = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            norm_cfg=norm_cfg)
        self.rbr_1x1 = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=bias,
            norm_cfg=norm_cfg)

    def _forward(self, inputs):
        rbr_1x1_output = self.rbr_1x1(inputs)

        if self.rbr_dense is None:
            dense_output = 0
        else:
            dense_output = self.rbr_dense(inputs)

        return rbr_1x1_output, dense_output

    def forward(self, inputs):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        rbr_1x1_output, drop_path_output = self._forward(inputs)

        out = drop_path_output + rbr_1x1_output + id_out
        out = self.nonlinearity(out)

        return out

    def fuse_conv_bn(self, conv, bn):
        """
        # n,c,h,w - conv
        # n - bn (scale, bias, mean, var)

        if type(bn) is nn.Identity or type(bn) is None:
            return

        conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        """
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t


        conv = spconv.SubMConv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=True,
        )

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)

        return conv

    def fuse_repvgg_block(self):
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense.conv, self.rbr_dense.bn)

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1.conv, self.rbr_1x1.bn)
        rbr_1x1_bias = self.rbr_1x1.bias

        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias)

        self.rbr_1x1 = nn.Identity()

        self.fused = True
