import spconv.pytorch as spconv
import torch
import numpy as np
from functools import partial
from spconv.pytorch import SparseConv2d
from torch import nn
from torch.nn import functional as F

from ..registry import BACKBONES
from ..utils import build_norm_layer
from .base import UpsampleLayer, Dense2DBasicBlock, Dense2DBasicBlockV

from det3d.ops.pillar_ops.pillar_modules import PillarMaxPoolingV2a


@BACKBONES.register_module
class SpPillarParallelEncoderMFuse(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            name="SpPillarParallelEncoderMFuse", **kwargs
    ):
        super(SpPillarParallelEncoderMFuse, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 64],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.conv = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, 3, stride=1, bias=False),
            build_norm_layer(dict(type="BN", eps=1e-3, momentum=0.01), 64)[1],
        )

        self.conv1 = nn.Sequential(
            Dense2DBasicBlockV(64, 64, stride=1),
            Dense2DBasicBlock(64, 64, stride=1),
        )
        self.deconv1 = UpsampleLayer(64, 128, stride=pillar_cfg['pool1']['us_stride'])

        # self.pillar_pooling2 = PillarMaxPoolingV2a(
        #     # radius=pillar_cfg['pool2']['radius'],
        #     mlps=[6 + num_input_features, 128],
        #     norm_cfg=norm_cfg,
        #     bev_size=pillar_cfg['pool2']['bev'],
        #     point_cloud_range=pc_range
        # )  # [752, 752]

        self.conv2 = nn.Sequential(
            Dense2DBasicBlockV(64, 128, stride=2),
            Dense2DBasicBlock(128, 128, stride=1),
        )
        self.deconv2 = UpsampleLayer(128, 128, stride=pillar_cfg['pool2']['us_stride'])

        # self.pillar_pooling3 = PillarMaxPoolingV2a(
        #     # radius=pillar_cfg['pool3']['radius'],
        #     mlps=[6 + num_input_features, 256],
        #     norm_cfg=norm_cfg,
        #     bev_size=pillar_cfg['pool3']['bev'],
        #     point_cloud_range=pc_range
        # )  # [752, 752]

        self.conv3 = nn.Sequential(
            Dense2DBasicBlockV(64, 256, stride=4),
            Dense2DBasicBlock(256, 256, stride=1),
        )
        self.deconv3 = UpsampleLayer(256, 128, stride=pillar_cfg['pool3']['us_stride'])

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        x = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x = x.dense()
        x = self.conv(x)

        x_conv1 = self.conv1(x)
        x_conv1 = self.deconv1(x_conv1)

        # x_conv2 = self.down_sample2(x_conv1)
        # x_conv2 = self.pillar_pooling2(xyz, xyz_batch_cnt, pt_features)
        x_conv2 = self.conv2(x)
        x_conv2 = self.deconv2(x_conv2)

        # x_conv3 = self.down_sample3(x_conv2)
        # x_conv3 = self.pillar_pooling3(xyz, xyz_batch_cnt, pt_features)
        x_conv3 = self.conv3(x)
        x_conv3 = self.deconv3(x_conv3)

        ret = torch.cat((x_conv1, x_conv2, x_conv3), dim=1)

        return ret, None


