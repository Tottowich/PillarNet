import time
import numpy as np
import math

import torch

from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.torchie.trainer import load_checkpoint
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.utils import change_default_args
from det3d.ops.pillar_ops.pillar_modules import PillarMaxPoolingV2a

from .. import builder
from ..registry import NECKS
from ..utils import build_norm_layer


@NECKS.register_module
class ParallelRPN(nn.Module):
    def __init__(
        self,
        pillar_cfg,
        layer_nums,
        ds_num_filters,
        ds_layer_strides,
        us_layer_strides,
        us_num_filters,
        num_point_features,
        name="ParallelRPN",
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        logger=None,
        **kwargs
    ):
        super(ParallelRPN, self).__init__()
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters

        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.pillar_pooling1 = PillarMaxPoolingV2a(
            mlps=[6 + num_point_features, 64],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )

        # self.pillar_pooling2 = PillarMaxPoolingV2a(
        #     mlps=[6 + num_point_features, ds_num_filters[0]],
        #     norm_cfg=norm_cfg,
        #     bev_size=pillar_cfg['pool2']['bev'],
        #     point_cloud_range=pc_range
        # )
        #
        # self.pillar_pooling3 = PillarMaxPoolingV2a(
        #     mlps=[6 + num_point_features, ds_num_filters[1]],
        #     norm_cfg=norm_cfg,
        #     bev_size=pillar_cfg['pool3']['bev'],
        #     point_cloud_range=pc_range
        # )  # [752, 752]
        #
        # self.pillar_pooling4 = PillarMaxPoolingV2a(
        #     mlps=[6 + num_point_features, ds_num_filters[2]],
        #     norm_cfg=norm_cfg,
        #     bev_size=pillar_cfg['pool4']['bev'],
        #     point_cloud_range=pc_range
        # )  # [752, 752]

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        # self.conv = Sequential(
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(64, 64, 3, stride=1, bias=False),
        #     build_norm_layer(norm_cfg, 64)[1],
        # )
        ds_layer_strides = [2, 4, 8]
        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        blocks = []
        dsblocks = []
        deblocks = []
        for i, layer_num in enumerate(self._layer_nums):
            dsblock, block, num_out_filters = self._make_layer(
                ds_num_filters[0],
                self._num_filters[i],
                layer_num,
                ds_layer_strides[i],
            )
            dsblocks.append(dsblock)
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                if stride > 1:
                    deblock = Sequential(
                        nn.ConvTranspose2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = Sequential(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.dsblocks = nn.ModuleList(dsblocks)
        self.deblocks = nn.ModuleList(deblocks)

        logger.info("Finish ParallelRPN Initialization")

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        dsblock = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU(),
        )

        block = Sequential()
        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(norm_cfg, planes)[1],
                # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
            block.add(nn.ReLU())

        return dsblock, block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, xyz, xyz_batch_cnt, pt_features):

        x = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x = x.dense()

        # pool2 = self.pillar_pooling2(xyz, xyz_batch_cnt, pt_features)
        x_conv1 = self.dsblocks[0](x)
        x_conv1 = self.blocks[0](x_conv1)
        x_conv1 = self.deblocks[0](x_conv1)

        # pool3 = self.pillar_pooling3(xyz, xyz_batch_cnt, pt_features)
        x_conv2 = self.dsblocks[1](x)
        x_conv2 = self.blocks[1](x_conv2)
        x_conv2 = self.deblocks[1](x_conv2)

        # pool4 = self.pillar_pooling4(xyz, xyz_batch_cnt, pt_features)
        x_conv3 = self.dsblocks[2](x)
        x_conv3 = self.blocks[2](x_conv3)
        x_conv3 = self.deblocks[2](x_conv3)

        x = torch.cat([x_conv1, x_conv2, x_conv3], dim=1)

        return x
