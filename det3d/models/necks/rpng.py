import time
import numpy as np
import math

import spconv.pytorch
import torch

from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.torchie.trainer import load_checkpoint
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.utils import change_default_args

from .. import builder
from ..registry import NECKS
from ..utils import build_norm_layer

@NECKS.register_module
class RPNV2(nn.Module):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
        **kwargs
    ):
        super(RPNV2, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        self.block_5, num_out_filters = self._make_layer(
            self._num_input_features[1],
            self._num_filters[1],
            self._layer_nums[1],
            stride=1,
        )
        self.deblock_5 = Sequential(
            nn.ConvTranspose2d(
                num_out_filters,
                self._num_input_features[0],
                2,
                stride=2,
                bias=False,
            ),
            build_norm_layer(
                self._norm_cfg,
                self._num_input_features[0],
            )[1],
            nn.ReLU(),
        )

        self.block_4, num_out_filters = self._make_layer(
            self._num_input_features[0] * 2,
            self._num_input_features[0] * 2,
            self._layer_nums[0],
            stride=1,
        )

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
                # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
            block.add(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, pillar_features, **kwargs):
        outs = dict()
        if isinstance(pillar_features, dict):
            x_conv3 = pillar_features['x_conv3']
            x_conv4 = pillar_features['x_conv4']
            x_conv5 = pillar_features['x_conv5']
            if isinstance(x_conv3, spconv.pytorch.SparseConvTensor):
                x_conv3 = x_conv3.dense()
            if isinstance(x_conv4, spconv.pytorch.SparseConvTensor):
                x_conv4 = x_conv4.dense()
            if isinstance(x_conv5, spconv.pytorch.SparseConvTensor):
                x_conv5 = x_conv5.dense()

        ups = [x_conv4]
        x = self.block_5(x_conv5)
        ups.append(self.deblock_5(x))
        x = torch.cat(ups, dim=1)
        x = self.block_4(x)
        return x

@NECKS.register_module
class RPNG3(nn.Module):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
        **kwargs
    ):
        super(RPNG3, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        self.deblock_5 = Sequential(
            nn.ConvTranspose2d(
                self._num_input_features[1],
                self._num_input_features[0],
                2,
                stride=2,
                bias=False,
            ),
            build_norm_layer(
                self._norm_cfg,
                self._num_input_features[0],
            )[1],
            nn.ReLU(),
        )

        self.block_4, num_out_filters = self._make_layer(
            self._num_input_features[0] * 2,
            self._num_input_features[0] * 2,
            self._layer_nums[0],
            stride=1,
        )

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
                # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
            block.add(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, pillar_features, **kwargs):
        if isinstance(pillar_features, dict):
            x_conv4 = pillar_features['x_conv4']
            x_conv5 = pillar_features['x_conv5']
            if isinstance(x_conv4, spconv.pytorch.SparseConvTensor):
                x_conv4 = x_conv4.dense()
            if isinstance(x_conv5, spconv.pytorch.SparseConvTensor):
                x_conv5 = x_conv5.dense()

        ups = [x_conv4]
        ups.append(self.deblock_5(x_conv5))
        x = torch.cat(ups, dim=1)
        x = self.block_4(x)
        return x

@NECKS.register_module
class RPNG1(nn.Module):
    def __init__(
        self,
        layer_nums,
        num_input_features,
        share_conv_channel=64,
        norm_cfg=None,
        name="rpn",
        logger=None,
        **kwargs
    ):
        super(RPNG1, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        self.deblock_5 = Sequential(
            nn.ConvTranspose2d(
                num_input_features[0],
                num_input_features[1],
                2,
                stride=2,
                bias=False,
            ),
            build_norm_layer(
                self._norm_cfg,
                num_input_features[1],
            )[1],
            nn.ReLU(),
        )

        self.block_4, num_out_filters = self._make_layer(
            num_input_features[1] * 2,
            num_input_features[1] * 2,
            layer_nums[0],
            stride=1,
        )

        self.shared_conv_4 = nn.Sequential(
            nn.Conv2d(num_out_filters, share_conv_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU()
        )

        self.deblock_4 = Sequential(
            nn.ConvTranspose2d(
                num_input_features[1] * 2,
                num_input_features[2],
                2,
                stride=2,
                bias=False,
            ),
            build_norm_layer(
                self._norm_cfg,
                num_input_features[2],
            )[1],
            nn.ReLU(),
        )

        self.block_3, num_out_filters = self._make_layer(
            num_input_features[2] * 2,
            num_input_features[2] * 2,
            layer_nums[0],
            stride=1,
        )

        self.shared_conv_3 = nn.Sequential(
            nn.Conv2d(num_out_filters, share_conv_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU()
        )

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
                # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
            block.add(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, pillar_features, **kwargs):
        outs = dict()
        if isinstance(pillar_features, dict):
            x_conv3 = pillar_features['x_conv3']
            x_conv4 = pillar_features['x_conv4']
            x_conv5 = pillar_features['x_conv5']
            if isinstance(x_conv3, spconv.pytorch.SparseConvTensor):
                x_conv3 = x_conv3.dense()
            if isinstance(x_conv4, spconv.pytorch.SparseConvTensor):
                x_conv4 = x_conv4.dense()
            if isinstance(x_conv5, spconv.pytorch.SparseConvTensor):
                x_conv5 = x_conv5.dense()

        ups = [x_conv4]
        ups.append(self.deblock_5(x_conv5))
        ups = torch.cat(ups, dim=1)
        outs['8'] = self.shared_conv_4(self.block_4(ups))
        ups = self.deblock_4(ups)
        outs['4'] = self.shared_conv_3(self.block_3(torch.cat([ups, x_conv3], dim=1)))
        return outs

@NECKS.register_module
class RPNB(nn.Module):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="RPN",
        logger=None,
        **kwargs
    ):
        super(RPNB, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        blocks = []
        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                self._num_input_features[i],
                self._num_filters[i],
                layer_num,
                stride=1,
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
                # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
            block.add(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, pillar_features):
        ups = []
        x1 = self.blocks[0](pillar_features['x_conv4'].dense())
        ups.append(x1)
        x2 = self.blocks[1](pillar_features['x_conv5'].dense())
        ups.append(F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False))

        x = torch.cat(ups, dim=1)
        return x

@NECKS.register_module
class LRPN(nn.Module):
    def __init__(
        self,
        layer_num,
        dilations,
        num_out_features,
        num_input_features,
        norm_cfg=None,
        logger=None,
        **kwargs
    ):
        super(LRPN, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        assert len(layer_num) == len(num_out_features) == 2

        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(num_input_features, num_out_features[0], 3, stride=1, bias=False),
            build_norm_layer(norm_cfg, num_out_features[0])[1],
            nn.ReLU(),
        )

        self.block1_1 = Sequential(
            nn.Conv2d(num_out_features[0], num_out_features[0], 3,
                      dilation=dilations[0], padding=dilations[0], bias=False),
            build_norm_layer(norm_cfg, num_out_features[0])[1],
            nn.ReLU(),
        )

        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(num_out_features[0], num_out_features[1], 3, stride=1, bias=False),
            build_norm_layer(norm_cfg, num_out_features[1])[1],
            nn.ReLU(),
        )

        self.block2_1 = Sequential(
            nn.Conv2d(num_out_features[1], num_out_features[1], 3,
                      dilation=dilations[1], padding=dilations[1], bias=False),
            build_norm_layer(norm_cfg, num_out_features[1])[1],
            nn.ReLU(),
        )

        logger.info("Finish LRPN Initialization")

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        x = self.block1(x)
        x = self.block1_1(x)
        x = self.block2(x)
        x = self.block2_1(x)
        return x
