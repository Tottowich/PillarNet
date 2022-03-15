import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.ops.pillar_ops.pillar_modules import SparseInterpolate3D, PillarMaxPooling

from ..registry import NECKS
from ..utils import build_norm_layer


@NECKS.register_module
class RPNP(nn.Module):
    def __init__(
        self,
        interp_cfg,
        pool_cfg,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_point_features,
        num_input_features,
        pc_range,
        name="rpnp",
        logger=None,
        **kwargs
    ):
        super(RPNP, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        iterpblocks = {}
        for v_src, p_cfg in interp_cfg.items():
            num_point_features += p_cfg['planes']
            iterpblocks[v_src] = SparseInterpolate3D(p_cfg['kernel'], p_cfg['voxel_size'])

        self.point_fusion_layer = Sequential(
            nn.Linear(num_point_features, num_input_features // 2, bias=False),
            build_norm_layer(dict(type="BN1d", eps=1e-3, momentum=0.01), num_input_features // 2)[1],
            nn.ReLU()
        )

        self.pillar_pooling = PillarMaxPooling(
            radius=pool_cfg['radius'],
            mlps=[6 + num_input_features // 2, num_input_features],
            norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
            bev_size=pool_cfg['bev'],
            bev_flag=True,
            point_cloud_range=pc_range
        )
        self.iterpblocks = iterpblocks

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                norm_cfg,
                stride=self._layer_strides[i],
            )
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
        self.deblocks = nn.ModuleList(deblocks)

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, norm_cfg, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(build_norm_layer(norm_cfg, planes)[1])
            block.add(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, example, voxel_feature):

        point_features = [example['pt_features']]
        for conv, layer in self.iterpblocks.items():
            point_features.append(self.iterpblocks[conv](voxel_feature[conv], example['xyz'], example['xyz_batch_cnt']))

        point_features = torch.cat(point_features, dim=1)
        point_features = self.point_fusion_layer(point_features)

        x = self.pillar_pooling(example['xyz'], example['xyz_batch_cnt'], point_features)

        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)

        return x

@NECKS.register_module
class RPNP2(nn.Module):
    def __init__(
        self,
        interp_cfg,
        pool_cfg,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_point_features,
        num_input_features,
        pc_range,
        name="rpnp",
        logger=None,
        **kwargs
    ):
        super(RPNP2, self).__init__()
        self._layer_strides = [1, 1]
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        iterpblocks = {}
        for v_src, p_cfg in interp_cfg.items():
            num_point_features += p_cfg['planes']
            iterpblocks[v_src] = SparseInterpolate3D(p_cfg['kernel'], p_cfg['voxel_size'])

        self.pillar_pooling = PillarMaxPooling(
            radius=pool_cfg['radius'],
            mlps=[6 + num_point_features, num_input_features],
            norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
            bev_size=pool_cfg['bev'],
            bev_flag=True,
            point_cloud_range=pc_range
        )
        self.pillar_pooling2x = PillarMaxPooling(
            radius=pool_cfg['radius'] * 2,
            mlps=[6 + num_point_features, num_input_features],
            norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
            bev_size=pool_cfg['bev'] * 2,
            bev_flag=True,
            point_cloud_range=pc_range
        )
        self.iterpblocks = iterpblocks

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        in_filters = [self._num_input_features, self._num_input_features]
        blocks = []
        deblocks = []

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                norm_cfg,
                stride=self._layer_strides[i],
            )
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
        self.deblocks = nn.ModuleList(deblocks)

        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, norm_cfg, stride=1):

        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(build_norm_layer(norm_cfg, planes)[1])
            block.add(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, example, voxel_feature):

        point_features_list = [example['point_features']]
        for conv, layer in self.iterpblocks.items():
            point_features_list.append(self.iterpblocks[conv](voxel_feature[conv], example['xyz'], example['xyz_batch_cnt']))

        point_features = torch.cat(point_features_list, dim=1)
        x = self.pillar_pooling(example['xyz'], example['xyz_batch_cnt'], point_features)
        x2 = self.pillar_pooling2x(example['xyz'], example['xyz_batch_cnt'], point_features)

        ups = []
        ups.append(self.deblocks[0](self.blocks[0](x)))
        ups.append(self.deblocks[1](self.blocks[1](x2)))
        # for i in range(len(self.blocks)):
        #     x = self.blocks[i](x)
        #     if i - self._upsample_start_idx >= 0:
        #         ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)

        return x

