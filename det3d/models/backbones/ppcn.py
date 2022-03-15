import torch
try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv2d, SparseMaxPool2d, SparseInverseConv2d
except:
    import spconv
    from spconv import ops
    from spconv import SparseConv2d, SparseMaxPool2d, SparseInverseConv2d

from torch import nn
from timm.models.layers import trunc_normal_

from ..registry import BACKBONES
from ..utils import build_norm_layer
from .base import conv2D3x3, Sparse2DBasicBlock, Sparse2DBasicBlockV, Sparse2DMerge, Sparse2DInverseBasicBlock, \
    Dense2DBasicBlock, Dense2DInverseBasicBlock, Sparse2DBottleneck, Sparse2DBottleneckV, Dense2DBottleneck

from det3d.ops.pillar_ops.pillar_modules import PillarMaxPoolingV1a, PillarMaxPoolingV2a, PillarMaxPoolingV2a1


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   dilation=1, conv_type='subm', norm_cfg=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, dilation=dilation,
                                 padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(),
    )

    return m

def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_cfg=None):
    m = spconv.SparseSequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(),
    )

    return m


@BACKBONES.register_module
class ParallelPillarEncoderHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="PillarEncoder", **kwargs
    ):
        super(ParallelPillarEncoderHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 16*double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )

        self.pillar_pooling2 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 32*double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )

        block = post_act_block
        dense_block = post_act_block_dense
        self.conv1_b1 = spconv.SparseSequential(
            spconv.SubMConv2d(16*double, 16*double, 3, padding=1, bias=False, indice_key="subm1_1"),
            build_norm_layer(norm_cfg, 16*double)[1],
            block(16*double, 16*double, 3, norm_cfg=norm_cfg, indice_key="subm1_1"),
        )

        self.conv2_b1 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16*double, 32*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2_1', conv_type='spconv'),
            block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2_1'),
            block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2_1'),
        )

        self.conv3_b1 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32*double, 64*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3_1', conv_type='spconv'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3_1'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3_1'),
        )

        self.conv4_b1 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*double, 128*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv4_1', conv_type='spconv'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4_1'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4_1'),
        )

        self.conv1_b2 = spconv.SparseSequential(
            spconv.SubMConv2d(32 * double, 32 * double, 3, padding=1, bias=False, indice_key="subm1_2"),
            build_norm_layer(norm_cfg, 32 * double)[1],
            block(32 * double, 32 * double, 3, norm_cfg=norm_cfg, indice_key="subm1_2"),
        )

        self.conv2_b2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32 * double, 64 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2_2',
                  conv_type='spconv'),
            block(64 * double, 64 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2_2'),
            block(64 * double, 64 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2_2'),
        )

        self.conv3_b2 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(64 * double, 128 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3_2',
                  conv_type='spconv'),
            block(128 * double, 128 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3_2'),
            block(128 * double, 128 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3_2'),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv4_b2 = nn.Sequential(
            # [400, 352, 11] <- [200, 176, 5]
            dense_block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor1 = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv1_b1 = self.conv1_b1(sp_tensor1)
        x_conv2_b1 = self.conv2_b1(x_conv1_b1)
        x_conv3_b1 = self.conv3_b1(x_conv2_b1)
        x_conv4_b1 = self.conv4_b1(x_conv3_b1)

        sp_tensor2 = self.pillar_pooling2(xyz, xyz_batch_cnt, pt_features)
        x_conv1_b2 = self.conv1_b2(sp_tensor2)
        x_conv2_b2 = self.conv2_b2(x_conv1_b2)
        x_conv3_b2 = self.conv3_b2(x_conv2_b2)
        x_conv3_b2 = x_conv3_b2.dense()
        x_conv4_b2 = self.conv4_b2(x_conv3_b2)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4_b1,
            x_conv5=x_conv4_b2,
        )


@BACKBONES.register_module
class SpMiddleParallelPillarEncoderHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoderHA", **kwargs
    ):
        super(SpMiddleParallelPillarEncoderHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 16 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )

        self.pillar_pooling2 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 32 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )

        self.conv1_b1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1_1"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1_1"),
        )

        self.conv2_b1 = spconv.SparseSequential(
            SparseConv2d(
                16*double, 32*double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2_1"),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2_1"),
        )

        self.conv3_b1 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3_1"),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3_1"),
        )

        self.conv4_b1 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4_1"),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4_1"),
        )

        self.conv1_b2 = spconv.SparseSequential(
            Sparse2DBasicBlockV(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1_2"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1_2"),
        )

        self.conv2_b2 = spconv.SparseSequential(
            SparseConv2d(
                32 * double, 64 * double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 64 * double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2_2"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2_2"),
        )

        self.conv3_b2 = spconv.SparseSequential(
            SparseConv2d(
                64 * double, 128 * double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 128 * double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3_2"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3_2"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv4_b2 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor1 = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv1_b1 = self.conv1_b1(sp_tensor1)
        x_conv2_b1 = self.conv2_b1(x_conv1_b1)
        x_conv3_b1 = self.conv3_b1(x_conv2_b1)
        x_conv4_b1 = self.conv4_b1(x_conv3_b1)

        sp_tensor2 = self.pillar_pooling2(xyz, xyz_batch_cnt, pt_features)
        x_conv1_b2 = self.conv1_b2(sp_tensor2)
        x_conv2_b2 = self.conv2_b2(x_conv1_b2)
        x_conv3_b2 = self.conv3_b2(x_conv2_b2)
        x_conv3_b2 = x_conv3_b2.dense()
        x_conv4_b2 = self.conv4_b2(x_conv3_b2)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4_b1,
            x_conv5=x_conv4_b2,
        )


@BACKBONES.register_module
class SpMiddleParallelPillarEncoder34HA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder34HA", **kwargs
    ):
        super(SpMiddleParallelPillarEncoder34HA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 16 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )

        self.pillar_pooling2 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 32 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )

        self.conv1_1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1_1"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1_1"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1_1"),
        )

        self.conv2_1 = spconv.SparseSequential(
            SparseConv2d(
                16*double, 32*double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2_1"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2_1"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2_1"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2_1"),
        )

        self.conv3_1 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3_1"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3_1"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3_1"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3_1"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3_1"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3_1"),
        )

        self.conv4_1 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4_1"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4_1"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4_1"),
        )

        self.conv1_2 = spconv.SparseSequential(
            Sparse2DBasicBlockV(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1_2"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1_2"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1_2"),
        )

        self.conv2_2 = spconv.SparseSequential(
            SparseConv2d(
                32 * double, 64 * double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 64 * double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2_2"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2_2"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2_2"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2_2"),
        )

        self.conv3_2 = spconv.SparseSequential(
            SparseConv2d(
                64 * double, 128 * double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 128 * double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3_2"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3_2"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3_2"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3_2"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3_2"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3_2"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor1 = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv1_b1 = self.conv1_b1(sp_tensor1)
        x_conv2_b1 = self.conv2_b1(x_conv1_b1)
        x_conv3_b1 = self.conv3_b1(x_conv2_b1)
        x_conv4_b1 = self.conv4_b1(x_conv3_b1)

        sp_tensor2 = self.pillar_pooling2(xyz, xyz_batch_cnt, pt_features)
        x_conv1_b2 = self.conv1_b2(sp_tensor2)
        x_conv2_b2 = self.conv2_b2(x_conv1_b2)
        x_conv3_b2 = self.conv3_b2(x_conv2_b2)
        x_conv3_b2 = x_conv3_b2.dense()
        x_conv4_b2 = self.conv4_b2(x_conv3_b2)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4_b1,
            x_conv5=x_conv4_b2,
        )


@BACKBONES.register_module
class SpMiddleParallelPillarEncoder50HA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder", **kwargs
    ):
        super(SpMiddleParallelPillarEncoder50HA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 16*double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.pillar_pooling2 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 32*double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        expansion = Sparse2DBottleneck.expansion
        self.conv1_1 = spconv.SparseSequential(
            Sparse2DBottleneckV(16*double, 16*double, norm_cfg=norm_cfg, indice_key="res1_1"),
            Sparse2DBottleneck(16*double*expansion, norm_cfg=norm_cfg, indice_key="res1_1"),
            Sparse2DBottleneck(16*double*expansion, norm_cfg=norm_cfg, indice_key="res1_1"),
        )

        self.conv2_1 = spconv.SparseSequential(
            SparseConv2d(
                16*double*expansion, 32*double*expansion, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32*double*expansion)[1],
            nn.ReLU(),
            Sparse2DBottleneck(32*double*expansion, norm_cfg=norm_cfg, indice_key="res2_1"),
            Sparse2DBottleneck(32*double*expansion, norm_cfg=norm_cfg, indice_key="res2_1"),
            Sparse2DBottleneck(32*double*expansion, norm_cfg=norm_cfg, indice_key="res2_1"),
            Sparse2DBottleneck(32*double*expansion, norm_cfg=norm_cfg, indice_key="res2_1"),
        )

        self.conv3_1 = spconv.SparseSequential(
            SparseConv2d(
                32*double*expansion, 64*double*expansion, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double*expansion)[1],
            nn.ReLU(),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res3_1"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res3_1"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res3_1"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res3_1"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res3_1"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res3_1"),
        )

        self.conv4_1 = spconv.SparseSequential(
            SparseConv2d(
                64*double*expansion, 128*double*expansion, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double*expansion)[1],
            nn.ReLU(),
            Sparse2DBottleneck(128*double*expansion, norm_cfg=norm_cfg, indice_key="res4_1"),
            Sparse2DBottleneck(128*double*expansion, norm_cfg=norm_cfg, indice_key="res4_1"),
            Sparse2DBottleneck(128*double*expansion, norm_cfg=norm_cfg, indice_key="res4_1"),
        )

        self.conv1_2 = spconv.SparseSequential(
            Sparse2DBottleneckV(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res1_2"),
            Sparse2DBottleneck(32*double*expansion, norm_cfg=norm_cfg, indice_key="res1_2"),
            Sparse2DBottleneck(32*double*expansion, norm_cfg=norm_cfg, indice_key="res1_2"),
        )

        self.conv2_2 = spconv.SparseSequential(
            SparseConv2d(
                32*double*expansion, 64*double*expansion, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 64*double*expansion)[1],
            nn.ReLU(),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res2_2"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res2_2"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res2_2"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res2_2"),
        )

        self.conv3_2 = spconv.SparseSequential(
            SparseConv2d(
                64*double*expansion, 128*double*expansion, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 128*double*expansion)[1],
            nn.ReLU(),
            Sparse2DBottleneck(128*double*expansion, norm_cfg=norm_cfg, indice_key="res3_2"),
            Sparse2DBottleneck(128*double*expansion, norm_cfg=norm_cfg, indice_key="res3_2"),
            Sparse2DBottleneck(128*double*expansion, norm_cfg=norm_cfg, indice_key="res3_2"),
            Sparse2DBottleneck(128*double*expansion, norm_cfg=norm_cfg, indice_key="res3_2"),
            Sparse2DBottleneck(128*double*expansion, norm_cfg=norm_cfg, indice_key="res3_2"),
            Sparse2DBottleneck(128*double*expansion, norm_cfg=norm_cfg, indice_key="res3_2"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(
                128 * double * expansion, 256, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor1 = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv1_b1 = self.conv1_b1(sp_tensor1)
        x_conv2_b1 = self.conv2_b1(x_conv1_b1)
        x_conv3_b1 = self.conv3_b1(x_conv2_b1)
        x_conv4_b1 = self.conv4_b1(x_conv3_b1)

        sp_tensor2 = self.pillar_pooling2(xyz, xyz_batch_cnt, pt_features)
        x_conv1_b2 = self.conv1_b2(sp_tensor2)
        x_conv2_b2 = self.conv2_b2(x_conv1_b2)
        x_conv3_b2 = self.conv3_b2(x_conv2_b2)
        x_conv3_b2 = x_conv3_b2.dense()
        x_conv4_b2 = self.conv4_b2(x_conv3_b2)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4_b1,
            x_conv5=x_conv4_b2,
        )
