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
class PillarEncoderHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="PillarEncoder", **kwargs
    ):
        super(PillarEncoderHA, self).__init__()
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

        block = post_act_block
        dense_block = post_act_block_dense
        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv2d(16*double, 16*double, 3, padding=1, bias=False, indice_key="subm1"),
            build_norm_layer(norm_cfg, 16*double)[1],
            block(16*double, 16*double, 3, norm_cfg=norm_cfg, indice_key="subm1"),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16*double, 32*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
            block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32*double, 64*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*double, 128*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            dense_block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)

        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class PillarEncoderLHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="PillarEncoder", **kwargs
    ):
        super(PillarEncoderLHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling0 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 8*double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool0']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        block = post_act_block
        dense_block = post_act_block_dense
        self.conv0 = spconv.SparseSequential(
            spconv.SubMConv2d(8*double, 8*double, 3, padding=1, bias=False, indice_key="subm1"),
            build_norm_layer(norm_cfg, 8*double)[1],
            block(8*double, 8*double, 3, norm_cfg=norm_cfg, indice_key="subm1"),
        )

        self.conv1 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(8 * double, 16 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv1',
                  conv_type='spconv'),
            block(16 * double, 16 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm1'),
            block(16 * double, 16 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16*double, 32*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
            block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32*double, 64*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*double, 128*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            dense_block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling0(xyz, xyz_batch_cnt, pt_features)
        x_conv0 = self.conv0(sp_tensor)
        x_conv1 = self.conv1(x_conv0)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class PillarEncoder2xHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="PillarEncoder2xHA", **kwargs
    ):
        super(PillarEncoder2xHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 32*double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        block = post_act_block
        dense_block = post_act_block_dense
        # self.conv1 = spconv.SparseSequential(
        #     spconv.SubMConv2d(16*double, 16*double, 3, padding=1, bias=False, indice_key="subm1"),
        #     build_norm_layer(norm_cfg, 16*double)[1],
        #     block(16*double, 16*double, 3, norm_cfg=norm_cfg, indice_key="subm1"),
        # )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            # block(16*double, 32*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
            block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32*double, 64*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*double, 128*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            dense_block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)

        # x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(sp_tensor)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )

@BACKBONES.register_module
class PillarEncoder4xHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="PillarEncoder", **kwargs
    ):
        super(PillarEncoder4xHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 64*double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool3']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        block = post_act_block
        dense_block = post_act_block_dense
        # self.conv1 = spconv.SparseSequential(
        #     spconv.SubMConv2d(16*double, 16*double, 3, padding=1, bias=False, indice_key="subm1"),
        #     build_norm_layer(norm_cfg, 16*double)[1],
        #     block(16*double, 16*double, 3, norm_cfg=norm_cfg, indice_key="subm1"),
        # )

        # self.conv2 = spconv.SparseSequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(16*double, 32*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        #     block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        # )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            # block(32*double, 64*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
            block(64*double, 64*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*double, 128*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
            block(128*double, 128*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            dense_block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)

        # x_conv1 = self.conv1(sp_tensor)
        # x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(sp_tensor)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class PillarEncoder8xHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="PillarEncoder", **kwargs
    ):
        super(PillarEncoder8xHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 128 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool4']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        block = post_act_block
        dense_block = post_act_block_dense
        # self.conv1 = spconv.SparseSequential(
        #     spconv.SubMConv2d(16*double, 16*double, 3, padding=1, bias=False, indice_key="subm1"),
        #     build_norm_layer(norm_cfg, 16*double)[1],
        #     block(16*double, 16*double, 3, norm_cfg=norm_cfg, indice_key="subm1"),
        # )

        # self.conv2 = spconv.SparseSequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(16*double, 32*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        #     block(32*double, 32*double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        # )

        # self.conv3 = spconv.SparseSequential(
        #     # [800, 704, 21] <- [400, 352, 11]
        #     block(32*double, 64*double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     block(64 * double, 64 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        #     block(64 * double, 64 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        # )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            # block(64 * double, 128 * double, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv4',
            #       conv_type='spconv'),
            block(128 * double, 128 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
            block(128 * double, 128 * double, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            dense_block(128 * double, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
            dense_block(256, 256, 3, norm_cfg=norm_cfg, padding=1),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)

        # x_conv1 = self.conv1(sp_tensor)
        # x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(sp_tensor)
        x_conv4 = self.conv4(sp_tensor)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoder(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder", **kwargs
    ):
        super(SpMiddlePillarEncoder, self).__init__()
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

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(16*double, 16*double, norm_cfg=norm_cfg, indice_key="res0"),
            Sparse2DBasicBlock(16*double, 16*double, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                16*double, 32*double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res3"),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)

        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        return dict(
            x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
        )


@BACKBONES.register_module
class SpMiddlePillarEncoderNeck(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=1,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoderNeck", **kwargs
    ):
        super(SpMiddlePillarEncoderNeck, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 16 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                16 * double, 32 * double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32 * double)[1],
            nn.ReLU(inplace=True),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32 * double, 64 * double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64 * double)[1],
            nn.ReLU(inplace=True),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64 * double, 128 * double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128 * double)[1],
            nn.ReLU(inplace=True),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4_up = spconv.SparseSequential(
            conv2D3x3(128 * double, 128 * double, bias=False, indice_key="res3"),
            build_norm_layer(norm_cfg, 128 * double)[1],
            nn.ReLU(inplace=True),
        )

        self.conv5 = spconv.SparseSequential(
            SparseConv2d(
                128 * double, 256 * double, 3, 2, padding=1, bias=False, indice_key="res4_cp"
            ),
            build_norm_layer(norm_cfg, 256 * double)[1],
            nn.ReLU(inplace=True),
            Sparse2DBasicBlock(256 * double, 256 * double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(256 * double, 256 * double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        self.inverse_conv5 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(256*double, 128*double, 3, bias=False, indice_key="res4_cp"),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)

        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        conv4_neck = self.conv4_up(x_conv4)

        x_conv5 = self.conv5(x_conv4)
        conv5_neck = self.inverse_conv5(x_conv5)

        x_conv4.features = torch.cat((conv4_neck.features, conv5_neck.features), dim=1)

        ret = x_conv4.dense()
        return ret

@BACKBONES.register_module
class SpMiddlePillarEncoder34(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder", **kwargs
    ):
        super(SpMiddlePillarEncoder34, self).__init__()
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

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(16*double, 16*double, norm_cfg=norm_cfg, indice_key="res0"),
            Sparse2DBasicBlock(16*double, 16*double, norm_cfg=norm_cfg, indice_key="res0"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                16*double, 32*double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3"),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)

        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        ret = x_conv4.dense()

        return ret, None

@BACKBONES.register_module
class SpMiddlePillarEncoderL1(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoderL", **kwargs
    ):
        super(SpMiddlePillarEncoderL1, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling0 = PillarMaxPoolingV2a(
            mlps=[6 + num_input_features, 8*double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool0']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.conv0 = spconv.SparseSequential(
            Sparse2DBasicBlockV(8 * double, 8 * double, norm_cfg=norm_cfg, indice_key="res0"),
            Sparse2DBasicBlock(8 * double, 8 * double, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv1 = spconv.SparseSequential(
            SparseConv2d(
                8 * double, 16 * double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 16 * double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                16*double, 32*double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling0(xyz, xyz_batch_cnt, pt_features)
        x_conv0 = self.conv0(sp_tensor)
        x_conv1 = self.conv1(x_conv0)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        return dict(
            x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
        )

@BACKBONES.register_module
class SpMiddlePillarEncoderL2(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoderL", **kwargs
    ):
        super(SpMiddlePillarEncoderL2, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling0 = PillarMaxPoolingV2a(
            mlps=[6 + num_input_features, 8*double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool0']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.conv0 = spconv.SparseSequential(
            SparseConv2d(
                8 * double, 16 * double, 2, 2, padding=0, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 16 * double)[1],
        )

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                16*double, 32*double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]agent
            build_norm_layer(norm_cfg, 32*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling0(xyz, xyz_batch_cnt, pt_features)
        x_conv0 = self.conv0(sp_tensor)
        x_conv1 = self.conv1(x_conv0)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        return dict(
            x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
        )


@BACKBONES.register_module
class SpMiddlePillarEncoderH(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoderH", **kwargs
    ):
        super(SpMiddlePillarEncoderH, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 16 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                16*double, 32*double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        self.conv5 = spconv.SparseSequential(
            SparseConv2d(
                128 * double, 256, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res5"),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res5"),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        return dict(
            x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoderHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoderHA", **kwargs
    ):
        super(SpMiddlePillarEncoderHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 16 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                16*double, 32*double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoderLHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoderHA", **kwargs
    ):
        super(SpMiddlePillarEncoderLHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling0 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 8 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool0']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.conv0 = spconv.SparseSequential(
            Sparse2DBasicBlockV(8 * double, 8 * double, norm_cfg=norm_cfg, indice_key="res0"),
            Sparse2DBasicBlock(8 * double, 8 * double, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv1 = spconv.SparseSequential(
            SparseConv2d(
                8 * double, 16 * double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 16 * double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                16*double, 32*double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling0(xyz, xyz_batch_cnt, pt_features)
        x_conv0 = self.conv0(sp_tensor)
        x_conv1 = self.conv1(x_conv0)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoder34HA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder34HA", **kwargs
    ):
        super(SpMiddlePillarEncoder34HA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 16 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                16*double, 32*double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoder34x2HA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder34x2HA", **kwargs
    ):
        super(SpMiddlePillarEncoder34x2HA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 32 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        # self.conv1 = spconv.SparseSequential(
        #     Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        #     Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        #     Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        # )

        self.conv2 = spconv.SparseSequential(
            # SparseConv2d(
            #     16*double, 32*double, 3, 2, padding=1, bias=False
            # ),  # [752, 752] -> [376, 376]
            # build_norm_layer(norm_cfg, 32*double)[1],
            # nn.ReLU(),
            Sparse2DBasicBlockV(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        # x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(sp_tensor)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoder34x4HA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder34x4HA", **kwargs
    ):
        super(SpMiddlePillarEncoder34x4HA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 64 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool3']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        # self.conv1 = spconv.SparseSequential(
        #     Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        #     Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        #     Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        # )
        #
        # self.conv2 = spconv.SparseSequential(
        #     SparseConv2d(
        #         16*double, 32*double, 3, 2, padding=1, bias=False
        #     ),  # [752, 752] -> [376, 376]
        #     build_norm_layer(norm_cfg, 32*double)[1],
        #     nn.ReLU(),
        #     Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
        #     Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
        #     Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
        #     Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
        # )

        self.conv3 = spconv.SparseSequential(
            # SparseConv2d(
            #     32*double, 64*double, 3, 2, padding=1, bias=False
            # ),  # [376, 376] -> [188, 188]
            # build_norm_layer(norm_cfg, 64*double)[1],
            # nn.ReLU(),
            Sparse2DBasicBlockV(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        # x_conv1 = self.conv1(sp_tensor)
        # x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(sp_tensor)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )

@BACKBONES.register_module
class SpMiddlePillarEncoder34x8HA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2, num_layers=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder34x8HA", **kwargs
    ):
        super(SpMiddlePillarEncoder34x8HA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 128 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool4']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        # self.conv1 = spconv.SparseSequential(
        #     Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        #     Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        #     Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        # )
        #
        # self.conv2 = spconv.SparseSequential(
        #     SparseConv2d(
        #         16*double, 32*double, 3, 2, padding=1, bias=False
        #     ),  # [752, 752] -> [376, 376]
        #     build_norm_layer(norm_cfg, 32*double)[1],
        #     nn.ReLU(),
        #     Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
        #     Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
        #     Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
        #     Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res2"),
        # )

        # self.conv3 = spconv.SparseSequential(
        #     # SparseConv2d(
        #     #     32*double, 64*double, 3, 2, padding=1, bias=False
        #     # ),  # [376, 376] -> [188, 188]
        #     # build_norm_layer(norm_cfg, 64*double)[1],
        #     # nn.ReLU(),
        #     Sparse2DBasicBlockV(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
        #     Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
        #     Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
        #     Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
        #     Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
        #     Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res3"),
        # )

        layer_blocks = []
        for _ in range(num_layers):
            layer_blocks.append(Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"))

        self.conv4 = spconv.SparseSequential(
            # SparseConv2d(
            #     64*double, 128*double, 3, 2, padding=1, bias=False
            # ),
            # build_norm_layer(norm_cfg, 128*double)[1],
            # nn.ReLU(),
            Sparse2DBasicBlockV(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"),
            # Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"),
            # Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        self.conv4_layers = spconv.SparseSequential(*layer_blocks)

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        # x_conv1 = self.conv1(sp_tensor)
        # x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(sp_tensor)
        x_conv4 = self.conv4(sp_tensor)
        x_conv4 = self.conv4_layers(x_conv4)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoder2xHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder2xHA", **kwargs
    ):
        super(SpMiddlePillarEncoder2xHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 32 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        # self.conv1 = spconv.SparseSequential(
        #     Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        #     Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        # )

        self.conv2 = spconv.SparseSequential(
            # SparseConv2d(
            #     16*double, 32*double, 3, 2, padding=1, bias=False
            # ),  # [752, 752] -> [376, 376]
            # build_norm_layer(norm_cfg, 32*double)[1],
            # nn.ReLU(),
            Sparse2DBasicBlockV(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        # x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(sp_tensor)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoder4xHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder4xHA", **kwargs
    ):
        super(SpMiddlePillarEncoder4xHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 64 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool3']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        # self.conv1 = spconv.SparseSequential(
        #     Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        #     Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        # )

        # self.conv2 = spconv.SparseSequential(
            # SparseConv2d(
            #     16*double, 32*double, 3, 2, padding=1, bias=False
            # ),  # [752, 752] -> [376, 376]
            # build_norm_layer(norm_cfg, 32*double)[1],
            # nn.ReLU(),
        #     Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
        #     Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
        # )

        self.conv3 = spconv.SparseSequential(
            # SparseConv2d(
            #     32*double, 64*double, 3, 2, padding=1, bias=False
            # ),  # [376, 376] -> [188, 188]
            # build_norm_layer(norm_cfg, 64*double)[1],
            # nn.ReLU(),
            Sparse2DBasicBlockV(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        # x_conv1 = self.conv1(sp_tensor)
        # x_conv2 = self.conv2(sp_tensor)
        x_conv3 = self.conv3(sp_tensor)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )

@BACKBONES.register_module
class SpMiddlePillarEncoder8xHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder8xHA", **kwargs
    ):
        super(SpMiddlePillarEncoder8xHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 128 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool4']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        # self.conv1 = spconv.SparseSequential(
        #     Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        #     Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        # )

        # self.conv2 = spconv.SparseSequential(
            # SparseConv2d(
            #     16*double, 32*double, 3, 2, padding=1, bias=False
            # ),  # [752, 752] -> [376, 376]
            # build_norm_layer(norm_cfg, 32*double)[1],
            # nn.ReLU(),
        #     Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
        #     Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
        # )

        # self.conv3 = spconv.SparseSequential(
            # SparseConv2d(
            #     32*double, 64*double, 3, 2, padding=1, bias=False
            # ),  # [376, 376] -> [188, 188]
            # build_norm_layer(norm_cfg, 64*double)[1],
            # nn.ReLU(),
        #     Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
        #     Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
        # )

        self.conv4 = spconv.SparseSequential(
            # SparseConv2d(
            #     64*double, 128*double, 3, 2, padding=1, bias=False
            # ),
            # build_norm_layer(norm_cfg, 128*double)[1],
            # nn.ReLU(),
            Sparse2DBasicBlockV(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128 * double, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
            Dense2DBasicBlock(256, 256, norm_cfg=norm_cfg),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        # x_conv1 = self.conv1(sp_tensor)
        # x_conv2 = self.conv2(sp_tensor)
        # x_conv3 = self.conv3(sp_tensor)
        x_conv4 = self.conv4(sp_tensor)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoderNext1HA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            drop_path_rate=0.7, layer_scale_init_value=1.0,
            name="SpMiddlePillarEncoderNext1HA", **kwargs
    ):
        super(SpMiddlePillarEncoderNext1HA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="LN", eps=1e-6)

        self.pillar_pooling1 = PillarMaxPoolingV2a1(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 16 * double],
            norm_cfg=norm_cfg,
            activation=nn.GELU(),
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.conv1 = spconv.SparseSequential(
            Sparse2DInverseBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res1"),
            Sparse2DInverseBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res1"),
            Sparse2DInverseBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res1"),
        )
        self.norm1 = build_norm_layer(norm_cfg, 16*double)[1]
        self.conv2 = spconv.SparseSequential(
            SparseConv2d(16*double, 32*double, 2, 2),  # [752, 752] -> [376, 376]
            Sparse2DInverseBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res2"),
            Sparse2DInverseBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res2"),
            Sparse2DInverseBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res2"),
        )
        self.norm2 = build_norm_layer(norm_cfg, 32 * double)[1]
        self.conv3 = spconv.SparseSequential(
            SparseConv2d(32*double, 64*double, 2, 2),  # [376, 376] -> [188, 188]
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
        )
        self.norm3 = build_norm_layer(norm_cfg, 64 * double)[1]
        self.conv4 = spconv.SparseSequential(
            SparseConv2d(64*double, 128*double, 2, 2),  # [376, 376] -> [188, 188]
            Sparse2DInverseBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res4"),
            Sparse2DInverseBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res4"),
            Sparse2DInverseBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res4"),

        )
        self.norm4 = build_norm_layer(norm_cfg, 128 * double)[1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(128*double, 256, kernel_size=2, stride=2),  # [376, 376] -> [188, 188]
            Dense2DInverseBasicBlock(256, 256, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                     layer_scale_init_value=layer_scale_init_value),
            Dense2DInverseBasicBlock(256, 256, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                     layer_scale_init_value=layer_scale_init_value),
            Dense2DInverseBasicBlock(256, 256, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                     layer_scale_init_value=layer_scale_init_value),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, spconv.SparseConv2d) or isinstance(m, spconv.SubMConv2d):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            else:
                print(m)

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        sp_tensor.dense()
        x_conv1 = self.conv1(sp_tensor)
        x_conv1 = x_conv1.replace_feature(self.norm1(x_conv1.features))
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = x_conv2.replace_feature(self.norm2(x_conv2.features))
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = x_conv3.replace_feature(self.norm3(x_conv3.features))
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.replace_feature(self.norm4(x_conv4.features))
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoderNextH(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            drop_path_rate=0.7, layer_scale_init_value=1.0,
            name="SpMiddlePillarEncoderNextH", **kwargs
    ):
        super(SpMiddlePillarEncoderNextH, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="LN", eps=1e-6)

        self.pillar_pooling1 = PillarMaxPoolingV2a1(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 16 * double],
            norm_cfg=norm_cfg,
            activation=nn.GELU(),
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.conv1 = spconv.SparseSequential(
            Sparse2DInverseBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res1"),
            Sparse2DInverseBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res1"),
            Sparse2DInverseBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res1"),
        )
        self.norm1 = build_norm_layer(norm_cfg, 16*double)[1]
        self.conv2 = spconv.SparseSequential(
            SparseConv2d(16*double, 32*double, 2, 2),  # [752, 752] -> [376, 376]
            Sparse2DInverseBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res2"),
            Sparse2DInverseBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res2"),
            Sparse2DInverseBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res2"),
        )
        self.norm2 = build_norm_layer(norm_cfg, 32 * double)[1]
        self.conv3 = spconv.SparseSequential(
            SparseConv2d(32*double, 64*double, 2, 2),  # [376, 376] -> [188, 188]
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
            Sparse2DInverseBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res3"),
        )
        self.norm3 = build_norm_layer(norm_cfg, 64 * double)[1]
        self.conv4 = spconv.SparseSequential(
            SparseConv2d(64*double, 128*double, 2, 2),  # [376, 376] -> [188, 188]
            Sparse2DInverseBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res4"),
            Sparse2DInverseBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res4"),
            Sparse2DInverseBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res4"),

        )
        self.norm4 = build_norm_layer(norm_cfg, 128 * double)[1]
        self.conv5 = spconv.SparseSequential(
            SparseConv2d(128*double, 256, 2, 2),  # [376, 376] -> [188, 188]
            Sparse2DInverseBasicBlock(256, 256, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res5"),
            Sparse2DInverseBasicBlock(256, 256, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res5"),
            Sparse2DInverseBasicBlock(256, 256, norm_cfg=norm_cfg, drop_path=drop_path_rate,
                                      layer_scale_init_value=layer_scale_init_value, indice_key="res5"),
        )
        self.norm5 = build_norm_layer(norm_cfg, 256)[1]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, spconv.SparseConv2d) or isinstance(m, spconv.SubMConv2d):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            else:
                print(m)

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        sp_tensor.dense()
        x_conv1 = self.conv1(sp_tensor)
        x_conv1 = x_conv1.replace_feature(self.norm1(x_conv1.features))
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = x_conv2.replace_feature(self.norm2(x_conv2.features))
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = x_conv3.replace_feature(self.norm3(x_conv3.features))
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.replace_feature(self.norm4(x_conv4.features))
        x_conv5 = self.conv5(x_conv4)
        x_conv5 = x_conv5.replace_feature(self.norm5(x_conv5.features))
        return dict(
            x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )


@BACKBONES.register_module
class SpMiddlePillarEncoderLH(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoderL", **kwargs
    ):
        super(SpMiddlePillarEncoderLH, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling0 = PillarMaxPoolingV2a(
            mlps=[6 + num_input_features, 8*double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool0']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.conv0 = spconv.SparseSequential(
            Sparse2DBasicBlockV(8 * double, 8 * double, norm_cfg=norm_cfg, indice_key="res0"),
            Sparse2DBasicBlock(8 * double, 8 * double, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv1 = spconv.SparseSequential(
            SparseConv2d(
                8 * double, 16 * double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 16 * double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                16*double, 32*double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32*double, 64*double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(64*double, 64*double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double, 128*double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128*double, 128*double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        self.conv5 = spconv.SparseSequential(
            SparseConv2d(
                128 * double, 256, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res5"),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res5"),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling0(xyz, xyz_batch_cnt, pt_features)

        x_conv0 = self.conv0(sp_tensor)
        x_conv1 = self.conv1(x_conv0)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        return dict(
            x_conv1=x_conv1,
            x_conv2=x_conv2,
            x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5,
        )


@BACKBONES.register_module
class SpMiddlePillarEncoderNeckMergeF(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoderNeckMergeF", **kwargs
    ):
        super(SpMiddlePillarEncoderNeckMergeF, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 16 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool1']['bev'],
            point_cloud_range=pc_range
        )  # [752, 752]

        self.pillar_pooling4 = PillarMaxPoolingV1a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 128 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool4']['bev'],
            point_cloud_range=pc_range
        )

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
            Sparse2DBasicBlock(16 * double, 16 * double, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2_ds = spconv.SparseSequential(
            SparseConv2d(
                16 * double, 32 * double, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32 * double)[1],
            nn.ReLU(),
        )

        self.conv2 = spconv.SparseSequential(
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3_ds = spconv.SparseSequential(
            SparseConv2d(
                32 * double, 64 * double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64 * double)[1],
            nn.ReLU(),
        )

        self.conv3 = spconv.SparseSequential(
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4_ds = spconv.SparseSequential(
            SparseConv2d(
                64 * double, 128 * double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128 * double)[1],
            nn.ReLU(),
        )

        self.conv4 = spconv.SparseSequential(
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.merge4 = Sparse2DMerge(128 * double, norm_cfg=norm_cfg, indice_key="res3")

        self.conv4_up = spconv.SparseSequential(
            conv2D3x3(128 * double, 128 * double, bias=False, indice_key="res3"),
            build_norm_layer(norm_cfg, 128 * double)[1],
            nn.ReLU(),
        )

        self.conv5_ds = spconv.SparseSequential(
            SparseConv2d(
                128 * double, 256 * double, 3, 2, padding=1, bias=False, indice_key="res4_cp"
            ),
            build_norm_layer(norm_cfg, 256 * double)[1],
            nn.ReLU()
        )

        self.conv5 = spconv.SparseSequential(
            Sparse2DBasicBlock(256 * double, 256 * double, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(256 * double, 256 * double, norm_cfg=norm_cfg, indice_key="res4"),
        )

        self.inverse_conv5 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(256*double, 128*double, 3, bias=False, indice_key="res4_cp"),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
        )

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)

        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2_ds(x_conv1)
        x_conv2 = self.conv2(x_conv2)
        x_conv3 = self.conv3_ds(x_conv2)
        x_conv3 = self.conv3(x_conv3)

        x_conv4 = self.conv4_ds(x_conv3)
        pool4 = self.pillar_pooling4(xyz, xyz_batch_cnt, pt_features, x_conv4)
        x_conv4 = self.merge4(x_conv4, pool4)
        x_conv4 = self.conv4(x_conv4)
        conv4_neck = self.conv4_up(x_conv4)

        x_conv5 = self.conv5_ds(x_conv4)
        x_conv5 = self.conv5(x_conv5)
        conv5_neck = self.inverse_conv5(x_conv5)

        x_conv4.features = torch.cat((conv4_neck.features, conv5_neck.features), dim=1)

        ret = x_conv4.dense()

        return ret, None


@BACKBONES.register_module
class SpMiddlePillarEncoder50HA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder", **kwargs
    ):
        super(SpMiddlePillarEncoder50HA, self).__init__()
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

        expansion = Sparse2DBottleneck.expansion
        self.conv1 = spconv.SparseSequential(
            Sparse2DBottleneckV(16*double, 16*double, norm_cfg=norm_cfg, indice_key="res0"),
            Sparse2DBottleneck(16*double*expansion, norm_cfg=norm_cfg, indice_key="res0"),
            Sparse2DBottleneck(16*double*expansion, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                16*double*expansion, 32*double*expansion, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 32*double*expansion)[1],
            nn.ReLU(),
            Sparse2DBottleneck(32*double*expansion, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBottleneck(32*double*expansion, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBottleneck(32*double*expansion, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBottleneck(32*double*expansion, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32*double*expansion, 64*double*expansion, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64*double*expansion)[1],
            nn.ReLU(),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBottleneck(64*double*expansion, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64*double*expansion, 128*double*expansion, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double*expansion)[1],
            nn.ReLU(),
            Sparse2DBottleneck(128*double*expansion, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBottleneck(128*double*expansion, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBottleneck(128*double*expansion, norm_cfg=norm_cfg, indice_key="res3"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
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
        sp_tensor = self.pillar_pooling1(xyz, xyz_batch_cnt, pt_features)
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)
        return dict(
            # x_conv1=x_conv1,
            # x_conv2=x_conv2,
            # x_conv3=x_conv3,
            x_conv4=x_conv4,
            x_conv5=x_conv5
        )
