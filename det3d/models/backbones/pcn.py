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
from .base import conv2D3x3, Sparse2DBasicBlock, Sparse2DBasicBlockV, \
    Dense2DBasicBlock, Dense2DInverseBasicBlock

from det3d.ops.pillar_ops.pillar_modules import PillarMaxPoolingV2a


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
            block(64*double, 128, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(128, 128, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
            block(128, 128, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            dense_block(128, 256, 3, norm_cfg=norm_cfg, stride=2, padding=1),
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
                64*double, 128, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
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
                64*double, 128, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128*double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
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
                64*double, 128, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
        )

        self.conv5 = spconv.SparseSequential(
            SparseConv2d(
                128, 256, 3, 2, padding=1, bias=False
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
                64*double, 128, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, padding=1, bias=False),
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
                64*double, 128, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, padding=1, bias=False),
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
                64*double, 128, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, padding=1, bias=False),
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
                64*double, 128, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, padding=1, bias=False),
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
            Sparse2DBasicBlockV(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, padding=1, bias=False),
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

