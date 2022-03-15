import spconv
import torch
from functools import partial
from spconv import SparseConv2d
from torch import nn
from torch.nn import functional as F

from ..registry import BACKBONES
from ..utils import build_norm_layer
from .base import conv2D3x3, Sparse2DBasicBlock, Sparse2DBasicBlockV, Sparse2DCat, \
                Sparse2DMerge, Dense2DBasicBlock, Dense2DBasicBlockV

from det3d.ops.pillar_ops.pillar_modules import PillarMaxPoolingV1, PillarMaxPoolingV1a, \
    PillarMaxPoolingV2, PillarMaxPoolingV2a


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_cfg=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
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

@BACKBONES.register_module
class SpMiddlePillarEncoderA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=1,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoderA", **kwargs
    ):
        super(SpMiddlePillarEncoderA, self).__init__()
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
        ret = x_conv4.dense()

        return ret, None

@BACKBONES.register_module
class SpMiddlePillarEncoder(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=1,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddlePillarEncoder", **kwargs
    ):
        super(SpMiddlePillarEncoder, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling1 = PillarMaxPoolingV2(
            radius=pillar_cfg['pool1']['radius'],
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
        ret = x_conv4.dense()

        return ret, None

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

        self.pillar_pooling1 = PillarMaxPoolingV2(
            radius=pillar_cfg['pool1']['radius'],
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
            nn.ReLU(),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(32 * double, 32 * double, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                32 * double, 64 * double, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 64 * double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                64 * double, 128 * double, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 128 * double)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4_up = spconv.SparseSequential(
            conv2D3x3(128 * double, 128 * double, bias=False, indice_key="res3"),
            build_norm_layer(norm_cfg, 128 * double)[1],
            nn.ReLU(),
        )

        self.conv5 = spconv.SparseSequential(
            SparseConv2d(
                128 * double, 256 * double, 3, 2, padding=1, bias=False, indice_key="res4_cp"
            ),
            build_norm_layer(norm_cfg, 256 * double)[1],
            nn.ReLU(),
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
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        conv4_neck = self.conv4_up(x_conv4)

        x_conv5 = self.conv5(x_conv4)
        conv5_neck = self.inverse_conv5(x_conv5)

        x_conv4.features = torch.cat((conv4_neck.features, conv5_neck.features), dim=1)

        ret = x_conv4.dense()

        return ret, None


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
