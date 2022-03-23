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
    Dense2DBasicBlock, Dense2DInverseBasicBlock, Sparse2DBottleneck, Sparse2DBottleneckV, Dense2DBottleneck, Sparse2DAttBlock
from det3d.models.utils import Sequential
from det3d.ops.pillar_ops.pillar_modules import PillarMaxPoolingV1a, PillarMaxPoolingV2a, PillarMaxPoolingV2b


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
class SpMiddleDoublePillarEncoderHA(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddleDoublePillarEncoderHA", **kwargs
    ):
        super(SpMiddleDoublePillarEncoderHA, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling2 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 32 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )

        self.pillar_pooling3 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 64 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool3']['bev'],
            point_cloud_range=pc_range
        )

        self.conv2_b1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2_1"),
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

        self.conv2_b2 = spconv.SparseSequential(
            Sparse2DBasicBlockV(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2_2"),
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

        self.conv4_b2s = self._make_layer(256, 256, 5, stride=1, norm_cfg=norm_cfg)

        self.deblock_b1 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128*double, 128, 3, stride=1, bias=False),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU()
        )
        self.deblock_b2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU()
        )

    def _make_layer(self, inplanes, planes, num_blocks, stride=1, norm_cfg=None):
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

        return block

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor1 = self.pillar_pooling2(xyz, xyz_batch_cnt, pt_features)
        x_conv2_b1 = self.conv2_b1(sp_tensor1)
        x_conv3_b1 = self.conv3_b1(x_conv2_b1)
        x_conv4_b1 = self.conv4_b1(x_conv3_b1)
        x_conv4_b1 = x_conv4_b1.dense()
        x_conv4_b1 = self.deblock_b1(x_conv4_b1)

        sp_tensor2 = self.pillar_pooling3(xyz, xyz_batch_cnt, pt_features)
        x_conv2_b2 = self.conv2_b2(sp_tensor2)
        x_conv3_b2 = self.conv3_b2(x_conv2_b2)
        x_conv3_b2 = x_conv3_b2.dense()
        x_conv4_b2 = self.conv4_b2(x_conv3_b2)
        x_conv4_b2 = self.conv4_b2s(x_conv4_b2)
        x_conv4_b2 = self.deblock_b2(x_conv4_b2)

        x_conv4 = torch.cat([x_conv4_b1, x_conv4_b2], dim=1)

        return dict(
            x_conv4=x_conv4,
        )


@BACKBONES.register_module
class SpMiddleTriplePillarEncoderHAM(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddleTriplePillarEncoderHAM", **kwargs
    ):
        super(SpMiddleTriplePillarEncoderHAM, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling2 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 32 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )

        self.pillar_pooling3 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 64 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool3']['bev'],
            point_cloud_range=pc_range
        )

        self.pillar_pooling4 = PillarMaxPoolingV2b(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[8 + num_input_features, 128 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool4']['bev'],
            point_cloud_range=pc_range
        )

        self.conv2_b1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2_1"),
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

        self.conv2_b2 = spconv.SparseSequential(
            Sparse2DBasicBlockV(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2_2"),
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

        self.conv4_b2s = self._make_layer(256, 256, 5, stride=1, norm_cfg=norm_cfg)

        self.deblock_b1 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128*double, 128, 3, stride=1, bias=False),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU()
        )
        self.deblock_b2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU()
        )

        # self.conv4_b3 = spconv.SparseSequential(
        #     Sparse2DBasicBlockV(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4_3"),
        #     Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4_3")
        # )

    def _make_layer(self, inplanes, planes, num_blocks, stride=1, norm_cfg=None):
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

        return block

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor1 = self.pillar_pooling2(xyz, xyz_batch_cnt, pt_features)
        x_conv2_b1 = self.conv2_b1(sp_tensor1)
        x_conv3_b1 = self.conv3_b1(x_conv2_b1)
        x_conv4_b1 = self.conv4_b1(x_conv3_b1)
        x_conv4_b1 = x_conv4_b1.dense()

        sp_tensor3 = self.pillar_pooling4(xyz, xyz_batch_cnt, pt_features)
        # sp_tensor3 = self.conv4_b3(sp_tensor3)

        # module factor
        x_conv4_b3 = sp_tensor3.dense()
        x_conv4_b1 = x_conv4_b3 + x_conv4_b1
        x_conv4_b1 = self.deblock_b1(x_conv4_b1)

        sp_tensor2 = self.pillar_pooling3(xyz, xyz_batch_cnt, pt_features)
        x_conv2_b2 = self.conv2_b2(sp_tensor2)
        x_conv3_b2 = self.conv3_b2(x_conv2_b2)
        x_conv3_b2 = x_conv3_b2.dense()
        x_conv4_b2 = self.conv4_b2(x_conv3_b2)
        x_conv4_b2 = self.conv4_b2s(x_conv4_b2)
        x_conv4_b2 = self.deblock_b2(x_conv4_b2)

        x_conv4 = torch.cat([x_conv4_b1, x_conv4_b2], dim=1)

        return dict(
            x_conv4=x_conv4,
        )


@BACKBONES.register_module
class SpMiddleTriplePillarEncoderHAAtt(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddleTriplePillarEncoderHAAtt", **kwargs
    ):
        super(SpMiddleTriplePillarEncoderHAAtt, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling2 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 32 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )

        self.pillar_pooling3 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 64 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool3']['bev'],
            point_cloud_range=pc_range
        )

        self.pillar_pooling4 = PillarMaxPoolingV2b(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[8 + num_input_features, 128 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool4']['bev'],
            point_cloud_range=pc_range
        )

        self.conv2_b1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2_1"),
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

        self.conv2_b2 = spconv.SparseSequential(
            Sparse2DBasicBlockV(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2_2"),
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

        self.conv4_b2s = self._make_layer(256, 256, 5, stride=1, norm_cfg=norm_cfg)

        self.deblock_b1 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128*double, 128, 3, stride=1, bias=False),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU()
        )
        self.deblock_b2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU()
        )

        self.conv4_b3 = spconv.SparseSequential(
            Sparse2DBasicBlockV(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4_3"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4_3")
        )

        self.conv4_b3_att = Sparse2DAttBlock(128 * double, 128, kernel_size=7, norm_cfg=norm_cfg, indice_key="res4_3")

    def _make_layer(self, inplanes, planes, num_blocks, stride=1, norm_cfg=None):
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

        return block

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor1 = self.pillar_pooling2(xyz, xyz_batch_cnt, pt_features)
        x_conv2_b1 = self.conv2_b1(sp_tensor1)
        x_conv3_b1 = self.conv3_b1(x_conv2_b1)
        x_conv4_b1 = self.conv4_b1(x_conv3_b1)
        x_conv4_b1 = x_conv4_b1.dense()
        x_conv4_b1 = self.deblock_b1(x_conv4_b1)

        sp_tensor3 = self.pillar_pooling4(xyz, xyz_batch_cnt, pt_features)
        sp_tensor3 = self.conv4_b3(sp_tensor3)
        sp_tensor3 = self.conv4_b3_att(sp_tensor3)
        x_conv4_att = sp_tensor3.dense()
        x_conv4_b1 = x_conv4_b1 + x_conv4_b1 * x_conv4_att

        sp_tensor2 = self.pillar_pooling3(xyz, xyz_batch_cnt, pt_features)
        x_conv2_b2 = self.conv2_b2(sp_tensor2)
        x_conv3_b2 = self.conv3_b2(x_conv2_b2)
        x_conv3_b2 = x_conv3_b2.dense()
        x_conv4_b2 = self.conv4_b2(x_conv3_b2)
        x_conv4_b2 = self.conv4_b2s(x_conv4_b2)
        x_conv4_b2 = self.deblock_b2(x_conv4_b2)

        x_conv4 = torch.cat([x_conv4_b1, x_conv4_b2], dim=1)

        return dict(
            x_conv4=x_conv4,
        )


@BACKBONES.register_module
class SpMiddleTriplePillarEncoderHAMAtt(nn.Module):
    def __init__(
            self, norm_cfg=None, pillar_cfg=None,
            num_input_features=2, double=2,
            pc_range=[-75.2, -75.2, 75.2, 75.2],
            name="SpMiddleTriplePillarEncoderHAMAtt", **kwargs
    ):
        super(SpMiddleTriplePillarEncoderHAMAtt, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.pillar_pooling2 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 32 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool2']['bev'],
            point_cloud_range=pc_range
        )

        self.pillar_pooling3 = PillarMaxPoolingV2a(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[6 + num_input_features, 64 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool3']['bev'],
            point_cloud_range=pc_range
        )

        self.pillar_pooling4 = PillarMaxPoolingV2b(
            # radius=pillar_cfg['pool1']['radius'],
            mlps=[8 + num_input_features, 128 * double],
            norm_cfg=norm_cfg,
            bev_size=pillar_cfg['pool4']['bev'],
            point_cloud_range=pc_range
        )

        self.conv2_b1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(32*double, 32*double, norm_cfg=norm_cfg, indice_key="res2_1"),
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

        self.conv2_b2 = spconv.SparseSequential(
            Sparse2DBasicBlockV(64 * double, 64 * double, norm_cfg=norm_cfg, indice_key="res2_2"),
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

        self.conv4_b2s = self._make_layer(256, 256, 5, stride=1, norm_cfg=norm_cfg)

        self.deblock_b1 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128*double, 128, 3, stride=1, bias=False),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU()
        )
        self.deblock_b2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU()
        )

        self.conv4_b3 = spconv.SparseSequential(
            Sparse2DBasicBlockV(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4_3"),
            Sparse2DBasicBlock(128 * double, 128 * double, norm_cfg=norm_cfg, indice_key="res4_3")
        )

        self.conv4_b3_att = Sparse2DAttBlock(128 * double, 128, kernel_size=7, norm_cfg=norm_cfg, indice_key="res4_3")

    def _make_layer(self, inplanes, planes, num_blocks, stride=1, norm_cfg=None):
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

        return block

    def forward(self, xyz, xyz_batch_cnt, pt_features):
        sp_tensor1 = self.pillar_pooling2(xyz, xyz_batch_cnt, pt_features)
        x_conv2_b1 = self.conv2_b1(sp_tensor1)
        x_conv3_b1 = self.conv3_b1(x_conv2_b1)
        x_conv4_b1 = self.conv4_b1(x_conv3_b1)
        x_conv4_b1 = x_conv4_b1.dense()

        sp_tensor3 = self.pillar_pooling4(xyz, xyz_batch_cnt, pt_features)
        sp_tensor3 = self.conv4_b3(sp_tensor3)

        # module factor
        x_conv4_b3 = sp_tensor3.dense()
        x_conv4_b1 = x_conv4_b3 + x_conv4_b1
        x_conv4_b1 = self.deblock_b1(x_conv4_b1)

        # attention
        sp_tensor3 = self.conv4_b3_att(sp_tensor3)
        x_conv4_att = sp_tensor3.dense()

        sp_tensor2 = self.pillar_pooling3(xyz, xyz_batch_cnt, pt_features)
        x_conv2_b2 = self.conv2_b2(sp_tensor2)
        x_conv3_b2 = self.conv3_b2(x_conv2_b2)
        x_conv3_b2 = x_conv3_b2.dense()
        x_conv4_b2 = self.conv4_b2(x_conv3_b2)
        x_conv4_b2 = self.conv4_b2s(x_conv4_b2)
        x_conv4_b2 = self.deblock_b2(x_conv4_b2)
        x_conv4_b2 = x_conv4_b2 + x_conv4_b2 * x_conv4_att

        x_conv4 = torch.cat([x_conv4_b1, x_conv4_b2], dim=1)

        return dict(
            x_conv4=x_conv4,
        )

