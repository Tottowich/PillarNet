import numpy as np
try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv3d, SubMConv3d
except:
    import spconv
    from spconv import ops
    from spconv import SparseConv3d, SubMConv3d

from torch import nn
from torch.nn import functional as F

from ..registry import BACKBONES
from ..utils import build_norm_layer

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, dilation=1,
                   conv_type='subm', norm_cfg=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, padding=dilation,
                                 bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(),
    )

    return m

def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


@BACKBONES.register_module
class VoxelBackBone8x(nn.Module):
    def __init__(
            self, num_input_features=128, norm_cfg=None,
            name="VoxelBackBone8x", **kwargs
    ):
        super(VoxelBackBone8x, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="subm1"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU()
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_cfg=norm_cfg, indice_key='subm2'),
            block(32, 32, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_cfg=norm_cfg, indice_key='subm3'),
            block(64, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_cfg=norm_cfg, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_cfg=norm_cfg, indice_key='subm4'),
            block(64, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):
        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        ret = self.conv_out(x_conv4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            # 'conv1': x_conv1,
            # 'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret, multi_scale_voxel_features


@BACKBONES.register_module
class SpMiddleResNetFHD(nn.Module):
    def __init__(
        self, num_input_features=5, norm_cfg=None,
            name="SpMiddleResNetFHD", **kwargs
    ):
        super(SpMiddleResNetFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU()
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        ret = self.extra_conv(x_conv4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            # 'conv1': x_conv1,
            # 'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret, multi_scale_voxel_features


@BACKBONES.register_module
class SpMiddleResNet34FHD(nn.Module):
    def __init__(
        self, num_input_features=5, norm_cfg=None,
            name="SpMiddleResNetFHD", **kwargs
    ):
        super(SpMiddleResNet34FHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU()
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        ret = self.extra_conv(x_conv4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            # 'conv1': x_conv1,
            # 'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret, multi_scale_voxel_features


@BACKBONES.register_module
class SpMiddleResNetVoxel(nn.Module):
    def __init__(
        self, num_input_features=5, norm_cfg=None, name="SpMiddleResNetVoxel", **kwargs
    ):
        super(SpMiddleResNetVoxel, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU()
        )

        self.conv1 = spconv.SparseSequential(        
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(
                64, 128, 3, 2, padding=1, bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        # sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        sparse_shape = np.array(input_shape[::-1])

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        multi_scale_voxel_features = {
            # 'conv1': x_conv1,
            # 'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return multi_scale_voxel_features


@BACKBONES.register_module
class VoxelBackBoneEnc8x(nn.Module):
    def __init__(
            self, num_input_features=5, norm_cfg=None,
            name="VoxelBackBoneEnc8x", **kwargs
    ):
        super(VoxelBackBoneEnc8x, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="subm1"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU()
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_cfg=norm_cfg, indice_key='subm2'),
            block(32, 32, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_cfg=norm_cfg, indice_key='subm3'),
            block(64, 64, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(128, 128, 3, norm_cfg=norm_cfg, indice_key='subm4'),
            block(128, 128, 3, norm_cfg=norm_cfg, padding=1, indice_key='subm4'),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):
        # input: # [41, 1600, 1408]
        # input: # [41, 1600, 1408]
        # sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        sparse_shape = np.array(input_shape[::-1])

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        multi_scale_voxel_features = {
            # 'conv1': x_conv1,
            # 'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return multi_scale_voxel_features
