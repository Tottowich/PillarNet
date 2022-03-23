import torch
from torch import nn
import numpy as np
try:
    import spconv.pytorch as spconv
except:
    import spconv

from ..utils import build_norm_layer
from ..registry import SECOND_STAGE
from det3d.core.utils.center_utils import (
    bilinear_interpolate_torch,
)


@SECOND_STAGE.register_module
class BEVFeatureNeck(nn.Module):
    def __init__(self, bone_features, roi_grid_size, pillar_size, out_stride, pc_start, share_channels, out_channels, **kwargs):
        super().__init__()
        self.pillar_size = pillar_size
        self.out_stride = out_stride
        self.pc_start = pc_start
        self.roi_grid_size = roi_grid_size

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        c_in = 0
        stride = int(8 / self.out_stride)
        self.bev_layer = nn.Sequential(
            nn.ConvTranspose2d(num_bev_features, out_channels, stride, stride=stride, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU()
        )
        c_in += out_channels

        self.second_layers = nn.ModuleList()
        self.second_layer_names = []

        for src_name, bone_cfg in bone_features.items():
            input_channels = bone_cfg.in_channels

            stride = bone_cfg.stride / self.out_stride
            self.PA_strides.append(int(stride))

            if stride >= 1:
                stride = int(stride)
                deblock = nn.Sequential(
                    nn.ConvTranspose2d(input_channels, out_channels, stride, stride=stride, bias=False),
                    build_norm_layer(norm_cfg, out_channels)[1],
                    nn.ReLU()
                )
            else:
                stride = int(np.round(1 / stride))
                deblock = spconv.SparseSequential(
                    spconv.SparseConv2d(input_channels, out_channels, kernel_size=stride, stride=stride, bias=False),
                    build_norm_layer(norm_cfg, out_channels)[1],
                    nn.ReLU()
                )

            self.PA_layers.append(deblock)
            self.PA_layer_names.append(src_name)

            c_in += out_channels

        self.share_conv = nn.Sequential(
            nn.Conv2d(c_in, share_channels, 3, stride=1, bias=False),
            build_norm_layer(norm_cfg, share_channels)[1],
            nn.ReLU()
        )

    def forward(self, example, batch_centers, num_point):
        bev_feature = example['bev_feature']
        bone_feature = example['bone_feature']

        batch_size = len(example['bev_feature'])

        cur_features = self.bev_layer(bev_feature)
        multi_scale_features = [cur_features]

        for k, src_name in enumerate(self.PA_layer_names):
            cur_features = bone_feature[src_name]

            if self.PA_strides[k] >= 1:
                if isinstance(cur_features, spconv.SparseConvTensor):
                    cur_features = cur_features.dense()
                cur_features = self.PA_layers[k](cur_features)
            else:
                cur_features = self.PA_layers[k](cur_features)
                cur_features = cur_features.dense()

            multi_scale_features.append(cur_features)

        multi_scale_features = torch.cat(multi_scale_features, dim=1)
        multi_scale_features = self.share_conv(multi_scale_features)

        ret_maps = []
        # N C H W -> N H W C
        multi_scale_features = multi_scale_features.permute(0, 2, 3, 1).contiguous()

        for batch_idx in range(batch_size):
            xs, ys = self.absl_to_relative(batch_centers[batch_idx])

            feature_map = bilinear_interpolate_torch(multi_scale_features[batch_idx],
                                                     xs.view(-1), ys.view(-1))  # N x C
            num_obi = xs.shape[0]
            ret_maps.append(feature_map.view(num_obi, -1))

        example['features'] = ret_maps
        return example

    # def forward(self, example, batch_centers, num_point):
    #     batch_size = len(example['bev_feature'])
    #     ret_maps = []
    #
    #     if isinstance(num_point, tuple):
    #         num_point = num_point[0] * num_point[1]
    #
    #     # N C H W -> N H W C
    #     example['bev_feature'] = example['bev_feature'].permute(0, 2, 3, 1).contiguous()
    #
    #     for batch_idx in range(batch_size):
    #         xs, ys = self.absl_to_relative(batch_centers[batch_idx])
    #
    #         feature_map = bilinear_interpolate_torch(example['bev_feature'][batch_idx],
    #                                                  xs.view(-1), ys.view(-1))  # N x C
    #
    #         # if num_point > 1:
    #         #     section_size = len(feature_map) // num_point
    #         #     feature_map = torch.cat([feature_map[i*section_size: (i+1)*section_size]
    #         #                              for i in range(num_point)], dim=1)
    #         num_obi = xs.shape[0]
    #         ret_maps.append(feature_map.view(num_obi, -1))
    #
    #     example['features'] = ret_maps
    #     return example


