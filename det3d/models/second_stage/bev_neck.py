import torch
from torch import nn
import numpy as np
try:
    import spconv.pytorch as spconv
except:
    import spconv

from ..utils import build_norm_layer
from ..registry import SECOND_STAGE


@SECOND_STAGE.register_module
class BEVFeatureNeck(nn.Module):
    def __init__(self, bone_cfg, num_channels, share_channels, out_stride, norm_cfg=None, **kwargs):
        super().__init__()
        self.bone_cfg = bone_cfg

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        c_in = 0
        self.second_layers = nn.ModuleList()
        self.second_layer_names = []
        for layer_name, cfg in bone_cfg.items():
            c_in += num_channels
            stride =  cfg['stride'] / out_stride
            if stride > 1:
                stride = int(stride)
                deblock = nn.Sequential(
                    nn.ConvTranspose2d(
                        cfg['planes'],
                        num_channels,
                        stride,
                        stride=stride,
                        bias=False,
                    ),
                    build_norm_layer(
                        norm_cfg,
                        num_channels,
                    )[1],
                    nn.ReLU(),
                )
            else:
                stride = np.round(1 / stride).astype(np.int64)
                deblock = nn.Sequential(
                    nn.Conv2d(
                        cfg['planes'],
                        num_channels,
                        stride,
                        stride=stride,
                        bias=False,
                    ),
                    build_norm_layer(
                        norm_cfg,
                        num_channels,
                    )[1],
                    nn.ReLU(),
                )
            self.second_layer_names.append(layer_name)
            self.second_layers.append(deblock)

        self.shared_conv = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(c_in, share_channels, 3, stride=1, bias=False),
            build_norm_layer(norm_cfg, share_channels)[1],
            nn.ReLU(),
        )

    def forward(self, example, a=None, b=None):
        bev_feature = example['bev_feature']
        bone_feature = example['bone_feature']

        k = 0
        bev_features = []
        if 'bev' in self.second_layer_names:
            k = 1
            bev_features.append(self.second_layers[0](bev_feature))

        for kk in range(k, len(self.second_layer_names)):
            layer_name = self.second_layer_names[kk]
            x = bone_feature[layer_name]
            if isinstance(x, spconv.SparseConvTensor):
                x = x.dense()
            bev_features.append(self.second_layers[k](x))

        bev_features = torch.cat(bev_features, dim=1)
        bev_features = self.shared_conv(bev_features)
        example['bev_feature'] = bev_features
        return example
