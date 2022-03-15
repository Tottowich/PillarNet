from .pillar_encoder import PillarFeatureNet, PointPillarsScatter
from .voxel_encoder import VoxelFeatureExtractorV3, VoxelDownsample

__all__ = [
    "Identity",
    "VoxelDownsample",
    "VoxelFeatureExtractorV3",
    "PillarFeatureNet",
    "PointPillarsScatter",
]

import torch
from torch import nn
from ..registry import READERS

@READERS.register_module
class Identity(nn.Module):
    def __init__(self, pc_range, name="Identity", **kwargs):
        super(Identity, self).__init__()
        self.name = name
        self.pc_range = pc_range

    @torch.no_grad()
    def absl_to_relative(self, absolute):
        relative = absolute.detach().clone()
        relative[..., 0] -= self.pc_range[0]
        relative[..., 1] -= self.pc_range[1]
        relative[..., 2] -= self.pc_range[2]

        return relative

    def forward(self, example, **kwargs):
        points_list = example.pop("points")

        device = points_list[0].device

        xyz = []
        pt_features = []
        xyz_batch_cnt = []
        for points in points_list:
            points = self.absl_to_relative(points)

            xyz_batch_cnt.append(len(points))
            xyz.append(points[:, :3])
            pt_features.append(points[:, 3:])

        example["xyz"] = torch.cat(xyz, dim=0).contiguous()
        example["pt_features"] = torch.cat(pt_features, dim=0).contiguous()
        example["xyz_batch_cnt"] = torch.tensor(xyz_batch_cnt, dtype=torch.int32).to(device)
        return example