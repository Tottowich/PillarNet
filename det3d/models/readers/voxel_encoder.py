import torch
from torch import nn
from torch.nn import functional as F

from ..registry import READERS


@READERS.register_module
class VoxelFeatureExtractorV3(nn.Module):
    def __init__(
        self, num_input_features=4, name="VoxelFeatureExtractorV3", **kwargs
    ):
        super(VoxelFeatureExtractorV3, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, data, **kwargs):
        features = data["features"]
        num_voxels = data["num_voxels"]

        assert self.num_input_features == features.shape[-1]

        points_mean = features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)

        data["features"] = points_mean.contiguous()
        return data

@READERS.register_module
class VoxelFeatureExtractor(nn.Module):
    def __init__(
        self, pc_range, num_input_features=4, name="VoxelFeatureExtractor", **kwargs
    ):
        super(VoxelFeatureExtractor, self).__init__()
        self.name = name
        self.num_input_features = num_input_features
        self.pc_range = pc_range

    @torch.no_grad()
    def absl_to_relative(self, absolute):
        relative = absolute.detach().clone()
        relative[..., 0] -= self.pc_range[0]
        relative[..., 1] -= self.pc_range[1]
        relative[..., 2] -= self.pc_range[2]

        return relative

    def forward(self, data, **kwargs):
        features = data["features"]
        num_voxels = data["num_voxels"]

        assert self.num_input_features == features.shape[-1]

        points_mean = features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)

        data["features"] = points_mean.contiguous()

        points_list = data.pop("points")
        device = points_list[0].device

        xyz = []
        pt_features = []
        xyz_batch_cnt = []
        for points in points_list:
            points = self.absl_to_relative(points)

            xyz_batch_cnt.append(len(points))
            xyz.append(points[:, :3])
            pt_features.append(points[:, 3:])

        data["xyz"] = torch.cat(xyz, dim=0).contiguous()
        data["pt_features"] = torch.cat(pt_features, dim=0).contiguous()
        data["xyz_batch_cnt"] = torch.tensor(xyz_batch_cnt, dtype=torch.int32).to(device)

        return data


@READERS.register_module
class VoxelDownsample(nn.Module):
    def __init__(self, voxel, pc_range, max_points=5, method='unique', name="VoxelDownsample"):
        super(VoxelDownsample, self).__init__()
        self.name = name
        self.method = method

        assert method in ['unique', 'average']
        self.max_points = max_points
        self.voxel = voxel
        self.pc_range = pc_range

    @torch.no_grad()
    def absl_to_relative(self, absolute):
        relative = absolute.detach().clone()
        relative[..., 0] -= self.pc_range[0]
        relative[..., 1] -= self.pc_range[1]
        relative[..., 2] -= self.pc_range[2]

        return relative

    def forward(self, data, **kwargs):
        points_list = data.pop("points")
        device = points_list[0].device

        xyz = []
        pt_features = []
        xyz_batch_cnt = []
        for points in points_list:
            points = self.absl_to_relative(points)

            xyz_batch_cnt.append(len(points))
            xyz.append(points[:, :3])
            pt_features.append(points[:, 3:])

        data["xyz"] = torch.cat(xyz, dim=0).contiguous()
        data["pt_features"] = torch.cat(pt_features, dim=0).contiguous()
        data["xyz_batch_cnt"] = torch.tensor(xyz_batch_cnt, dtype=torch.int32).to(device)
        return data

