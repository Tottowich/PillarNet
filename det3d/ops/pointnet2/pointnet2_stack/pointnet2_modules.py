from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils
from det3d.models.utils import build_norm_layer


class StackPointnetSimple(nn.Module):
    def __init__(self, *, nsample: int, in_channels: int, multiplier: int, extra:float, cuboid=True, norm_cfg=None):
        """
        Args:
            nsample: int, number of points in each ball query
            pool_method: max_pool / avg_pool
        """
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.groupers = pointnet2_utils.QueryAndGroupTusimple(extra, nsample, cuboid)

        self.pre_norm = build_norm_layer(norm_cfg, in_channels)[1],

        self.pointnetfeature = nn.Sequential(
            nn.Conv1d(in_channels, 64 * multiplier, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, 64 * multiplier)[1],
            nn.ReLU(inplace=True),
            nn.Conv1d(64 * multiplier, 128 * multiplier, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, 128 * multiplier)[1],
            nn.ReLU(inplace=True),
            nn.Conv1d(128 * multiplier, 512 * multiplier, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, 512 * multiplier)[1],
        )

        self.post_mlps = nn.Sequential(
            nn.Linear(512 * multiplier, 256 * multiplier, bias=False),
            build_norm_layer(norm_cfg, 256 * multiplier)[1],
            nn.ReLU(inplace=True),
            nn.Linear(256 * multiplier, 256, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(inplace=True)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, batch_rois, features=None):
        """
        Args:
            xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            batch_rois: (B, M, 7+)
            features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        Returns:
            new_features: (B, M, C) tensor of the new_features descriptors
        """
        B, M = batch_rois.shape[:2]
        new_features = self.groupers(xyz, xyz_batch_cnt, batch_rois, features)  # (B*M, C, nsample)

        new_features = self.pointnetfeature(self.pre_norm(new_features))  # (B*M, C, nsample)
        new_features = torch.max(new_features, dim=2)[0]   # (B*M, C)
        new_features = self.post_mlps(new_features.view(B*M, -1))  # (B*M, 256)

        return new_features.view(B, M, -1)


class StackSAModuleMSG(nn.Module):

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]],
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M1 + M2, C, nsample)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[k](new_features)  # (1, C, M1 + M2 ..., nsample)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features


class StackPointnetFPModule(nn.Module):
    def __init__(self, *, mlp: List[int]):
        """
        Args:
            mlp: list of int
        """
        super().__init__()
        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, unknown, unknown_batch_cnt, known, known_batch_cnt, unknown_feats=None, known_feats=None):
        """
        Args:
            unknown: (N1 + N2 ..., 3)
            known: (M1 + M2 ..., 3)
            unknow_feats: (N1 + N2 ..., C1)
            known_feats: (M1 + M2 ..., C2)

        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        dist, idx = pointnet2_utils.three_nn(unknown, unknown_batch_cnt, known, known_batch_cnt)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)  # (N1 + N2 ..., C2 + C1)
        else:
            new_features = interpolated_feats
        new_features = new_features.permute(1, 0)[None, :, :, None]  # (1, C, N1 + N2 ..., 1)
        new_features = self.mlp(new_features)

        new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)  # (N1 + N2 ..., C)
        return new_features
