import torch.nn as nn
from typing import List
try:
    import spconv.pytorch as spconv
except:
    import spconv
from .pillar_utils import PillarQueryAndGroup, PillarQueryAndGroupV1, PillarQueryAndGroupV1a, \
    PillarQueryAndGroupV2, PillarQueryAndGroupV2a, bev_spatial_shape
from .scatter_utils import scatter_max, scatter_bev_max
from .interp_utils import sparse_interpolate2d, sparse_interpolate3d
from det3d.models.utils import build_norm_layer


class PillarMaxPoolingV1(nn.Module):
    def __init__(self, radius:float, mlps: List[int], bev_size:float, point_cloud_range:List[float], norm_cfg=None):
        super().__init__()

        self.groups = PillarQueryAndGroupV1(radius, bev_size, point_cloud_range)

        shared_mlp = []
        for k in range(len(mlps) - 1):
            shared_mlp.extend([
                nn.Conv1d(mlps[k], mlps[k + 1], 1, bias=False),
                build_norm_layer(norm_cfg, mlps[k + 1])[1],
                nn.ReLU()
            ])
        self.shared_mlps = nn.Sequential(*shared_mlp)

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, xyz, xyz_batch_cnt, point_features, sp_pillars):
        """
        Args:
            xyz: (N1+N2..., 3)
            xyz_batch_cnt:  (N1, N2, ...)
            point_features: (N1+N2..., C)
            pillars: (M1+M2..., 3) [byx]
            spatial_shape: [B, H, W]
        Return:
            pillar_features: (M, C)
        """
        pillar_indices = sp_pillars.indices
        spatial_shape = sp_pillars.spatial_shape

        pillar_set_indices, group_features = \
                self.groups(xyz, xyz_batch_cnt, point_features, pillar_indices, spatial_shape)
        group_features = self.shared_mlps(group_features.transpose(1, 0).unsqueeze(dim=0))  # (1, C, L)

        pillar_features = scatter_max(group_features.squeeze(dim=0), pillar_set_indices, pillar_indices.shape[0])  # (C, M)
        pillar_features = pillar_features.transpose(1, 0).contiguous()   # (M, C)
        return pillar_features

class PillarMaxPoolingV1a(nn.Module):
    def __init__(self, mlps: List[int], bev_size:float, point_cloud_range:List[float], norm_cfg=None):
        super().__init__()

        self.groups = PillarQueryAndGroupV1a(bev_size, point_cloud_range)

        shared_mlp = []
        for k in range(len(mlps) - 1):
            shared_mlp.extend([
                nn.Conv1d(mlps[k], mlps[k + 1], 1, bias=False),
                build_norm_layer(norm_cfg, mlps[k + 1])[1],
                nn.ReLU()
            ])
        self.shared_mlps = nn.Sequential(*shared_mlp)

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, xyz, xyz_batch_cnt, point_features, sp_pillars):
        """
        Args:
            xyz: (N1+N2..., 3)
            xyz_batch_cnt:  (N1, N2, ...)
            point_features: (N1+N2..., C)
            pillars: (M1+M2..., 3) [byx]
            spatial_shape: [B, H, W]
        Return:
            pillar_features: (M, C)
        """
        pillar_indices = sp_pillars.indices
        spatial_shape = sp_pillars.spatial_shape

        pillar_set_indices, group_features = \
                self.groups(xyz, xyz_batch_cnt, point_features, pillar_indices, spatial_shape)
        group_features = self.shared_mlps(group_features.transpose(1, 0).unsqueeze(dim=0))  # (1, C, L)

        pillar_features = scatter_max(group_features.squeeze(dim=0), pillar_set_indices, pillar_indices.shape[0])  # (C, M)
        pillar_features = pillar_features.transpose(1, 0).contiguous()   # (M, C)
        return pillar_features

class PillarMaxPoolingV2(nn.Module):
    def __init__(self, radius:float, mlps: List[int], bev_size:float, point_cloud_range:List[float], norm_cfg=None):
        super().__init__()

        self.bev_width, self.bev_height = bev_spatial_shape(point_cloud_range, bev_size)

        self.groups = PillarQueryAndGroupV2(radius, bev_size, point_cloud_range)

        shared_mlp = []
        for k in range(len(mlps) - 1):
            shared_mlp.extend([
                nn.Conv1d(mlps[k], mlps[k + 1], kernel_size=1, bias=False),
                build_norm_layer(norm_cfg, mlps[k + 1])[1],
                nn.ReLU()
            ])
        self.shared_mlps = nn.Sequential(*shared_mlp)

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, xyz, xyz_batch_cnt, point_features):
        """
        Args:
            xyz: (N1+N2..., 3)
            xyz_batch_cnt:  (N1, N2, ...)
            point_features: (N1+N2..., C)
            spatial_shape: [B, H, W]
        Return:
            pillars: (M1+M2..., 3) [byx]
            pillar_features: (M, C)
        """
        B = xyz_batch_cnt.shape[0]

        pillar_indices, pillar_set_indices, group_features = \
                self.groups(xyz, xyz_batch_cnt, point_features)
        group_features = self.shared_mlps(group_features.transpose(1, 0).unsqueeze(dim=0))  # (1, C, L)

        pillar_features = scatter_max(group_features.squeeze(dim=0), pillar_set_indices, pillar_indices.shape[0])   # (C, M)
        pillar_features = pillar_features.transpose(1, 0)   # (M, C)
        return spconv.SparseConvTensor(pillar_features, pillar_indices, (self.bev_height, self.bev_width), B)

class PillarMaxPoolingV2a(nn.Module):
    def __init__(self, mlps: List[int], bev_size:float, point_cloud_range:List[float], norm_cfg=None):
        super().__init__()

        self.bev_width, self.bev_height = bev_spatial_shape(point_cloud_range, bev_size)

        self.groups = PillarQueryAndGroupV2a(bev_size, point_cloud_range)

        shared_mlp = []
        for k in range(len(mlps) - 1):
            shared_mlp.extend([
                nn.Conv1d(mlps[k], mlps[k + 1], kernel_size=1, bias=False),
                build_norm_layer(norm_cfg, mlps[k + 1])[1],
                nn.ReLU()
            ])
        self.shared_mlps = nn.Sequential(*shared_mlp)

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, xyz, xyz_batch_cnt, point_features):
        """
        Args:
            xyz: (N1+N2..., 3)
            xyz_batch_cnt:  (N1, N2, ...)
            point_features: (N1+N2..., C)
            spatial_shape: [B, H, W]
        Return:
            pillars: (M1+M2..., 3) [byx]
            pillar_features: (M, C)
        """
        B = xyz_batch_cnt.shape[0]

        pillar_indices, pillar_set_indices, group_features = \
                self.groups(xyz, xyz_batch_cnt, point_features)
        group_features = self.shared_mlps(group_features.transpose(1, 0).unsqueeze(dim=0))  # (1, C, L)

        pillar_features = scatter_max(group_features.squeeze(dim=0), pillar_set_indices, pillar_indices.shape[0])   # (C, M)
        pillar_features = pillar_features.transpose(1, 0)   # (M, C)
        return spconv.SparseConvTensor(pillar_features, pillar_indices, (self.bev_height, self.bev_width), B)


class PillarMaxPoolingV2a1(nn.Module):
    def __init__(self, mlps: List[int], bev_size: float, point_cloud_range: List[float],
                 norm_cfg=None, activation=nn.ReLU()):
        super().__init__()

        self.bev_width, self.bev_height = bev_spatial_shape(point_cloud_range, bev_size)

        self.groups = PillarQueryAndGroupV2a(bev_size, point_cloud_range)

        shared_mlp = []
        for k in range(len(mlps) - 1):
            shared_mlp.extend([
                nn.Linear(mlps[k], mlps[k + 1]),
                build_norm_layer(norm_cfg, mlps[k + 1])[1],
                activation
            ])
        self.shared_mlps = nn.Sequential(*shared_mlp)

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, point_features):
        """
        Args:
            xyz: (N1+N2..., 3)
            xyz_batch_cnt:  (N1, N2, ...)
            point_features: (N1+N2..., C)
            spatial_shape: [B, H, W]
        Return:
            pillars: (M1+M2..., 3) [byx]
            pillar_features: (M, C)
        """
        B = xyz_batch_cnt.shape[0]

        pillar_indices, pillar_set_indices, group_features = \
            self.groups(xyz, xyz_batch_cnt, point_features)
        group_features = self.shared_mlps(group_features)  # (L, C)

        pillar_features = scatter_max(group_features.transpose(1, 0).contiguous(),
                                      pillar_set_indices, pillar_indices.shape[0])  # (C, M)
        pillar_features = pillar_features.transpose(1, 0)  # (M, C)
        return spconv.SparseConvTensor(pillar_features, pillar_indices, (self.bev_height, self.bev_width), B)


class PillarMaxPooling(nn.Module):
    def __init__(self, radius:float, mlps: List[int], bev_size:float, bev_flag:bool, point_cloud_range:List[float], norm_cfg=None):
        super().__init__()

        self.bev_flag = bev_flag
        self.bev_width = round((point_cloud_range[3] - point_cloud_range[0]) / bev_size)
        self.bev_height = round((point_cloud_range[4] - point_cloud_range[1]) / bev_size)

        self.groups = PillarQueryAndGroup(radius, bev_size, bev_flag, point_cloud_range)

        shared_mlp = []
        for k in range(len(mlps) - 1):
            shared_mlp.extend([
                nn.Conv1d(mlps[k], mlps[k + 1], 1, bias=False),
                build_norm_layer(norm_cfg, mlps[k + 1])[1],
                nn.ReLU()
            ])
        self.shared_mlps = nn.Sequential(*shared_mlp)
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, xyz, xyz_batch_cnt, point_features):
        """
        Args:
            xyz: (N1+N2..., 3)
            xyz_batch_cnt:  (N1, N2, ...)
            point_features: (N1+N2..., C)
        Return:
            pillar_features: (B, C, H, W) or (M, C)
        """
        B = xyz_batch_cnt.shape[0]

        pillar_indices, pillar_set_indices, group_features, indice2bev = \
            self.groups(xyz, xyz_batch_cnt, point_features)
        group_features = self.shared_mlps(group_features.transpose(1, 0).unsqueeze(dim=0))  # (1, C, L)

        if self.bev_flag:
            pillar_features = scatter_bev_max(group_features.squeeze(dim=0), pillar_set_indices, indice2bev,
                                              B * self.bev_height * self.bev_width)  # (C, B, H, W)
            pillar_features = pillar_features.view(-1, B, self.bev_height, self.bev_width).permute(1, 0, 2, 3)  # (B, C, H, W)
            return pillar_features
        else:
            pillar_features = scatter_max(group_features.squeeze(dim=0), pillar_set_indices, pillar_indices.shape[0])  # (C, M)
            pillar_features = pillar_features.transpose(1, 0)   # (1, C, M)
            return spconv.SparseConvTensor(pillar_features, pillar_indices, (self.bev_height, self.bev_width), B)


class SparseInterpolate3D(nn.Module):
    def __init__(self, kernel, voxel_size):
        super().__init__()

        self.kernel = kernel
        self.voxel_size = list(voxel_size[::-1])  # [vz vy vx]

    def forward(self, sparse_features, xyz, xyz_batch_cnt):
        # Note: xyz must be in velodyne coordinatess

        assert isinstance(sparse_features, spconv.SparseConvTensor)
        features = sparse_features.features
        indices = sparse_features.indices
        spatial_shape = sparse_features.spatial_shape  # [D, H, W]

        out_features = sparse_interpolate3d(self.kernel, *spatial_shape,
                                            *self.voxel_size, indices, features,
                                            xyz, xyz_batch_cnt)
        return out_features


class SparseInterpolate2D(nn.Module):
    def __init__(self, kernel, bev_size):
        super().__init__()

        self.kernel = kernel
        self.bev_size = list(bev_size[::-1])  # [by bx]

    def forward(self, sparse_features, xyz, xyz_batch_cnt):
        #  Note: xyz must be in velodyne coordinatess

        assert isinstance(sparse_features, spconv.SparseConvTensor)
        features = sparse_features.features
        indices = sparse_features.indices
        spatial_shape = sparse_features.spatial_shape  # [D, H, W]

        out_features = sparse_interpolate2d(self.kernel, *spatial_shape,
                                            *self.bev_sizes, indices, features,
                                            xyz, xyz_batch_cnt)

        return out_features