import torch
import torch.nn as nn
from typing import List
from .gnn_utils import QueryAndGroup
from .scatter_utils import _scatter_max, _scatter_avg

def rotate_points_in_axis(points, angle):
    """
    :param points: (N, 3)
    :param angle: (N, )
    :return:
    """
    cosa = torch.cos(angle).unsqueeze(1); sina = torch.sin(angle).unsqueeze(1)
    row_1 = torch.cat([cosa, -sina], dim=1).unsqueeze(1)
    row_2 = torch.cat([sina, cosa], dim=1).unsqueeze(1)
    R = torch.cat([row_1, row_2], dim=1)

    xy_temp = torch.matmul(points[:, None, :2], R).squeeze(1)
    points = torch.cat((xy_temp, points[:, 2:3]), dim=1)

    return points


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

        self.out_planes = []
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
          
            mlp_spec = mlps[i]
            shared_mlps = []
            if pool_method == 'max_pool':
                self.groupers.append(QueryAndGroup(radius, nsample, use_xyz=use_xyz))
                mlp_spec[0] += int(use_xyz) * 3

                for k in range(len(mlp_spec) - 1):
                    shared_mlps.extend([
                        nn.Linear(mlp_spec[k], mlp_spec[k + 1], bias=False),
                        nn.BatchNorm1d(mlp_spec[k + 1]),
                        nn.ReLU()
                    ])
            elif pool_method == 'avg_pool':
                self.groupers.append(QueryAndGroup(radius, nsample, use_xyz=False))
                shared_mlps.extend([
                    nn.Linear(int(use_xyz) * 3, mlp_spec[1], bias=True),
                ])

            self.mlps.append(nn.Sequential(*shared_mlps))
            self.out_planes.append(mlp_spec[-1])

        self.pool_method = pool_method
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, rois, features=None):
        """
       :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
       :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
       :param new_xyz: (B, M, 6x6x6, 3)
       :param rois: (B, M, 7+C), [x y z h w l ...] or [x y z w l h ...]
       :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
       :return:
           new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
       """
        out_size = new_xyz.view(-1, 3).shape[0]

        group_features_list = []
        for k in range(len(self.groupers)):
            # from tools.visual_utils.visualize_utils import draw_sphere_pts
            # from mayavi import mlab
            # points = xyz[xyz_batch_cnt[-1]:]
            # seeks = new_xyz[:new_xyz_batch_cnt[1]]
            # fig = draw_sphere_pts(points, color=(0.5, 0.5, 0.5))
            # fig = draw_sphere_pts(seeks, color=(1., 0., 1.), fig=fig, scale_factor=0.3)
            # mlab.show(stop=True)
           
            if self.pool_method == 'max_pool':
                group_features, group_new_idx = self.groupers[k](
                    xyz, xyz_batch_cnt, new_xyz, rois, features)  # (nindices, C)

                if len(group_features) == 0:
                    group_features_list.append(group_features.new_zeros((out_size, self.out_planes[k])))
                    continue
                group_features = self.mlps[k](group_features)  # (nindices, C)
                group_features = _scatter_max(group_features, group_new_idx, out_size)
                # group_features = graph_scatter_max_fn(group_features.transpose(0, 1),
                #                                       group_new_idx.long(), new_xyz.shape[0])  # (C, (M1 + M2))
            elif self.pool_method == 'avg_pool':
                group_features, group_xyz, group_new_idx = self.groupers[k](
                    xyz, xyz_batch_cnt, new_xyz, rois, features, ret_xyz=True)  # (nindices, C)

                if len(group_features) == 0:
                    group_features_list.append(group_features.new_zeros((out_size, self.out_planes[k])))
                    continue
                group_features = self.mlps[k](group_xyz) * group_features  # (nindices, C)
                group_features = _scatter_avg(group_features, group_new_idx, out_size)
                # group_features = graph_scatter_mean_fn(group_features.transpose(0, 1),
                #                                        group_new_idx, new_xyz.shape[0])  # (C, (M1 + M2))
            else:
                raise NotImplementedError
            # group_features = group_features.transpose(0, 1)  # (M1 + M2 ..., C)
            group_features_list.append(group_features)

        group_features = torch.cat(group_features_list, dim=1)  # (M1 + M2 ..., 2C)
        return None, group_features

