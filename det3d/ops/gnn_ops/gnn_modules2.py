import torch
import torch.nn as nn
from typing import List
from .gnn_utils import QueryAndGroup, QueryAndGroupAtt
from torch_scatter import scatter_max, scatter_mean,  scatter_add


def graph_scatter_max_fn(point_features, keypoint_index, dim_size):
    aggregated = point_features.new_zeros((point_features.shape[0], dim_size))
    scatter_max(point_features, keypoint_index, out=aggregated)
    return aggregated

def graph_scatter_mean_fn(point_features, keypoint_index, dim_size):
    aggregated = point_features.new_zeros((point_features.shape[0], dim_size))
    scatter_mean(point_features, keypoint_index, out=aggregated)
    return aggregated

def graph_scatter_sum_fn(point_features, keypoint_index, dim_size):
    aggregated = point_features.new_zeros((point_features.shape[0], dim_size))
    scatter_add(point_features, keypoint_index, out=aggregated)
    return aggregated

def graph_scatter_softmax_fn(point_features, keypoint_index, dim_size):
    keypoint_index = keypoint_index.long()

    @torch.no_grad()
    def decentralize(point_features, keypoint_index, dim_size):
        agg_max = graph_scatter_max_fn(point_features, keypoint_index, dim_size)
        return agg_max[:, keypoint_index]
    point_exp = torch.exp(point_features - decentralize(point_features, keypoint_index, dim_size))
    agg_sum = graph_scatter_sum_fn(point_exp, keypoint_index, dim_size)
    return point_exp / agg_sum[:, keypoint_index]

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
                 use_xyz: bool = True, use_label: bool = False, pool_method='max_pool'):
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
            self.groupers.append(QueryAndGroup(radius, nsample,
                                               use_xyz=use_xyz,
                                               use_label=use_label))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            if use_label:
                mlp_spec[0] += 1

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Linear(mlp_spec[k], mlp_spec[k + 1], bias=False),
                    nn.BatchNorm1d(mlp_spec[k + 1]),
                    nn.ReLU()
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

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, roi_label=None):
        """
       :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
       :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
       :param new_xyz: (M1 + M2 ..., 3)
       :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
       :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
       :return:
           // new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
           new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
       """
        group_features_list = []
        for k in range(len(self.groupers)):
            group_features, group_new_idx = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features, roi_label)  # ((M1 + M2)*nsample, C)
            # from tools.visual_utils.visualize_utils import draw_sphere_pts
            # from mayavi import mlab
            # points = xyz[xyz_batch_cnt[-1]:]
            # seeks = new_xyz[:new_xyz_batch_cnt[1]]
            # fig = draw_sphere_pts(points, color=(0.5, 0.5, 0.5))
            # fig = draw_sphere_pts(seeks, color=(1., 0., 1.), fig=fig, scale_factor=0.3)
            # mlab.show(stop=True)
            if len(group_features) == 0:
                group_features_list.append(group_features.new_zeros((new_xyz.shape[0], self.out_planes[k])))
                continue

            group_features = self.mlps[k](group_features)  # ((M1 + M2)*nsample, C)
            if self.pool_method == 'max_pool':
                group_features = graph_scatter_max_fn(group_features.transpose(0, 1),
                                                    group_new_idx, new_xyz.shape[0])  # (C, (M1 + M2))
            elif self.pool_method == 'avg_pool':
                group_features = graph_scatter_mean_fn(group_features.transpose(0, 1),
                                                     group_new_idx, new_xyz.shape[0])  # (C, (M1 + M2))
            else:
                raise NotImplementedError
            group_features = group_features.transpose(0, 1)  # (M1 + M2 ..., C)
            group_features_list.append(group_features)

        group_features = torch.cat(group_features_list, dim=1)  # (M1 + M2 ..., 2C)
        return None, group_features


class StackSAModuleAttMSG(nn.Module):

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]],
                 use_label: bool=False, pool_method='att_pool'):
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
            self.groupers.append(QueryAndGroupAtt(radius, nsample, use_label=use_label))

        mlp_spec = mlps[0]
        self.in_chn = mlp_spec[0]
        self.out_chn = mlp_spec[-1]

        att_mlps=nn.ModuleList()
        att_mlps.append(nn.Sequential(
            nn.Linear(5 + int(use_label), (mlp_spec[0]*mlp_spec[-1]) // 2, bias=False),
            nn.BatchNorm1d((mlp_spec[0]*mlp_spec[-1]) // 2),
            nn.ReLU(),
            nn.Linear((mlp_spec[0]*mlp_spec[-1]) // 2, (mlp_spec[0]*mlp_spec[-1]), bias=False),
            # nn.BatchNorm1d(mlp_spec[0]),
            # nn.ReLU()
        ))

        att_mlps.append(nn.Sequential(
            nn.BatchNorm1d((mlp_spec[0]*mlp_spec[-1])),
            nn.ReLU()
        ))
        # self.mlps.append(att_mlps)
        self.mlps = att_mlps

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

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt,
                features=None, empty_voxel_set_zeros=True, roi_label=None):
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
        group_features_list = []
        for k in range(len(self.groupers)):
            grouped_xyz, group_features, group_new_idx = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features, roi_label)  # ((M1 + M2)*nsample, C)

            if len(group_features) == 0:
                group_features_list.append(group_features.new_zeros((new_xyz.shape[0], self.out_chns[k])))
                continue

            # step 1. relative position encoding
            grouped_xyz = self.mlps[0](grouped_xyz)  # ((M1 + M2)*nsample, C*64)
            # group_features = torch.cat([grouped_xyz, group_features], dim=1)  # ((M1 + M2)*nsample, 2C)

            # grouped_xyz = graph_scatter_softmax_fn(grouped_xyz.transpose(0, 1), group_new_idx,
            #                                    new_xyz.shape[0])  # (C*64, (M1 + M2)*nsample)
            group_features = group_features.transpose(0, 1).unsqueeze(0) * grouped_xyz.view(self.out_chn, self.in_chn, -1)
            group_features = graph_scatter_sum_fn(group_features.view(self.out_chn*self.in_chn, -1), group_new_idx,
                                                  new_xyz.shape[0]).transpose(0, 1)  # ((M1 + M2), C)
            group_features = group_features.view(-1, self.out_chn, self.in_chn).sum(dim=1)

            group_features = self.mlps[1](group_features)
            group_features_list.append(group_features)

        group_features_list = torch.cat(group_features_list, dim=1)  # (M1 + M2 ..., 2C)
        return new_xyz, group_features_list