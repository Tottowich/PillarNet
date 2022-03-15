from typing import List
import torch
from torch.autograd import Variable
from torch.autograd import Function
from . import points_cuda


def points_in_rois(rois, points, extra=1.5):
    """
    fetch enough points in each grid region of proposal.
    :param rois: (K, 7 + C) [x, y, z, w, l, h, ry] in raw lidar data
    :param points: (N, 3)
    :return
        in_boxes_batch (M,)  points batch index
        in_boxes_points (M, 3) points in voxel coord
    """
    assert rois.is_contiguous()
    assert points.is_contiguous()

    in_mask = torch.cuda.BoolTensor(points.shape[0]).zero_()
    points_cuda.query_points_in_boxes(extra, rois, points, in_mask)

    return points[in_mask]

def furthest_point_downsample(npoint, points):
    idx = furthest_point_sample(npoint, points)
    return gather_points(points, idx)

def voxel_downsample(points, voxel, pc_range, method='unique', max_points=5):
    assert method in ['average', 'unique']

    if method == 'unique':
        ds_points = voxel_unique_sample(voxel, pc_range, points)
    else:
        ds_points = voxel_average_sample(voxel, pc_range, max_points, points)

    return ds_points

def voxel_downsample_relative(points, voxel, pc_range, method='unique', max_points=5):
    assert method in ['average', 'unique']

    if method == 'unique':
        ds_points = voxel_unique_sample_relative(voxel, pc_range, points)
    else:
        ds_points = voxel_average_sample_relative(voxel, pc_range, max_points, points)

    return ds_points


class VoxelAverageDownsample(Function):
    @staticmethod
    def forward(ctx, voxel: float, pc_range: List[float], max_points: int, points: torch.Tensor):
        """
        voxel-based sample for unordered points,
             by averaging points lying in each voxel for points with limited number.
        :param points: (N, C) [x y z ...]
        :param voxel: voxel size for uniform sampling
        :param pc_range: [lx ly lz hx hy hz]
        :param max_points: average maximum number of points for the resulting value in each voxel.
        :return:
            uniform sampling points
        """
        assert points.is_contiguous()
        lx, ly, lz, hx, hy, hz = pc_range

        # points tensor is inplace changed
        order_points, mask = points_cuda.voxel_average_down_sample(points, voxel, max_points,
                                                     lx, ly, lz, hx, hy, hz)
        return order_points[mask]

    @staticmethod
    def backward(ctx, a=None):
        return None

voxel_average_sample = VoxelAverageDownsample.apply

class VoxelAverageDownsampleRelative(Function):
    @staticmethod
    def forward(ctx, voxel: float, pc_range: List[float], max_points: int, points: torch.Tensor):
        """
        voxel-based sample for unordered points,
             by averaging points lying in each voxel for points with limited number.
        :param points: (N, C) [x y z ...]
        :param voxel: voxel size for uniform sampling
        :param pc_range: [lx ly lz hx hy hz]
        :param max_points: average maximum number of points for the resulting value in each voxel.
        :return:
            uniform sampling points
        """
        assert points.is_contiguous()
        VX = round(pc_range[3] - pc_range[0]) / voxel
        VY = round(pc_range[4] - pc_range[1]) / voxel
        VZ = round(pc_range[5] - pc_range[2]) / voxel

        # points tensor is inplace changed
        order_points, mask = points_cuda.voxel_average_down_sample_v1(points, voxel, max_points,
                                                                      int(VX), int(VY), int(VZ))
        return order_points[mask]

    @staticmethod
    def backward(ctx, a=None):
        return None

voxel_average_sample_relative = VoxelAverageDownsampleRelative.apply

class VoxelUniqueDownsample(Function):
    @staticmethod
    def forward(ctx, voxel: float, pc_range: List[float], points: torch.Tensor):
        """
        voxel-based sample for unordered points,
             by averaging points lying in each voxel for points with limited number.
        :param points: (N, C) [x y z ...]
        :param voxel: voxel size for uniform sampling
        :param pc_range: [lx ly lz hx hy hz]
        :return:
            uniform sampling points
        """
        assert points.is_contiguous()
        lx, ly, lz, hx, hy, hz = pc_range

        mask = points_cuda.voxel_unique_down_sample(points, voxel, lx, ly, lz, hx, hy, hz)
        return points[mask]

    @staticmethod
    def backward(ctx, a=None):
        return None

voxel_unique_sample = VoxelUniqueDownsample.apply

class VoxelUniqueDownsampleRelative(Function):
    @staticmethod
    def forward(ctx, voxel: float, pc_range: List[float], points: torch.Tensor):
        """
        voxel-based sample for unordered points,
             by averaging points lying in each voxel for points with limited number.
        :param points: (N, C) [x y z ...]  in relative coordinates
        :param voxel: voxel size for uniform sampling
        :param pc_range: [lx ly lz hx hy hz]
        :return:
            uniform sampling points
        """
        assert points.is_contiguous()
        VX = round(pc_range[3] - pc_range[0]) / voxel
        VY = round(pc_range[4] - pc_range[1]) / voxel
        VZ = round(pc_range[5] - pc_range[2]) / voxel

        mask = points_cuda.voxel_unique_down_sample_v1(points, voxel, int(VX), int(VY), int(VZ))
        return points[mask]

    @staticmethod
    def backward(ctx, a=None):
        return None

voxel_unique_sample_relative = VoxelUniqueDownsampleRelative.apply


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, npoint: int, points: torch.Tensor):
        """
        Args:
            ctx:
            points: (N, C) where N > npoint
            npoint: int, number of features in the sampled set

        Returns:
            idx: (npoint) tensor containing the set
        """
        assert points.is_contiguous()

        N, _ = points.size()
        idx = torch.cuda.IntTensor(npoint)
        temp = torch.cuda.FloatTensor(N).fill_(1e10)

        points_cuda.furthest_point_sampling_single_wrapper(npoint, points, temp, idx)
        return idx

    @staticmethod
    def backward(a=None):
        return None

furthest_point_sample = FurthestPointSampling.apply

class GatherPoints(Function):
    @staticmethod
    def forward(ctx, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param points: (N， C) [x y z ...]
        :param idx: (npoint) index tensor of the points to gather
        :return:
            output: (npoint, C)
        """
        assert points.is_contiguous()
        assert idx.is_contiguous()

        N, C = points.size()
        out = torch.cuda.FloatTensor(idx.shape[0], C)

        points_cuda.gather_points_wrapper(idx, points, out)

        ctx.for_backwards = (idx, N, C)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        idx, N, C = ctx.for_backwards

        grad_points = Variable(torch.cuda.FloatTensor(N, C).zero_())
        grad_out_data = grad_out.data.contiguous()
        points_cuda.gather_points_grad_wrapper(idx, grad_out_data, grad_points.data)
        return grad_points, None
gather_points = GatherPoints.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, features_batch_cnt: torch.Tensor,
                idx: torch.Tensor, idx_batch_cnt: torch.Tensor):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
            idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with

        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        assert features.is_contiguous()
        assert features_batch_cnt.is_contiguous()
        assert idx.is_contiguous()
        assert idx_batch_cnt.is_contiguous()

        assert features.shape[0] == features_batch_cnt.sum(), \
            'features: %s, features_batch_cnt: %s' % (str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == idx_batch_cnt.sum(), \
            'idx: %s, idx_batch_cnt: %s' % (str(idx.shape), str(idx_batch_cnt))

        B = idx_batch_cnt.shape[0]
        M, nsample = idx.size()
        N, C = features.size()
        output = torch.cuda.FloatTensor(M, C, nsample)

        points_cuda.group_points_wrapper(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, output)

        ctx.for_backwards = (B, N, idx, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

        M, C, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())

        grad_out_data = grad_out.data.contiguous()
        points_cuda.group_points_grad_wrapper(B, M, C, N, nsample, grad_out_data, idx,
                                               idx_batch_cnt, features_batch_cnt, grad_features.data)
        return grad_features, None, None, None


grouping_operation = GroupingOperation.apply

