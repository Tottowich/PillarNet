import torch
import torch.nn as nn
from . import gnn_cuda
from torch.autograd import Function, Variable


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, rois: torch.Tensor):
        """
        Args:
            ctx:
            radius: float, radius of the ball query
            nsample: int, maximum number of rois per point
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (B, M, 6x6x6, 3) centers of the ball query
            rois: (B, M, 7+C), [x, y, z, h, w, l, ry, ...] or [x, y, z, w, l, h, ry, ...]

        Returns:
            idx: (N1 + N2 ..., nsample) tensor with the indicies of the features
        """
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()
        assert new_xyz.is_contiguous()
        assert len(new_xyz.shape) == 4

        ball = torch.cat((rois[:, :, :3], torch.norm(rois[:, :, 3:6], p=2, dim=-1, keepdim=True)), dim=-1)
        idx = xyz_batch_cnt.new_zeros(xyz.shape[0], nsample).fill_(-1)

        gnn_cuda.ball_query_wrapper(radius, xyz, xyz_batch_cnt, new_xyz, ball, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None

ball_query = BallQuery.apply

class BallIndice(Function):
    """
    Args:
        ctx:
        ball_idx: (N1 + N2 ..., nsample) tensor containing the indicies of features to group with

    Returns:
        set_indices: (nindices,)
        set_new_indices: (nindices,)
    """

    @staticmethod
    def forward(ctx, ball_idx: torch.Tensor):
        assert ball_idx.is_contiguous()

        out_offsets = torch.cumsum(torch.sum((ball_idx > -1).int(), dim=1), dim=0, dtype=ball_idx.dtype)
        nindices = out_offsets[-1].item()

        set_indices = ball_idx.new_zeros(nindices)
        set_new_indices = ball_idx.new_zeros(nindices)
        gnn_cuda.ball_indice_wrapper(ball_idx, out_offsets, set_indices, set_new_indices)

        return set_indices, set_new_indices

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None

ball_indice = BallIndice.apply


class GatherFeature(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, set_indices: torch.Tensor):
        """
        Args:
            ctx:
            features: (N, C)
            set_indices: (nindices, )
        Returns:
            new_features: (nindices, C)
        """
        assert features.is_contiguous()
        assert set_indices.is_contiguous()

        new_features = features.new_zeros((set_indices.shape[0], features.shape[1]))
        gnn_cuda.gather_feature_wrapper(set_indices, features, new_features)
        
        ctx.for_backwards = (features.shape[0], features.shape[1], set_indices)
        return new_features

    @staticmethod
    def backward(ctx, grad_out):
        N, C, set_indices = ctx.for_backwards
        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())
        grad_out_data = grad_out.data.contiguous()
        
        gnn_cuda.gather_feature_grad_wrapper(set_indices, grad_out_data, grad_features)
        return grad_features, None

gather_feature = GatherFeature.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
            use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, rois: torch.Tensor,
                features: torch.Tensor = None, ret_xyz = False):
        """
        Args:
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (B, M, 6x6x6, 3) centers of the ball query
            rois: (B, M, 7+C), [x y z h w l ...] or [x y z w l h ...]
            features: (N1 + N2 ..., C) tensor of features to group

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum()

        # idx: (M1 + M2 ..., nsample) set_xyz: (nindices, 3) new_set_xyz: (nindices, 3)
        ball_idx = ball_query(self.radius, self.nsample, xyz, xyz_batch_cnt, new_xyz, rois)

        set_indices, set_new_indices = ball_indice(ball_idx)
        group_xyz = gather_feature(xyz, set_indices)
        group_new_xyz = gather_feature(new_xyz.view(-1, 3), set_new_indices)  # (nindices, 3)

        if features is not None:
            group_features = gather_feature(features, set_indices)  # (nindices, C)
            if self.use_xyz:
                group_features = torch.cat([group_xyz - group_new_xyz, group_features], dim=1)  # (nindices, 3+C)
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            group_features = group_xyz - new_xyz

        if ret_xyz: return group_features, group_xyz - group_new_xyz, set_new_indices
        return group_features, set_new_indices
