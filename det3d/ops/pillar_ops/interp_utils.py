import torch
from torch.autograd import Variable
from torch.autograd import Function
from . import pillar_cuda


class SparseInterpolate2DFunction(Function):
    @staticmethod
    def forward(ctx, kernel:int, H:int, W:int, by:float, bx:float,
                indices: torch.Tensor, features: torch.Tensor,
                xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor):
        """
        xyz must be in relative coordinates in coordance with indices
        by bx is the equal size of bev grid.
        Args:
            indices: (M1+M2..., 3) [byx]
            features: (M1+M2..., C)
            xyz: (N1+N2..., 3) [xyz]
            xyz_batch_cnt: (N1, N2, ...)  batch_size
        """
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()
        assert indices.is_contiguous()
        assert features.is_contiguous()
        assert xyz.shape[1] == 3

        N = xyz.shape[0]
        M, C = features.shape

        K = (2 * kernel + 1) ** 2

        device = features.device
        indice_kernel = torch.full([N, K], -1, dtype=torch.int32, device=device, requires_grad=False)
        inverse_dist = torch.zeros([N, K], dtype=torch.float32, device=device, requires_grad=False)

        pillar_cuda.getInterpIndices2D(kernel, H, W, by, bx, indices, xyz, xyz_batch_cnt,
                                       indice_kernel, inverse_dist)

        point_features = features.new_zeros((N, C))
        pillar_cuda.indiceInterpForward(features, indice_kernel, inverse_dist, point_features)
        ctx.for_backwards = (M, C, indice_kernel, inverse_dist)
        return point_features

    @staticmethod
    def backward(ctx, grad_out):
        M, C, indice_kernel, inverse_dist = ctx.for_backwards

        grad_src = Variable(grad_out.new_zeros((M, C)))
        grad_out = grad_out.data.contiguous()
        pillar_cuda.indiceInterpBackward(grad_out, indice_kernel, inverse_dist, grad_src)
        return None, None, None, None, None, \
               None, grad_src, None, None

sparse_interpolate2d = SparseInterpolate2DFunction.apply


class SparseInterpolate3DFunction(Function):
    @staticmethod
    def forward(ctx, kernel:int, D:int, H:int, W:int, vz:float, vy:float, vx:float,
                indices: torch.Tensor, features: torch.Tensor,
                xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor):
        """
        Args:
            indices: (M1+M2..., 4) [bzyx]
            features: (M1+M2..., C)

            xyz: (N1+N2..., 3) [xyz]
            xyz_batch_cnt: (N1, N2, ...)  batch_size
        """
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()
        assert indices.is_contiguous()
        assert features.is_contiguous()
        assert xyz.shape[1] == 3

        M, C = features.shape
        N = xyz.shape[0]

        K = pow(2 * kernel + 1, 3)

        device = features.device
        indice_kernel = torch.full([N, K], -1, dtype=torch.int32, device=device, requires_grad=False)
        inverse_dist = torch.zeros([N, K], dtype=torch.float32, device=device, requires_grad=False)

        pillar_cuda.getInterpIndices3D(kernel, D, H, W, vz, vy, vx, indices, xyz, xyz_batch_cnt,
                                       indice_kernel, inverse_dist)

        point_features = features.new_zeros((N, C))
        pillar_cuda.indiceInterpForward(features, indice_kernel, inverse_dist, point_features)
        ctx.for_backwards = (M, C, indice_kernel, inverse_dist)
        return point_features

    @staticmethod
    def backward(ctx, grad_out):
        M, C, indice_kernel, inverse_dist = ctx.for_backwards

        grad_src = Variable(grad_out.new_zeros((M, C)))
        grad_out = grad_out.data.contiguous()
        pillar_cuda.indiceInterpBackward(grad_out, indice_kernel, inverse_dist, grad_src)

        return None, None, None, None, None, None, None, \
               None, grad_src, None, None

sparse_interpolate3d = SparseInterpolate3DFunction.apply