import torch
import torch.nn as nn
from . import gnn_cuda
from torch.autograd import Function, Variable


class ScatterMax(Function):
    @staticmethod
    def forward(ctx, src: torch.Tensor, index: torch.Tensor, len: int):
        """
        Args:
            ctx:
            src: (L, C)
            index: (L, )
        Returns:
            out: (len, C)
        """
        assert src.is_contiguous()
        assert index.is_contiguous()
        
        arg = index.new_zeros((len, src.shape[1]))
        out = src.new_zeros((len, src.shape[1]))
        gnn_cuda.scatter_max_wrapper(src, index, arg, out)
        
        ctx.for_backwards = (src.shape[0], src.shape[1], arg)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        L, C, arg = ctx.for_backwards
        grad_src = Variable(torch.cuda.FloatTensor(L, C).zero_())
        grad_out_data = grad_out.data.contiguous()
        
        gnn_cuda.scatter_max_grad_wrapper(arg, grad_out_data, grad_src)
        return grad_src, None, None

_scatter_max = ScatterMax.apply


class ScatterAvg(Function):
    @staticmethod
    def forward(ctx, src: torch.Tensor, index: torch.Tensor, len: int):
        """
        Args:
            ctx:
            src: (L, C)
            index: (L, )
        Returns:
            out: (len, C)
        """
        assert src.is_contiguous()
        assert index.is_contiguous()
        
        out = src.new_zeros((len, src.shape[1]))
        count = index.new_zeros(len)
        gnn_cuda.scatter_avg_wrapper(src, index, count, out)
        ctx.for_backwards = (src.shape[0], src.shape[1], index, count)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        L, C, index, count = ctx.for_backwards
        grad_src = Variable(torch.cuda.FloatTensor(L, C).zero_())
        grad_out_data = grad_out.data.contiguous()
        
        gnn_cuda.scatter_avg_grad_wrapper(index, count, grad_out_data, grad_src)
        return grad_src, None, None

_scatter_avg = ScatterAvg.apply

