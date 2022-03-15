//
// Created by sgs on 2021/6/21.
//

#ifndef PILLAR_SCATTER_OPS_GPU_H
#define PILLAR_SCATTER_OPS_GPU_H

#include "cuda_utils.h"


int scatter_max_wrapper(at::Tensor index_tensor, at::Tensor src_tensor, at::Tensor arg_tensor, at::Tensor out_tensor);

int scatter_max_grad_wrapper(at::Tensor arg_tensor, at::Tensor grad_out_tensor, at::Tensor grad_src_tensor);

int update_index_kernel_wrapper(at::Tensor index2bev_tensor, at::Tensor index_tensor);

void update_index_kernel_launcher(int L, const int *index2bev, int *index);

void scatter_max_kernel_launcher(int C, int L, int M, const int *index, const float *src, int *arg, float *out);

void scatter_max_grad_kernel_launcher(int C, int M, const int *arg, const float *grad_out, float *grad_src);

#endif //SPARSE_OPS_SCATTER_OPS_GPU_H
