//
// Created by sgs on 2021/2/9.
//

#ifndef GNN_OPS_SCATTER_OPS_GPU_H
#define GNN_OPS_SCATTER_OPS_GPU_H

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <assert.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>


#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


void scatter_max_kernel_launcher(int L, int C, const int *index, const float *src, int *arg, float *out);

void scatter_max_grad_kernel_launcher(int N, int C, const int *arg, const float *grad_out, float *grad_src);

void scatter_avg_kernel_launcher(int L, int N, int C, const int *index, const float *src, int *count, float *out);
void scatter_avg_grad_kernel_launcher(int L, int C, const int *index, const int *count, const float *grad_out, float *grad_src);


int scatter_max_wrapper(at::Tensor src_tensor, at::Tensor index_tensor, at::Tensor arg_tensor, at::Tensor out_tensor);
int scatter_max_grad_wrapper(at::Tensor arg_tensor, at::Tensor grad_out_tensor, at::Tensor grad_src_tensor);

int scatter_avg_wrapper(at::Tensor src_tensor, at::Tensor index_tensor, at::Tensor count_tensor, at::Tensor out_tensor);
int scatter_avg_grad_wrapper(at::Tensor index_tensor, at::Tensor count_tensor, at::Tensor grad_out_tensor,at::Tensor grad_src_tensor);

#endif //GNN_OPS_SCATTER_OPS_GPU_H
