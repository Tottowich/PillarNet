/*
Stacked-batch-data version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#ifndef _STACK_BALL_QUERY_GPU_H
#define _STACK_BALL_QUERY_GPU_H

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
#include <torch/serialize/tensor.h>

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


int ball_query_wrapper_stack(float radius, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
   at::Tensor new_xyz_tensor, at::Tensor ball_tensor, at::Tensor idx_tensor);

void ball_query_kernel_launcher_stack(int B, int M, int G, int N, int nsample, float radius,
   const float *xyz, const int *xyz_batch_cnt, const float *new_xyz, const float *ball, int *idx);

#endif
