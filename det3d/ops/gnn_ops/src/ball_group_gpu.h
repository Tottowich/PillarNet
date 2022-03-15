//
// Created by sgs on 2020/11/25.
//

#ifndef _STACK_BALL_GROUP_H
#define _STACK_BALL_GROUP_H

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


int ball_indice_wrapper_stack(at::Tensor ball_idx_tensor, at::Tensor out_offsets_tensor,
                              at::Tensor set_indices_tensor, at::Tensor set_new_indices_tensor);

void ball_indice_kernel_launcher_stack(int N, int nsample, const int *ball_idx, const int *out_offsets,
                                       int *set_indices, int *set_new_indices);

void gather_feature_kernel_launcher(int nindices, int C, const int *set_indices, const float *features, float *out);

void gather_feature_grad_kernel_launcher(int nindices, int C, const int *set_indices, const float *outGrad, float *inGrad);

int gather_feature_grad_wrapper(at::Tensor set_indices_tensor, at::Tensor grad_out_tensor, at::Tensor grad_features_tensor);

int gather_feature_wrapper(at::Tensor set_indices_tensor, at::Tensor features_tensor, at::Tensor new_features_tensor);

#endif //SRC_BALL_GROUP_H
