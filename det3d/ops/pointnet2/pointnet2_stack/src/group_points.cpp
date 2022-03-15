/*
Stacked-batch-data version of point grouping, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <torch/serialize/tensor.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <THC/THC.h>
#include "group_points_gpu.h"

extern THCState *state;
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


int group_points_grad_wrapper_stack(at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor idx_batch_cnt_tensor,
    at::Tensor features_batch_cnt_tensor, at::Tensor grad_features_tensor) {

    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(idx_batch_cnt_tensor);
    CHECK_INPUT(features_batch_cnt_tensor);
    CHECK_INPUT(grad_features_tensor);

    int B = idx_batch_cnt_tensor.size(0);
    int M = idx_tensor.size(0);
    int nsample = idx_tensor.size(1);
    int N = grad_features_tensor.size(0);
    int C = grad_features_tensor.size(1);

    const float *grad_out = grad_out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    const int *idx_batch_cnt = idx_batch_cnt_tensor.data<int>();
    const int *features_batch_cnt = features_batch_cnt_tensor.data<int>();
    float *grad_features = grad_features_tensor.data<float>();

    group_points_grad_kernel_launcher_stack(B, M, C, N, nsample, grad_out, idx, idx_batch_cnt, features_batch_cnt, grad_features);
    return 1;
}


int group_points_wrapper_stack(at::Tensor features_tensor, at::Tensor features_batch_cnt_tensor,
    at::Tensor idx_tensor, at::Tensor idx_batch_cnt_tensor, at::Tensor out_tensor) {

    CHECK_INPUT(features_tensor);
    CHECK_INPUT(features_batch_cnt_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(idx_batch_cnt_tensor);
    CHECK_INPUT(out_tensor);

    int B = idx_batch_cnt_tensor.size(0);
    int M = idx_tensor.size(0);
    int nsample = idx_tensor.size(1);
    int C = features_batch_cnt_tensor.size(1);

    const float *features = features_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    const int *features_batch_cnt = features_batch_cnt_tensor.data<int>();
    const int *idx_batch_cnt = idx_batch_cnt_tensor.data<int>();
    float *out = out_tensor.data<float>();

    group_points_kernel_launcher_stack(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, out);
    return 1;
}