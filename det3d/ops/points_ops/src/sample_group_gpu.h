//
// Created by sgs on 2021/2/5.
//

#ifndef SPARSE_INTERP_SAMPLE_GROUP_GPU_H
#define SPARSE_INTERP_SAMPLE_GROUP_GPU_H

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

int query_points_in_boxes(float extra, torch::Tensor rois_tensor, torch::Tensor points_tensor, torch::Tensor points_mask_tensor);

std::tuple<at::Tensor, at::Tensor> voxel_average_down_sample(at::Tensor points_tensor, float voxel, float max_points,
									 float lx, float ly, float lz, float hx, float hy, float hz);
std::tuple<at::Tensor, at::Tensor> voxel_average_down_sample_v1(at::Tensor points_tensor, float voxel, float max_points,
																int VX, int VY, int VZ);

at::Tensor voxel_unique_down_sample(at::Tensor points_tensor, float voxel,
									float lx, float ly, float lz, float hx, float hy, float hz);
at::Tensor voxel_unique_down_sample_v1(at::Tensor points_tensor, float voxel, int VX, int VY, int VZ);

void points_in_boxes_kernel_launcher(int N, int M, int CP, int CR, float extra,
									 const float *rois, const float *points, bool *points_mask);

void average_points_kernel_launcher(int N, int C, int max_points, const long *sort_voxel_indice, float *points, bool *mask);

void unique_points_kernel_launcher(int N, const long *sort_voxel_indice, const long *sort_index, bool *mask);

void voxel_indice_kernel_launcher(int N, int C, float voxel, float lx, float ly, float lz, float hx, float hy, float hz,
								  const float *points, long *voxel_indice);
void voxel_indice_v1_kernel_launcher(int N, int C, float voxel, int VX, int VY, int VZ,
									 const float *points, long *voxel_indice);

int furthest_point_sampling_single_wrapper(int M, at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor);

void furthest_point_sampling_kernel_launcher(int B, int N, int M, int C, const float *dataset, float *temp, int *idxs);

void gather_points_kernel_launcher(int M, int C, const int *idx, const float *points, float *out);
int gather_points_wrapper(at::Tensor idx_tensor, at::Tensor points_tensor, at::Tensor out_tensor);

void gather_points_grad_kernel_launcher(int M, int C, const int *idx, const float *grad_out, float *grad_points);
int gather_points_grad_wrapper(at::Tensor idx_tensor, at::Tensor grad_out_tensor, at::Tensor grad_points_tensor);

int group_points_wrapper_stack(int B, int M, int C, int nsample,
                               at::Tensor features_tensor, at::Tensor features_batch_cnt_tensor,
                               at::Tensor idx_tensor, at::Tensor idx_batch_cnt_tensor, at::Tensor out_tensor);

void group_points_kernel_launcher_stack(int B, int M, int C, int nsample,
      const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out);

int group_points_grad_wrapper_stack(int B, int M, int C, int N, int nsample,
                                    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor idx_batch_cnt_tensor,
                                    at::Tensor features_batch_cnt_tensor, at::Tensor grad_features_tensor);

void group_points_grad_kernel_launcher_stack(int B, int M, int C, int N, int nsample,
      const float *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, float *grad_features);

#endif //SPARSE_INTERP_SAMPLE_GROUP_GPU_H
