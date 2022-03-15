/*
Stacked-batch-data version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "ball_query_gpu.h"

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

int ball_query_wrapper_stack(float radius, int nsample,
    at::Tensor new_xyz_tensor, at::Tensor new_xyz_batch_cnt_tensor,
    at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(new_xyz_batch_cnt_tensor);
    CHECK_INPUT(xyz_batch_cnt_tensor);

    int B = new_xyz_batch_cnt_tensor.size(0);
    int M = new_xyz_tensor.size(0);

    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    const int *new_xyz_batch_cnt = new_xyz_batch_cnt_tensor.data<int>();
    const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
    int *idx = idx_tensor.data<int>();

    ball_query_kernel_launcher_stack(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);
    return 1;
}

int roi_cylinder_pool3d_wrapper(float extra, at::Tensor rois_tensor, at::Tensor roi_batch_cnt_tensor,
	at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor, at::Tensor idx_tensor){
	CHECK_INPUT(rois_tensor);
	CHECK_INPUT(xyz_tensor);
	CHECK_INPUT(roi_batch_cnt_tensor);
	CHECK_INPUT(xyz_batch_cnt_tensor);

	int B = roi_batch_cnt_tensor.size(0);
	int M = rois_tensor.size(0);
	int C = rois_tensor.size(1);
	int nsample = idx_tensor.size(1);

	const float *rois = rois_tensor.data<float>();
	const float *xyz = xyz_tensor.data<float>();
	const int *roi_batch_cnt = roi_batch_cnt_tensor.data<int>();
	const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
	int *idx = idx_tensor.data<int>();

	roi_cylinder_pool3d_kernel_launcher_stack(B, M, C, nsample, extra, rois, roi_batch_cnt, xyz, xyz_batch_cnt, idx);

	return 1;
}

int roi_cuboid_pool3d_wrapper(float extra, at::Tensor rois_tensor, at::Tensor roi_batch_cnt_tensor,
    at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor, at::Tensor idx_tensor){
	CHECK_INPUT(rois_tensor);
	CHECK_INPUT(xyz_tensor);
	CHECK_INPUT(roi_batch_cnt_tensor);
	CHECK_INPUT(xyz_batch_cnt_tensor);

	int B = roi_batch_cnt_tensor.size(0);
	int M = rois_tensor.size(0);
	int C = rois_tensor.size(1);
	int nsample = idx_tensor.size(1);

	const float *rois = rois_tensor.data<float>();
	const float *xyz = xyz_tensor.data<float>();
	const int *roi_batch_cnt = roi_batch_cnt_tensor.data<int>();
	const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
	int *idx = idx_tensor.data<int>();

	roi_cuboid_pool3d_kernel_launcher_stack(B, M, C, nsample, extra, rois, roi_batch_cnt, xyz, xyz_batch_cnt, idx);

	return 1;
}
