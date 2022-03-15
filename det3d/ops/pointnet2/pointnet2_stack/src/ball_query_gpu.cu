/*
Stacked-batch-data version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_gpu.h"
#include "cuda_utils.h"


__global__ void ball_query_kernel_stack(int B, int M, float radius, int nsample, \
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx) {
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= M) return;

    int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += new_xyz_batch_cnt[k];
        bs_idx = k;
    }

    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];
    // for (int k = 0; k < bs_idx; k++) new_xyz_batch_start_idx += new_xyz_batch_cnt[k];

    new_xyz += pt_idx * 3;
    xyz += xyz_batch_start_idx * 3;
    idx += pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int n = xyz_batch_cnt[bs_idx];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
    if (cnt == 0) idx[0] = -1;
}


__global__ void roi_cylinder_pool3d_kernel_stack(int B, int M, int C, int nsample, float extra,
    const float *rois, const int *roi_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx) {
	// :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
	// :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
	// :param rois: (M1 + M2 ..., C)
	// :param roi_batch_cnt: (batch_size), [M1, M2, ...]  [x y z w l h ry ...]
	// output:
	//      idx: (M1+M2..., nsample)
	int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pt_idx >= M) return;

	int bs_idx = 0, pt_cnt = roi_batch_cnt[0];
	for (int k = 1; k < B; k++){
		if (pt_idx < pt_cnt) break;
		pt_cnt += roi_batch_cnt[k];
		bs_idx = k;
	}

	int xyz_batch_start_idx = 0;
	for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];

	xyz += xyz_batch_start_idx * 3;
	rois += pt_idx * C;
	idx += pt_idx * nsample;

	float new_x = rois[0];
	float new_y = rois[1];
	float new_z = rois[2];
	float dx = rois[3];
    float dy = rois[4];
	int n = xyz_batch_cnt[bs_idx];

	float radius = sqrtf(dx * dx + dy * dy) + extra;
	float radius2 = radius * radius;

	int cnt = 0;
	for (int k = 0; k < n; ++k) {
		float x = xyz[k * 3 + 0];
		float y = xyz[k * 3 + 1];
		float z = xyz[k * 3 + 2];
		float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
		if (d2 < radius2){
			idx[cnt] = k;
			++cnt;
			if (cnt >= nsample) break;
		}
	}
}

__device__ inline bool checkPointInBox3d(float extra, const float *pt, const float *box3d){
	// check point whether in bev of box3d
	// box3d: [x y z dx dy dz ry]

	float x = pt[0], y = pt[1];

	float cx = box3d[0], cy = box3d[1], dx = box3d[3], dy = box3d[4];
	float cosa = cos(-box3d[6]), sina = sin(-box3d[6]);

	float local_x = (x - cx) * cosa + (y - cy) * (-sina);
	float local_y = (x - cx) * sina + (y - cy) * cosa;

	bool in_flag = (fabs(local_x) < dx / 2. + extra) && (fabs(local_y) < dy / 2. + extra);
	return in_flag;
}

__global__ void roi_cuboid_pool3d_kernel_stack(int B, int M, int C, int nsample, float extra,
    const float *rois, const int *roi_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx) {
	// :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
	// :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
	// :param rois: (M1 + M2 ..., C)
	// :param roi_batch_cnt: (batch_size), [M1, M2, ...]  [x y z w l h ry ...]
	// output:
	//      idx: (M1+M2..., nsample)
	int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pt_idx >= M) return;

	int bs_idx = 0, pt_cnt = roi_batch_cnt[0];
	for (int k = 1; k < B; k++){
		if (pt_idx < pt_cnt) break;
		pt_cnt += roi_batch_cnt[k];
		bs_idx = k;
	}

	int xyz_batch_start_idx = 0;
	for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];

	xyz += xyz_batch_start_idx * 3;
	rois += pt_idx * C;
	idx += pt_idx * nsample;

	int n = xyz_batch_cnt[bs_idx];

	int cnt = 0;
	for (int k = 0; k < n; ++k) {
		if (checkPointInBox3d(extra, xyz + k * 3, rois)){
			idx[cnt] = k;
			++cnt;
			if (cnt >= nsample) break;
		}
	}
}

void ball_query_kernel_launcher_stack(int B, int M, float radius, int nsample,
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx){
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel_stack<<<blocks, threads>>>(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void roi_cylinder_pool3d_kernel_launcher_stack(int B, int M, int C, int nsample, float extra,
	const float *rois, const int *roi_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx){
	cudaError_t err;

	dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);

	roi_cylinder_pool3d_kernel_stack<<<blocks, threads>>>(B, M, C, nsample, extra, rois, roi_batch_cnt,
														  xyz, xyz_batch_cnt, idx);
	// cudaDeviceSynchronize();  // for using printf in kernel function
	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

void roi_cuboid_pool3d_kernel_launcher_stack(int B, int M, int C, int nsample, float extra,
   const float *rois, const int *roi_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx){
	cudaError_t err;

	dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);

	roi_cuboid_pool3d_kernel_stack<<<blocks, threads>>>(B, M, C, nsample, extra, rois, roi_batch_cnt,
														xyz, xyz_batch_cnt, idx);
	// cudaDeviceSynchronize();  // for using printf in kernel function
	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

