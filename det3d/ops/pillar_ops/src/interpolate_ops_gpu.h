//
// Created by sgs on 2021/7/18.
//

#ifndef PILLAR_INTERPOLATE_OPS_GPU_H
#define PILLAR_INTERPOLATE_OPS_GPU_H

#include "cuda_utils.h"


int getInterpIndices2D(int kernel, int H, int W, float by, float bx,
					   at::Tensor indices_tensor, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
					   at::Tensor indice_kernel_tensor, at::Tensor inverse_dist_tensor);
int getInterpIndices3D(int kernel, int D, int H, int W, float vz, float vy, float vx,
					   at::Tensor indices_tensor, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
					   at::Tensor indice_kernel_tensor, at::Tensor inverse_dist_tensor);

void indice_interp2d_kernel_launcher(int M, int N, int B, int H, int W, int K, int kernel, float by, float bx,
									 const int *indices, int *gridIndice, const int *xyzBatchCnt, const float *xyz,
									 int *indiceKernel, float *inverseDist);
void indice_interp3d_kernel_launcher(int M, int N, int B, int D, int H, int W, int K, int kernel, float vz, float vy, float vx,
									 const int *indices, const int *xyzBatchCnt, const float *xyz,
									 int *gridIndice, int *indiceKernel, float *inverseDist);

void indiceInterpForward(at::Tensor features_tensor, at::Tensor indice_kernel_tensor, at::Tensor inverse_dist_tensor,
                         at::Tensor out_tensor);

void indiceInterpBackward(at::Tensor grad_out_tensor, at::Tensor indice_kernel_tensor,
                          at::Tensor inverse_dist_tensor, at::Tensor grad_in_tensor);

void interp_fwd_kernel_launcher(int N, int K, int C, const float *features,
                                const int *indiceKernel, const float *inverseDist, float *out);

void interp_bwd_kernel_launcher(int N, int K, int C, const float *outGrad,
                                const int *indiceKernel, const float *inverseDist, float *inGrad);

#endif //PILLAR_INTERPOLATE_OPS_GPU_H
