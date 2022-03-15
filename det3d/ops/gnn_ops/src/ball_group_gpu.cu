#include <stdio.h>
#include <stdlib.h>

#include "ball_group_gpu.h"

#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__global__ void ball_indice_kernel_stack(int N, int nsample, const int *ball_idx, const int *out_offsets,
    int *set_indices, int *set_new_indices) {
  // :param ball_idx: (N1 + N2 ..., nsample) tensor containing the indicies of features to group with
  // :param out_offsets: (N1 + N2 ...,) tensor of output indice
  // :param set_new_indices: (nindices) gathered points
  // :param set_indices: (nindices,) gathered query points

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N * nsample || ball_idx[tid] < 0) return;

  int ndx = tid / nsample;
  int odx = tid % nsample;

  if (ndx > 0) {
      odx += out_offsets[ndx - 1];
  }
  set_indices[odx] = ndx;
  set_new_indices[odx] = ball_idx[tid];
}

void ball_indice_kernel_launcher_stack(int N, int nsample, const int *ball_idx, const int *out_offsets,
    int *set_indices, int *set_new_indices) {

  cudaError_t err;
  dim3 blocks(DIVUP(N * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  ball_indice_kernel_stack<<<blocks, threads>>>(N, nsample, ball_idx, out_offsets, set_indices, set_new_indices);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void gather_feature_kernel(int nindices, int C, const int *set_indices, const float *features, float *out){
  // set_indices: (nindices,)         features: (N, C)
  // out: (nindices, C)

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nindices) return;

  set_indices += tid;
  out += tid * C;
  features += set_indices[0] * C;
  for (int c = 0; c < C; ++c){
    out[c] = features[c];
  }
}

__global__ void gather_feature_grad_kernel(int nindices, int C, const int *set_indices, const float *outGrad, float *inGrad){
  // set_indices: (nindices,)         grad_features: (nindices, C)
  // grad_out: (N, C)

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nindices) return;

  set_indices += tid;
  outGrad += tid * C;
  inGrad += set_indices[0] * C;
  for (int c = 0; c < C; ++c){
    atomicAdd(inGrad + c, outGrad[c]);
  }
}

void gather_feature_kernel_launcher(int nindices, int C, const int *set_indices, const float *features, float *out){
  cudaError_t err;
  dim3 blocks(DIVUP(nindices, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  gather_feature_kernel<<<blocks, threads>>>(nindices, C, set_indices, features, out);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void gather_feature_grad_kernel_launcher(int nindices, int C, const int *set_indices, const float *outGrad, float *inGrad){
  cudaError_t err;
  dim3 blocks(DIVUP(nindices, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  gather_feature_grad_kernel<<<blocks, threads>>>(nindices, C, set_indices, outGrad, inGrad);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
