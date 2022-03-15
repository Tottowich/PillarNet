//
// Created by sgs on 2021/2/9.
//

#include "atomics.cuh"
#include "scatter_ops_gpu.h"


#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__global__ void scatter_max_kernel(int L, int C, const int *index, const float *src, float *out){
  // index: (L) src: (L, C)  out: (N, C)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= L * C) return;

  int pid = tid / C;
  int cid = tid % C;

  index += pid;
  src += tid;

  atomMax(out + index[0] * C + cid, src[0]);
}

__global__ void scatter_arg_max_kernel(int L, int C, const int *index, const float *src, const float *out, int *arg){
// index: (L) src: (L, C)  out: (N, C)    arg: (N, C)

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= L * C) return;

  int pid = tid / C;
  int cid = tid % C;

  index += pid;
  src += tid;
  int odx = index[0];
  out += odx * C + cid;
  arg += odx * C + cid;

  if (abs(src[0] - out[0]) < 1e-5){
    arg[0] = pid;
  }
}

__global__ void scatter_max_grad_kernel(int N, int C, const int *arg, const float *grad_out, float *grad_src){
  // arg: (N, C)  grad_src: (L, C)  grad_out: (N, C)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N * C) return;

  int pid = tid / C;
  int cid = tid % C;

  grad_out += tid;
  arg += tid;

  grad_src[arg[0] * C + cid] = grad_out[0];
}

__global__ void scatter_count_kernel(int L, const int *index, int *count){
  // index: (L)  count: (N,)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= L) return;

  index += tid;
  atomAdd(count + index[0], 1);
}

__global__ void scatter_avg_kernel(int L, int C, const int *index, const float *src, const int *count, float *out){
  // index: (L) src: (L, C)  count: (N) out: (N, C)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= L * C) return;

  int pid = tid / C;
  int cid = tid % C;

  index += pid;
  src += tid;
  int odx = index[0];
  count += odx;

  atomAdd(out + odx * C + cid, src[0] / count[0]);
}

__global__ void scatter_avg_grad_kernel(int L, int C, const int *index, const int *count, const float *grad_out, float *grad_src){
  // index: (L,) count: (N,)  grad_out: (N, C)  grad_src: (L, C)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= L * C) return;

  int pid = tid / C;
  int cid = tid % C;

  index += pid;
  grad_src += tid;

  int odx = index[0];
  count += odx;
  grad_out += odx * C + cid;

  atomAdd(grad_src, grad_out[0] / count[0]);
}

void scatter_max_kernel_launcher(int L, int C, const int *index, const float *src, int *arg, float *out){
  cudaError_t err;
  dim3 blocks(DIVUP(L * C, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  scatter_max_kernel<<<blocks, threads>>>(L, C, index, src, out);
  scatter_arg_max_kernel<<<blocks, threads>>>(L, C, index, src, out, arg);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void scatter_max_grad_kernel_launcher(int N, int C, const int *arg, const float *grad_out, float *grad_src){
  cudaError_t err;
  dim3 blocks(DIVUP(N * C, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  scatter_max_grad_kernel<<<blocks, threads>>>(N, C, arg, grad_out, grad_src);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void scatter_avg_kernel_launcher(int L, int N, int C, const int *index, const float *src, int *count, float *out){
  cudaError_t err;
  dim3 threads(THREADS_PER_BLOCK);

  dim3 blocks1(DIVUP(L, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  scatter_count_kernel<<<blocks1, threads>>>(L, index, count);

  dim3 blocks2(DIVUP(L * C, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  scatter_avg_kernel<<<blocks2, threads>>>(L, C, index, src, count, out);

  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void scatter_avg_grad_kernel_launcher(int L, int C, const int *index, const int *count, const float *grad_out, float *grad_src){
  cudaError_t err;
  dim3 threads(THREADS_PER_BLOCK);

  dim3 blocks(DIVUP(L * C, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  scatter_avg_grad_kernel<<<blocks, threads>>>(L, C, index, count, grad_out, grad_src);

  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}