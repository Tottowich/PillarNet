//
// Created by sgs on 2021/7/18.
//

#include "indice.cuh"
#include "interpolate_ops_gpu.h"

#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


void indice_interp2d_kernel_launcher(int M, int N, int B, int H, int W, int K, int kernel, float by, float bx,
                                     const int *indices, int *gridIndice, const int *xyzBatchCnt, const float *xyz,
                                     int *indiceKernel, float *inverseDist) {
    cudaError_t err;
    if (N == 0) return;

    dim3 blocks1(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    assignGridIn2DKernel<<<blocks1, threads>>>(M, H, W, indices, gridIndice);

    dim3 blocks2(DIVUP(N, THREADS_PER_BLOCK));
    assignGridInterpIndice2DKernel<<<blocks2, threads>>>(N, B, H, W, K, kernel, by, bx, gridIndice,
                                                         xyzBatchCnt, xyz, indiceKernel, inverseDist);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void indice_interp3d_kernel_launcher(int M, int N, int B, int D, int H, int W, int K, int kernel, float vz, float vy, float vx,
                                     const int *indices, const int *xyzBatchCnt, const float *xyz,
                                     int *gridIndice, int *indiceKernel, float *inverseDist) {
    cudaError_t err;
    if (N == 0) return;

    dim3 blocks1(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    assignGridIn3DKernel<<<blocks1, threads>>>(M, D, H, W, indices, gridIndice);

    dim3 blocks2(DIVUP(N, THREADS_PER_BLOCK));
    assignGridInterpIndice3DKernel<<<blocks2, threads>>>(N, B, D, H, W, K, kernel, vz, vy, vx,
                                                         gridIndice, xyzBatchCnt, xyz, indiceKernel, inverseDist);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void interpFwdKernel(int N, int K, int C, const float *features,
                                const int *indiceKernel, const float *inverseDist, float *out) {
    // indiceKernel: (N, K)  inverseDist: (N, K)  features: (M, C)  --> out: (N, C)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N*C) return;

    int oid = tid / C;
    int cid = tid % C;

    indiceKernel += oid * K;
    inverseDist += oid * K;

    float weightSum = 0;
    for (int i = 0; i < K; ++i) {
        int index = indiceKernel[i];
        if (index < 0) break;
        weightSum += inverseDist[i] * features[index * C + cid];
    }
    out[tid] = weightSum;
}

__global__ void interpBwdKernel(int N, int K, int C, const float *outGrad,
                                const int *indiceKernel, const float *inverseDist, float *inGrad){
    // indiceKernel: (N, K)  inverseDist: (N, K)  grad_out: (N, C) --> grad_in: (M, C)

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N*C) return;

    int oid = tid / C;
    int cid = tid % C;

    indiceKernel += oid * K;
    inverseDist += oid * K;

    for (int i = 0; i < K; ++i) {
        if (indiceKernel[i] < 0) break;
        atomicAdd(inGrad + indiceKernel[i] * C + cid, outGrad[tid] * inverseDist[i]);
    }
}

void interp_fwd_kernel_launcher(int N, int K, int C, const float *features,
                                const int *indiceKernel, const float *inverseDist, float *out) {
    cudaError_t err;
    if (N == 0) return;

    dim3 blocks(DIVUP(N*C, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    interpFwdKernel<<<blocks, threads>>>(N, K, C, features, indiceKernel, inverseDist, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed (interp_fwd_kernel) : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void interp_bwd_kernel_launcher(int N, int K, int C, const float *outGrad,
                                const int *indiceKernel, const float *inverseDist, float *inGrad) {
    cudaError_t err;
    if (N == 0) return;

    dim3 blocks(DIVUP(N*C, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    interpBwdKernel<<<blocks, threads>>>(N, K, C, outGrad, indiceKernel, inverseDist, inGrad);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed (interp_bwd_kernel) : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}