/*
Stacked-batch-data version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_gpu.h"

#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__global__ void ball_query_kernel_stack(int B, int M, int G, int N, int nsample, float radius, \
  const float *xyz, const int *xyz_batch_cnt, const float *new_xyz, const float *ball, int *idx) {
  // two steps to query neighbor grid points: 1. filter by roi center and diameter of bounding ball; 2. Euclidean distance
  // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
  // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
  // :param new_xyz: (B, M, G, 3) centers of the ball query
  // :param ball: (B, M, 4), [x y z diameter]
  // output:
  //      idx: (N1 + N2 ...,, nsample)

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  int bid = 0, pt_cnt = xyz_batch_cnt[0];
  for (int k = 1; k < B; k++){
      if (tid < pt_cnt) break;
      pt_cnt += xyz_batch_cnt[k];
      bid = k;
  }

  xyz += tid * 3;
  idx += tid * nsample;
  new_xyz += bid * M * G *3;
  ball += bid * M * 4;

  float px = xyz[0];
  float py = xyz[1];
  float pz = xyz[2];

  float radius2 = radius * radius;
  const int base = bid * M * G;

  int cnt = 0;
  for (int m = 0; m < M; ++m){
      if (cnt >= nsample) break;
      float dx = px - ball[m * 4 + 0];
      float dy = py - ball[m * 4 + 1];
      float dz = pz - ball[m * 4 + 2];
      float dist = ball[m * 4 + 3] / 2 + radius;
      if (dx * dx + dy * dy + dz * dz < dist * dist) {

          for (int g = 0; g < G; ++g){

              float dx = px - new_xyz[(m * G + g) * 3 + 0];
              float dy = py - new_xyz[(m * G + g) * 3 + 1];
              float dz = pz - new_xyz[(m * G + g) * 3 + 2];

              if (dx * dx + dy * dy + dz * dz < radius2){
                  idx[cnt] = base + m * G + g;
                  ++cnt;
                  if (cnt >= nsample) break;
             }
          }
      }
  }
}


void ball_query_kernel_launcher_stack(int B, int M, int G, int N, int nsample, float radius,
  const float *xyz, const int *xyz_batch_cnt, const float *new_xyz, const float *ball, int *idx){

  cudaError_t err;

  dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  ball_query_kernel_stack<<<blocks, threads>>>(B, M, G, N, nsample, radius, xyz, xyz_batch_cnt, new_xyz, ball, idx);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
