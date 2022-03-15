//
// Created by sgs on 2021/7/20.
//

#ifndef PILLAR_INDICE_CUH
#define PILLAR_INDICE_CUH

#include <cuda.h>
#include <cuda_runtime_api.h>


__global__ void assignGridIn2DKernel(int M, int H, int W, const int *indices, int *gridIndice) {
    // indices: (M, 3) [byx]  gridIndice: (B*H*W)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    indices += tid * 3;
    const int index = indices[0] * H * W + indices[1] * W + indices[2];
    gridIndice[index] = tid;
}

__global__ void assignGridIn3DKernel(int M, int D, int H, int W, const int *indices, int *gridIndice) {
    // indices: (M, 4) [bzyx]  gridIndice: (B*D*H*W)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    indices += tid * 4;
    const int index = indices[0] * D * H * W + indices[1] * H * W + indices[2] * W + indices[3];
    gridIndice[index] = tid;
}

__global__ void assignGridInterpIndice2DKernel(int N, int B, int H, int W, int K, int kernel, float by, float bx,
                                               const int *gridIndice, const int *xyzBatchCnt, const float *xyz,
                                               int *indiceKernel, float *inverseDist) {
    // gridIndice: (B*H*W)   xyzBatchCnt: (batch_size) [N1, N2, ...]   xyz: (N1+N2..., 3) [xyz]
    // indiceKernel: (N, K)  inverseDist: (N, K)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    xyz += tid * 3;
    indiceKernel += tid * K;
    inverseDist += tid * K;

    int bId = 0, pointCnt = xyzBatchCnt[0];
    for (int b = 1; b < B; ++b){
        if (tid < pointCnt) break;
        pointCnt += xyzBatchCnt[b];
        bId = b;
    }

    gridIndice += bId * H * W;

    float cy = xyz[1] / by;
    float cx = xyz[0] / bx;
    int xId = int(cx);
    int yId = int(cy);

    int cnt = 0;
    float dist, sumDist = 0;
    for (int m = -kernel; m <= kernel; ++m){
        for (int n = -kernel; n <= kernel; ++n){
            if (cnt >= K) break;
            if (abs(m) + abs(n) > kernel) continue;

            int py = yId + m;
            int px = xId + n;

            if (py < 0 || py >= H || px < 0 || px >= W) continue;
            int index = gridIndice[py * W + px];
            if (index < 0) continue;

            indiceKernel[cnt] = index;
            dist = 1. / (1e-8 + sqrtf((py + 0.5 - cy) * (py + 0.5 - cy) + \
                                      (px + 0.5 - cx) * (px + 0.5 - cx)));
            inverseDist[cnt] = dist;
            sumDist += dist;
            cnt++;
        }
    }

    for (int j = 0; j < cnt; ++j)
        inverseDist[j] /= sumDist;
}


__global__ void assignGridInterpIndice3DKernel(int N, int B, int D, int H, int W, int K, int kernel, float vz, float vy, float vx,
                                               const int *gridIndice, const int *xyzBatchCnt, const float *xyz,
                                               int *indiceKernel, float *inverseDist) {
    // gridIndice: (B*D*H*W)    xyzBatchCnt: (batch_size) [N1, N2, ...]    xyz: (N1+N2..., 3) [xyz]
    // indiceKernel: (N, K)     inverseDist: (N, K)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    xyz += tid * 3;
    indiceKernel += tid * K;
    inverseDist += tid * K;

    int bId = 0, pointCnt = xyzBatchCnt[0];
    for (int b = 1; b < B; ++b){
        if (tid < pointCnt) break;
        pointCnt += xyzBatchCnt[b];
        bId = b;
    }

    gridIndice += bId * D * H * W;

    float cz = xyz[2] / vz;
    float cy = xyz[1] / vy;
    float cx = xyz[0] / vx;
    int xId = int(cx);
    int yId = int(cy);
    int zId = int(cz);

    int cnt = 0;
    float dist, sumDist = 0;
    for (int l = -kernel; l <= kernel; ++l){
        for (int m = -kernel; m <= kernel; ++m){
            for (int n = -kernel; n <= kernel; ++n){
                if (cnt >= K) break;
                if (abs(l) + abs(m) + abs(n) > kernel) continue;

                int pz = zId + l;
                int py = yId + m;
                int px = xId + n;

                if (pz < 0 || pz >= D || py < 0 || py >= H || px < 0 || px >= W) continue;
                int index = gridIndice[pz * H * W + py * W + px];
                if (index < 0) continue;

                indiceKernel[cnt] = index;
                dist = norm3df(pz + 0.5 - cz,
                               py + 0.5 - cy,
                               px + 0.5 - cx);
                dist = 1. / (dist + 1e-8);
                inverseDist[cnt] = dist;
                sumDist += dist;
                cnt++;
            }
        }
    }

    for (int j = 0; j < cnt; ++j)
        inverseDist[j] /= sumDist;
}



#endif //PILLAR_INDICE_CUH
