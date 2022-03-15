//
// Created by sgs on 2021/7/13.
//

#include "atomics.cuh"
#include "query_ops_gpu.h"


#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__global__ void createPillarIndicesStackKernel(int N, int B, int H, int W, float bev_size,
											   const float *xyz, const int *xyz_batch_cnt, bool *pillar_mask) {
	// xyz_batch_cnt: (N1+N2...)    xyz: (N1+N2..., 3) [x y z ...]
    // pillar_mask: [B, H, W]
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    xyz += tid * 3;

    int bid = 0, pts_cnt = xyz_batch_cnt[0];
    for (int b = 1; b < B; ++b){
    	if (tid < pts_cnt) break;
    	pts_cnt += xyz_batch_cnt[b];
    	bid = b;
    }

    pillar_mask += bid * H * W;

    float cx = xyz[0] / bev_size;
    float cy = xyz[1] / bev_size;
    int xid = int(cx);
    int yid = int(cy);

    if (xid < 0 || xid >= W || yid < 0 || yid >= H) return;
    pillar_mask[yid * W + xid] = 1;
}


__global__ void createPillarIndicePairsStackKernel(int N, int B, int H, int W, int K, float radius, float bev_size,
												   const float *xyz, const int *xyz_batch_cnt,
                                                   bool *bev_mask, int *indice_pairs) {
	// xyz_batch_cnt: (N1+N2...)    xyz: (N1+N2..., C) [x y z ...]
    // bev_mask: [B, H, W]          indice_pairs: (N1+N2..., K)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    xyz += tid * 3;
    indice_pairs += tid * K;

    int bid = 0, pts_cnt = xyz_batch_cnt[0];
    for (int b = 1; b < B; ++b){
    	if (tid < pts_cnt) break;
    	pts_cnt += xyz_batch_cnt[b];
    	bid = b;
    }

    const int prefix = bid * H * W;
    bev_mask += prefix;

    float cx = xyz[0] / bev_size;
    float cy = xyz[1] / bev_size;
    int xid = int(cx);
    int yid = int(cy);

    int range = round(radius / bev_size) + 1;
    float scale_radius2 = (radius / bev_size) * (radius / bev_size);

    int cnt = 0, index;
    for (int m = 1-range; m < range; ++m){
        for (int n = 1-range; n < range; ++n){
            if (cnt >= K) break;

            int px = xid + m;
            int py = yid + n;
            if (px < 0 || px >= W || py < 0 || py >= H) continue;

            float distance = (px + 0.5 - cx) * (px + 0.5 - cx) + (py + 0.5 - cy) * (py + 0.5 - cy);
            if (distance < scale_radius2) {
                index = py * W + px;
                bev_mask[index] = 1;
                indice_pairs[cnt] = index + prefix;
                ++cnt;
            }
        }
    }
}


__global__ void createPillarIndicePairsV2StackKernel(int N, int B, int H, int W, int K, float radius, float bev_size,
													 const float *xyz, const int *xyz_batch_cnt,
                                                     const int *pillar_bev_indices, int *indice_pairs) {
	// xyz_batch_cnt: (N1+N2...)    xyz: (N1+N2..., C) [x y z ...]    pillar_bev_indices: [B, H, W]
	// indice_pairs: (N1+N2..., K)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    xyz += tid * 3;
    indice_pairs += tid * K;

    int bid = 0, pts_cnt = xyz_batch_cnt[0];
    for (int b = 1; b < B; ++b){
    	if (tid < pts_cnt) break;
    	pts_cnt += xyz_batch_cnt[b];
    	bid = b;
    }

    const int prefix = bid * H * W;
    pillar_bev_indices += prefix;

    const float cx = xyz[0] / bev_size;
    const float cy = xyz[1] / bev_size;
    int xid = int(cx);
    int yid = int(cy);

    int range = round(radius / bev_size) + 1;
    const float sradius2 = radius * radius / (bev_size * bev_size);

    int cnt = 0, index;
    for (int m = 1-range; m < range; ++m){
        for (int n = 1-range; n < range; ++n){
            if (cnt >= K) break;

            int px = xid + m;
            int py = yid + n;
            if (px < 0 || px >= W || py < 0 || py >= H) continue;

            index = pillar_bev_indices[py * W + px];
            if (index < 0) continue;

            float sdistance2 = (px + 0.5 - cx) * (px + 0.5 - cx) + (py + 0.5 - cy) * (py + 0.5 - cy);
            if (sdistance2 < sradius2) {
            	indice_pairs[cnt] = index;
                ++cnt;
            }
        }
    }
}

__global__ void createPillarIndicePairsV2aStackKernel(int N, int B, int H, int W, float bev_size,
													  const float *xyz, const int *xyz_batch_cnt,
                                                      const int *pillar_bev_indices, int *indice_pairs) {
    // pointBatchCnt: (N1+N2...)    points: (N1+N2..., C) [x y z ...]    pillarBevIndices: [B, H, W]
    // indicePairs: (N1+N2..., 1)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    xyz += tid * 3;
    indice_pairs += tid;

    int bid = 0, pts_cnt = xyz_batch_cnt[0];
    for (int b = 1; b < B; ++b){
    	if (tid < pts_cnt) break;
    	pts_cnt += xyz_batch_cnt[b];
    	bid = b;
    }

    const int prefix = bid * H * W;
    pillar_bev_indices += prefix;

    const float cx = xyz[0] / bev_size;
    const float cy = xyz[1] / bev_size;
    int xid = int(cx);
    int yid = int(cy);

    if (xid < 0 || xid >= W || yid < 0 || yid >= H) return;
    indice_pairs[0] = pillar_bev_indices[yid * W + xid];
}

void create_pillar_indices_stack_kernel_launcher(int N, int B, int H, int W, float bev_size,
												 const float *xyz, const int *xyz_batch_cnt, bool *pillar_mask) {
    cudaError_t err;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    createPillarIndicesStackKernel<<<blocks, threads>>>(N, B, H, W, bev_size, xyz, xyz_batch_cnt, pillar_mask);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


void create_pillar_indice_pairs_stack_kernel_launcher(int N, int B, int H, int W, int K, float radius, float bev_size,
													  const float *xyz, const int *xyz_batch_cnt,
                                                      bool *bev_mask, int *indice_pairs) {
    cudaError_t err;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    createPillarIndicePairsStackKernel<<<blocks, threads>>>(N, B, H, W, K, radius, bev_size, xyz, xyz_batch_cnt,
															bev_mask, indice_pairs);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void create_pillar_indice_pairsv2_stack_kernel_launcher(int N, int B, int H, int W, int K, float radius, float bev_size,
														const float *xyz, const int *xyz_batch_cnt,
                                                        const int *pillar_bev_indices, int *indice_pairs) {
    cudaError_t err;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    createPillarIndicePairsV2StackKernel<<<blocks, threads>>>(N, B, H, W, K, radius, bev_size, xyz, xyz_batch_cnt,
															  pillar_bev_indices, indice_pairs);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void create_pillar_indice_pairsv2a_stack_kernel_launcher(int N, int B, int H, int W, float bev_size,
														 const float *xyz, const int *xyz_batch_cnt,
														 const int *pillar_bev_indices, int *indice_pairs) {
    cudaError_t err;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    createPillarIndicePairsV2aStackKernel<<<blocks, threads>>>(N, B, H, W, bev_size, xyz, xyz_batch_cnt,
															   pillar_bev_indices, indice_pairs);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void createBevIndicesKernel(int M, int H, int W, const int *pillarIndices, int *bevIndices){
	// pillarIndices: (L, 3)  [bId, yId, xId]   bevIndices: (B*H*W)

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    pillarIndices += tid * 3;

    int bId = pillarIndices[0];
    int yId = pillarIndices[1];
    int xId = pillarIndices[2];

    const int index = bId * H * W + yId * W + xId;
    bevIndices[index] = tid;
}

__global__ void createPillarIndicesKernel(int B, int H, int W, const int *bevIndices, int *pillarIndices){
	// pillarIndices: (L, 3)  [bId, yId, xId]   bevIndices: (B*H*W)

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B*H*W) return;

	bevIndices += tid;
	if (bevIndices[0] < 0) return;

	int bId = tid / (H * W);
	int xId = tid % W;
	int yId = (tid / W) % H;

	const int index = bevIndices[0] * 3;
	pillarIndices[index + 0] = bId;
	pillarIndices[index + 1] = yId;
	pillarIndices[index + 2] = xId;
}

void create_bev_indices_kernel_launcher(int M, int H, int W, const int *pillarIndices, int *bevIndices){
    cudaError_t err;
    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    createBevIndicesKernel<<<blocks, threads>>>(M, H, W, pillarIndices, bevIndices);

    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void create_pillar_indices_kernel_launcher(int B, int H, int W, const int *bevIndices, int *pillarIndices){
	cudaError_t err;
	dim3 blocks(DIVUP(B * H * W, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);

	createPillarIndicesKernel<<<blocks, threads>>>(B, H, W, bevIndices, pillarIndices);

	// cudaDeviceSynchronize();  // for using printf in kernel function
	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

__global__ void updateIndicePairsKernel(int N, int K, const int *location, int *indicePairs){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * K) return;

    indicePairs += tid;
    if (indicePairs[0] < 0) return;

    indicePairs[0] = location[indicePairs[0]];
}

__global__ void CreateIndice2BevKernel(int BHW, const int *location, int *indice2bev){
    // location: (B, H, W)    indice2bev: (M)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= BHW) return;

    location += tid;
    if (location[0] < 0) return;

    indice2bev[location[0]] = tid;
}

void update_indice_pairs_kernel_launcher(int N, int K, const int *location, int *indicePairs){
    cudaError_t err;
    dim3 blocks(DIVUP(N * K, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    updateIndicePairsKernel<<<blocks, threads>>>(N, K, location, indicePairs);

    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void create_indice2bev_kernel_launcher(int BHW, const int *location, int *indice2bev){
    cudaError_t err;
    dim3 blocks(DIVUP(BHW, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    CreateIndice2BevKernel<<<blocks, threads>>>(BHW, location, indice2bev);

    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}