//
// Created by sgs on 2021/2/5.
//
#include "sample_group_gpu.h"

#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define TOTAL_THREADS 1024

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

  return max(min(1 << pow_2, TOTAL_THREADS), 1);
}


__global__ void get_sampled_points_mask(int N, int M, int CP, int CR, float extra, const float* rois,
										const float* points, bool* points_mask){

	// param rois: (K, 7 + C) [x y z h w l ry] or [x y z w l h ry]
	// param points: (N, 3) [x y z] sparse tensor attr. raw value in lidar
	// return points_mask: (N)  0 - not exist, 1 - exist

	int rid = blockIdx.y;
	int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (rid >= M || pid >= N) return;

	rois += rid * CR;
	float cx = rois[0], cy = rois[1], cz = rois[2];
	float rh = rois[3], rw = rois[4], rl = rois[5];

	points += pid * CP;

	float radius = sqrtf(rh * rh + rw * rw + rl * rl) / 2. + extra;
	float dist2 = (points[0] - cx) * (points[0] - cx) + (points[1] - cy) * (points[1] - cy) + (points[2] - cz) * (points[2] - cz);
	if (dist2 < radius * radius) {
		points_mask[pid] = 1;
	}
}

void points_in_boxes_kernel_launcher(int N, int M, int CP, int CR, float extra,
									 const float *rois, const float *points, bool *points_mask){
	dim3 threads(THREADS_PER_BLOCK);
	dim3 blocks(DIVUP(N, THREADS_PER_BLOCK), M);

	get_sampled_points_mask<<<blocks, threads>>>(N, M, CP, CR, extra, rois, points, points_mask);
}

__global__ void voxel_indice_kernel(int N, int C, float voxel, float lx, float ly, float lz, float hx, float hy, float hz,
									const float *points, long *voxel_indice){
	// points: (N, C) x y z ...
	// voxel_indice: (N)

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) return;

	points += tid * C;

	if (points[0] < lx || points[0] > hx || points[1] < ly || points[1] > hy || points[2] < lz || points[2] > hz) return;

	int VX = ceil((hx - lx) / voxel);
	int VY = ceil((hy - ly) / voxel);
//	int VZ = ceil((hz - lz) / voxel);

	int xid = (points[0] - lx) / voxel;
	int yid = (points[1] - ly) / voxel;
	int zid = (points[2] - lz) / voxel;

	voxel_indice[tid] = zid * VY * VX + yid * VX + xid;
}

__global__ void voxel_indice_v1_kernel(int N, int C, float voxel, int VX, int VY, int VZ,
									   const float *points, long *voxel_indice){
	// points: (N, C) x y z ...
	// voxel_indice: (N)

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) return;

	points += tid * C;

	int xid = points[0] / voxel;
	int yid = points[1] / voxel;
	int zid = points[2] / voxel;

	if (xid < 0 || xid >= VX || yid < 0 || yid >= VY || zid < 0 || zid >= VZ) return;

	voxel_indice[tid] = zid * VY * VX + yid * VX + xid;
}

void voxel_indice_kernel_launcher(int N, int C, float voxel, float lx, float ly, float lz, float hx, float hy, float hz,
								  const float *points, long *voxel_indice) {
	cudaError_t err;
	dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);

	voxel_indice_kernel<<<blocks, threads>>>(N, C, voxel, lx, ly, lz, hx, hy, hz, points, voxel_indice);
	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

void voxel_indice_v1_kernel_launcher(int N, int C, float voxel, int VX, int VY, int VZ,
								     const float *points, long *voxel_indice) {
	cudaError_t err;
	dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);

	voxel_indice_v1_kernel<<<blocks, threads>>>(N, C, voxel, VX, VY, VZ, points, voxel_indice);
	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

__global__ void average_points_kernel(int N, int C, int max_points, const long *sort_voxel_indice, float *points, bool *mask){
	// sort_voxel_indice: (N), sort_index: (N)
	// points: (N, 3+C)  mask: (N,)

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) return;

	const int index = sort_voxel_indice[tid];
	if (index == -1) return;

    const int max_tid = min(tid + max_points, N);

	if (tid == 0 || index != sort_voxel_indice[tid-1]){
		mask[tid] = 1;

        int cid = tid + 1;
        while (cid < max_tid) {
          if (index != sort_voxel_indice[cid]) break;

          for (int c = 0; c < C; ++c){
		    points[tid * C + c] += points[cid * C + c];
		  }
		  cid += 1;
        }

        if (cid == tid + 1) return;
		for (int c = 0; c < C; ++c){
		  points[tid * C + c] = points[tid * C + c] / (cid - tid);
		}
	}
}

__global__ void unique_points_kernel(int N, const long *sort_voxel_indice, const long *sort_index, bool *mask){
	// sort_voxel_indice: (N), sort_index: (N)
	// points: (N, 3+C)  mask: (N,)

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) return;

	const int index = sort_voxel_indice[tid];
	if (index == -1) return;

	if (tid == 0 || index != sort_voxel_indice[tid-1]){
		const long idx = sort_index[tid];
		mask[idx] = 1;
	}
}

void average_points_kernel_launcher(int N, int C, int max_points, const long *sort_voxel_indice, float *points, bool *mask){
	cudaError_t err;
	dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);

	average_points_kernel<<<blocks, threads>>>(N, C, max_points, sort_voxel_indice, points, mask);
	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

void unique_points_kernel_launcher(int N, const long *sort_voxel_indice, const long *sort_index, bool *mask){
	cudaError_t err;
	dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);

	unique_points_kernel<<<blocks, threads>>>(N, sort_voxel_indice, sort_index, mask);
	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, int idx1, int idx2){
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}


template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(int N, int M, int C,
  const float *__restrict__ dataset, float *__restrict__ temp, int *__restrict__ idxs) {
  // dataset: (B, N, 3)
  // tmp: (B, N)
  // output:
  //      idx: (B, M)

  if (M <= 0) return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * N * C;
  temp += batch_index * N;
  idxs += batch_index * M;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0)
    idxs[0] = old;

  __syncthreads();
  for (int j = 1; j < M; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * C + 0];
    float y1 = dataset[old * C + 1];
    float z1 = dataset[old * C + 2];
    for (int k = tid; k < N; k += stride) {
      float x2, y2, z2;
      x2 = dataset[k * C + 0];
      y2 = dataset[k * C + 1];
      z2 = dataset[k * C + 2];
      // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      // if (mag <= 1e-3)
      // continue;

      float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
      if (tid < 512) {
        __update(dists, dists_i, tid, tid + 512);
      }
      __syncthreads();
    }

    if (block_size >= 512) {
      if (tid < 256) {
        __update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
      idxs[j] = old;
  }
}

void furthest_point_sampling_kernel_launcher(int B, int N, int M, int C,
  const float *dataset, float *temp, int *idxs) {
  // dataset: (B, N, C)
  // tmp: (B, N)
  // output:
  //      idx: (B, M)

  cudaError_t err;
  unsigned int n_threads = opt_n_threads(N);

  switch (n_threads) {
    case 1024:
      furthest_point_sampling_kernel<1024><<<B, n_threads>>>(N, M, C, dataset, temp, idxs); break;
    case 512:
      furthest_point_sampling_kernel<512><<<B, n_threads>>>(N, M, C, dataset, temp, idxs); break;
    case 256:
      furthest_point_sampling_kernel<256><<<B, n_threads>>>(N, M, C, dataset, temp, idxs); break;
    case 128:
      furthest_point_sampling_kernel<128><<<B, n_threads>>>(N, M, C, dataset, temp, idxs); break;
    case 64:
      furthest_point_sampling_kernel<64><<<B, n_threads>>>(N, M, C, dataset, temp, idxs); break;
    case 32:
      furthest_point_sampling_kernel<32><<<B, n_threads>>>(N, M, C, dataset, temp, idxs); break;
    case 16:
      furthest_point_sampling_kernel<16><<<B, n_threads>>>(N, M, C, dataset, temp, idxs); break;
    case 8:
      furthest_point_sampling_kernel<8><<<B, n_threads>>>(N, M, C, dataset, temp, idxs); break;
    case 4:
      furthest_point_sampling_kernel<4><<<B, n_threads>>>(N, M, C, dataset, temp, idxs); break;
    case 2:
      furthest_point_sampling_kernel<2><<<B, n_threads>>>(N, M, C, dataset, temp, idxs); break;
    case 1:
      furthest_point_sampling_kernel<1><<<B, n_threads>>>(N, M, C, dataset, temp, idxs); break;
    default:
      furthest_point_sampling_kernel<512><<<B, n_threads>>>(N, M, C, dataset, temp, idxs);
  }

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void gather_points_kernel(int M, int C, const int *idx, const float *points, float *out) {
    // idx: (M)  points: (N, C)
    // out: (M, C)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    out += tid * C;
    idx += tid;
    points += idx[0] * C;

    for (int c = 0; c < C; ++c){
      out[c] = points[c];
    }
}

void gather_points_kernel_launcher(int M, int C, const int *idx, const float *points, float *out) {
    cudaError_t err;
    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_kernel<<<blocks, threads>>>(M, C, idx, points, out);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void gather_points_grad_kernel(int M, int C, const int *idx, const float *grad_out, float *grad_points) {
    // idx: (M)  grad_out: (M, C)
    // grad_points: (N, C)

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    grad_out += tid * C;
    idx += tid;
    grad_points += idx[0] * C;
    for (int c = 0; c < C; ++c){
      atomicAdd(grad_points + c, grad_out[c]);
    }
}

void gather_points_grad_kernel_launcher(int M, int C, const int *idx, const float *grad_out, float *grad_points) {

    cudaError_t err;
    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_grad_kernel<<<blocks, threads>>>(M, C, idx, grad_out, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void group_points_grad_kernel_stack(int B, int M, int C, int N, int nsample,
                                               const float *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, float *grad_features) {
  // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
  // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
  // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
  // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
  // :return:
  //     grad_features: (N1 + N2 ..., C) gradient of the features
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int sample_idx = index % nsample;
  int C_idx = (index / nsample) % C;
  int pt_idx = (index / nsample / C);

  if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;

  int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
  for (int k = 1; k < B; k++){
    if (pt_idx < pt_cnt) break;
    pt_cnt += idx_batch_cnt[k];
    bs_idx = k;
  }

  int features_batch_start_idx = 0;
  for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k];

  grad_out += pt_idx * C * nsample + C_idx * nsample + sample_idx;
  idx += pt_idx * nsample + sample_idx;
  grad_features += (features_batch_start_idx + idx[0]) * C + C_idx;

  atomicAdd(grad_features, grad_out[0]);
}

void group_points_grad_kernel_launcher_stack(int B, int M, int C, int N, int nsample,
                                             const float *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, float *grad_features) {
  // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
  // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
  // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
  // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
  // :return:
  //     grad_features: (N1 + N2 ..., C) gradient of the features

  cudaError_t err;
  // dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  group_points_grad_kernel_stack<<<blocks, threads>>>(B, M, C, N, nsample, grad_out, idx, idx_batch_cnt, features_batch_cnt, grad_features);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void group_points_kernel_stack(int B, int M, int C, int nsample,
                                          const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out) {
  // :param features: (N1 + N2 ..., C) tensor of features to group
  // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
  // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
  // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
  // :return:
  //     output: (M1 + M2, C, nsample) tensor
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int sample_idx = index % nsample;
  int C_idx = (index / nsample) % C;
  int pt_idx = (index / nsample / C);

  if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;

  int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
  for (int k = 1; k < B; k++){
    if (pt_idx < pt_cnt) break;
    pt_cnt += idx_batch_cnt[k];
    bs_idx = k;
  }

  int features_batch_start_idx = 0;
  for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k];
  features += features_batch_start_idx * C;

  idx += pt_idx * nsample + sample_idx;
  int in_idx = idx[0] * C + C_idx;
  int out_idx = pt_idx * C * nsample + C_idx * nsample + sample_idx;

  out[out_idx] = features[in_idx];
}


void group_points_kernel_launcher_stack(int B, int M, int C, int nsample,
                                        const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out) {
  // :param features: (N1 + N2 ..., C) tensor of features to group
  // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
  // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
  // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
  // :return:
  //     output: (M1 + M2, C, nsample) tensor

  cudaError_t err;
  dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  group_points_kernel_stack<<<blocks, threads>>>(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, out);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}