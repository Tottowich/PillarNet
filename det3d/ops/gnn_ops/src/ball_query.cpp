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


int ball_query_wrapper_stack(float radius, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
    at::Tensor new_xyz_tensor, at::Tensor ball_tensor, at::Tensor idx_tensor) {

    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(ball_tensor);
    CHECK_INPUT(idx_tensor);

    int B = new_xyz_tensor.size(0);
    int M = new_xyz_tensor.size(1);
    int G = new_xyz_tensor.size(2);
    int N = idx_tensor.size(0);
    int nsample = idx_tensor.size(1);

    const float *xyz = xyz_tensor.data<float>();
    const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *ball = ball_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    ball_query_kernel_launcher_stack(B, M, G, N, nsample, radius, xyz, xyz_batch_cnt, new_xyz, ball, idx);
    return 1;
}

