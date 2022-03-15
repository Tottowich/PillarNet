#include "ball_group_gpu.h"


int ball_indice_wrapper_stack(at::Tensor ball_idx_tensor, at::Tensor out_offsets_tensor,
    at::Tensor set_indices_tensor, at::Tensor set_new_indices_tensor) {

  CHECK_INPUT(ball_idx_tensor);
  CHECK_INPUT(out_offsets_tensor);
  CHECK_INPUT(set_indices_tensor);
  CHECK_INPUT(set_new_indices_tensor);

  int N = ball_idx_tensor.size(0);
  int nsample = ball_idx_tensor.size(1);

  const int *ball_idx = ball_idx_tensor.data<int>();
  const int *out_offsets = out_offsets_tensor.data<int>();
  int *set_indices = set_indices_tensor.data<int>();
  int *set_new_indices = set_new_indices_tensor.data<int>();

  ball_indice_kernel_launcher_stack(N, nsample, ball_idx, out_offsets, set_indices, set_new_indices);
  return 1;
}


int gather_feature_wrapper(at::Tensor set_indices_tensor, at::Tensor features_tensor, at::Tensor new_features_tensor){

  CHECK_INPUT(set_indices_tensor);
  CHECK_INPUT(features_tensor);
  CHECK_INPUT(new_features_tensor);

  int nindices = set_indices_tensor.size(0);
  int C = features_tensor.size(1);

  const int *set_indices = set_indices_tensor.data<int>();
  const float *features = features_tensor.data<float>();
  float *new_features = new_features_tensor.data<float>();

  gather_feature_kernel_launcher(nindices, C, set_indices, features, new_features);

  return 1;
}

int gather_feature_grad_wrapper(at::Tensor set_indices_tensor, at::Tensor grad_out_tensor, at::Tensor grad_features_tensor){

  CHECK_INPUT(set_indices_tensor);
  CHECK_INPUT(grad_out_tensor);
  CHECK_INPUT(grad_features_tensor);

  int nindices = set_indices_tensor.size(0);
  int C = grad_features_tensor.size(1);

  const int *set_indices = set_indices_tensor.data<int>();
  const float *outGrad = grad_out_tensor.data<float>();
  float *inGrad = grad_features_tensor.data<float>();

  gather_feature_grad_kernel_launcher(nindices, C, set_indices, outGrad, inGrad);

  return 1;
}





