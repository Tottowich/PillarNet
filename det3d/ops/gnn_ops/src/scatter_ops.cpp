//
// Created by sgs on 2021/2/9.
//

#include "scatter_ops_gpu.h"


int scatter_max_wrapper(at::Tensor src_tensor, at::Tensor index_tensor, at::Tensor arg_tensor, at::Tensor out_tensor){
  // src: (L, C)   indices: (L)  arg: (N, C)  out: (N, C)
  CHECK_INPUT(src_tensor);
  CHECK_INPUT(index_tensor);
  CHECK_INPUT(arg_tensor);
  CHECK_INPUT(out_tensor);

  int L = src_tensor.size(0);
  int C = src_tensor.size(1);
//  int N = out_tensor.size(0);

  const float *src = src_tensor.data<float>();
  const int *index = index_tensor.data<int>();
  int * arg = arg_tensor.data<int>();
  float *out = out_tensor.data<float>();

  scatter_max_kernel_launcher(L, C, index, src, arg, out);
  return 1;
}

int scatter_max_grad_wrapper(at::Tensor arg_tensor, at::Tensor grad_out_tensor, at::Tensor grad_src_tensor){
  // arg: (N, C)   grad_out: (N, C)  grad_src: (L, C)
  CHECK_INPUT(arg_tensor);
  CHECK_INPUT(grad_out_tensor);
  CHECK_INPUT(grad_src_tensor);

  int N = grad_out_tensor.size(0);
  int C = grad_out_tensor.size(1);

  const int *arg = arg_tensor.data<int>();
  const float *grad_out = grad_out_tensor.data<float>();
  float *grad_src = grad_src_tensor.data<float>();

  scatter_max_grad_kernel_launcher(N, C, arg, grad_out, grad_src);
  return 1;
}

int scatter_avg_wrapper(at::Tensor src_tensor, at::Tensor index_tensor, at::Tensor count_tensor, at::Tensor out_tensor){
  // src: (L, C)   indices: (L)  out: (N, C)
  CHECK_INPUT(src_tensor);
  CHECK_INPUT(index_tensor);
  CHECK_INPUT(count_tensor);
  CHECK_INPUT(out_tensor);

  int L = src_tensor.size(0);
  int C = src_tensor.size(1);
  int N = out_tensor.size(0);

  const float *src = src_tensor.data<float>();
  const int *index = index_tensor.data<int>();
  int *count = count_tensor.data<int>();
  float *out = out_tensor.data<float>();

  scatter_avg_kernel_launcher(L, N, C, index, src, count, out);
  return 1;
}

int scatter_avg_grad_wrapper(at::Tensor index_tensor, at::Tensor count_tensor, at::Tensor grad_out_tensor,at::Tensor grad_src_tensor){
  // src: (L, C)   indices: (L)  out: (N, C)
  CHECK_INPUT(index_tensor);
  CHECK_INPUT(count_tensor);
  CHECK_INPUT(grad_out_tensor);
  CHECK_INPUT(grad_src_tensor);

  int L = grad_src_tensor.size(0);
  int C = grad_src_tensor.size(1);
//  int N = grad_out_tensor.size(0);

  const int *index = index_tensor.data<int>();
  const int *count = count_tensor.data<int>();
  const float *grad_out = grad_out_tensor.data<float>();
  float *grad_src = grad_src_tensor.data<float>();

  scatter_avg_grad_kernel_launcher(L, C, index, count, grad_out, grad_src);
  return 1;
}
