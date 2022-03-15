//
// Created by sgs on 2021/2/5.
//
#include "sample_group_gpu.h"


#define assert_expr(x) if(!(x)) { assert_fail(#x, __FILE__, __LINE__, ""); }
#define assert_expr_msg(x, msg) if(!(x)) { assert_fail(#x, __FILE__, __LINE__, msg); }
void assert_fail(const char *str, const char *file, int line, std::string msg) {
  std::cerr << "Assertion failed " << str << " " << file << ":" << line << " " << msg << std::endl;
  abort();
}

int query_points_in_boxes(float extra, torch::Tensor rois_tensor, torch::Tensor points_tensor, torch::Tensor points_mask_tensor) {

	// param rois: (K, 7 + C) [x, y, z, w, l, h, ry] in raw lidar data format
	// param points: (N, 3+C) [x y z]
	// param points_mask: (N,) bool, initial - 0, 1 - exist

	CHECK_INPUT(rois_tensor);
	CHECK_INPUT(points_tensor);
	CHECK_INPUT(points_mask_tensor);

	int N = points_tensor.size(0);
	int CP = points_tensor.size(1);
	int M = rois_tensor.size(0);
	int CR = rois_tensor.size(1);

	const float* rois = rois_tensor.data_ptr<float>();
	const float* points = points_tensor.data_ptr<float>();

	bool* points_mask = points_mask_tensor.data<bool>();

	points_in_boxes_kernel_launcher(N, M, CP, CR, extra, rois, points, points_mask);

	return 1;
}

std::tuple<at::Tensor, at::Tensor> voxel_average_down_sample(at::Tensor points_tensor, float voxel, float max_points,
							         float lx, float ly, float lz, float hx, float hy, float hz) {
	// remove points outside of point cloud range [lx ly lz hx hy hz]
  CHECK_INPUT(points_tensor);

  int N = points_tensor.size(0);
  int C = points_tensor.size(1);
  const float *points = points_tensor.data_ptr<float>();

  std::string msg = "due to limits of int, the number of points ";
  msg += "must less than std::numeric_limits<int>::max() = 2e9";
  assert_expr_msg(N < std::numeric_limits<int>::max(), msg);

  std::string msg1 = "due to limits of long, thre voxel and pc range for grid";
  msg1 += "must less than std::numeric_limits<long>::max()";
  assert_expr_msg(ceil((hx - lx) / voxel) * ceil((hy - ly) / voxel) * ceil((hz - lz) / voxel) < std::numeric_limits<long>::max(), msg1);

  auto device = points_tensor.device();
  auto voxel_indice_tensor = torch::full({N}, -1, at::device(device).dtype(torch::kLong).requires_grad(false));
  long *voxel_indice = voxel_indice_tensor.data_ptr<long>();
  voxel_indice_kernel_launcher(N, C, voxel, lx, ly, lz, hx, hy, hz, points, voxel_indice);

  at::Tensor sort_voxel_indice_tensor, sort_index_tensor;
  std::tie(sort_voxel_indice_tensor, sort_index_tensor) = voxel_indice_tensor.sort();

  auto order_points_tensor = points_tensor.index_select(0, sort_index_tensor);
  auto mask_tensor = torch::zeros({N}, at::device(device).dtype(torch::kBool).requires_grad(false));

  float *order_points = order_points_tensor.data_ptr<float>();
  const long *sort_voxel_indice = sort_voxel_indice_tensor.data_ptr<long>();
//  const long *sort_index = sort_index_tensor.data_ptr<long>();
  bool *mask = mask_tensor.data_ptr<bool>();
  average_points_kernel_launcher(N, C, max_points, sort_voxel_indice, order_points, mask);

  return std::make_tuple(order_points_tensor, mask_tensor);
}

std::tuple<at::Tensor, at::Tensor> voxel_average_down_sample_v1(at::Tensor points_tensor, float voxel, float max_points,
															    int VX, int VY, int VZ) {
	// remove points outside of point cloud range [lx ly lz hx hy hz]
	CHECK_INPUT(points_tensor);

	int N = points_tensor.size(0);
	int C = points_tensor.size(1);
	const float *points = points_tensor.data_ptr<float>();

	std::string msg = "due to limits of int, the number of points ";
	msg += "must less than std::numeric_limits<int>::max() = 2e9";
	assert_expr_msg(N < std::numeric_limits<int>::max(), msg);

	std::string msg1 = "due to limits of long, thre voxel and pc range for grid";
	msg1 += "must less than std::numeric_limits<long>::max()";
	assert_expr_msg(VX * VY * VZ < std::numeric_limits<long>::max(), msg1);

	auto device = points_tensor.device();
	auto voxel_indice_tensor = torch::full({N}, -1, at::device(device).dtype(torch::kLong).requires_grad(false));
	long *voxel_indice = voxel_indice_tensor.data_ptr<long>();
	voxel_indice_v1_kernel_launcher(N, C, voxel, VX, VY, VZ, points, voxel_indice);

	at::Tensor sort_voxel_indice_tensor, sort_index_tensor;
	std::tie(sort_voxel_indice_tensor, sort_index_tensor) = voxel_indice_tensor.sort();

	auto order_points_tensor = points_tensor.index_select(0, sort_index_tensor);
	auto mask_tensor = torch::zeros({N}, at::device(device).dtype(torch::kBool).requires_grad(false));

	float *order_points = order_points_tensor.data_ptr<float>();
	const long *sort_voxel_indice = sort_voxel_indice_tensor.data_ptr<long>();
	//  const long *sort_index = sort_index_tensor.data_ptr<long>();
	bool *mask = mask_tensor.data_ptr<bool>();
	average_points_kernel_launcher(N, C, max_points, sort_voxel_indice, order_points, mask);

	return std::make_tuple(order_points_tensor, mask_tensor);
}

at::Tensor voxel_unique_down_sample(at::Tensor points_tensor, float voxel,
									float lx, float ly, float lz, float hx, float hy, float hz) {
	// remove points outside of point cloud range [lx ly lz hx hy hz]
	CHECK_INPUT(points_tensor);

	int N = points_tensor.size(0);
	int C = points_tensor.size(1);
	const float *points = points_tensor.data_ptr<float>();

	std::string msg = "due to limits of int, the number of points ";
	msg += "must less than std::numeric_limits<int>::max() = 2e9";
	assert_expr_msg(N < std::numeric_limits<int>::max(), msg);

	std::string msg1 = "due to limits of long, thre voxel and pc range for grid";
	msg1 += "must less than std::numeric_limits<long>::max() = 2e9";
	assert_expr_msg(ceil((hx - lx) / voxel) * ceil((hy - ly) / voxel) * ceil((hz - lz) / voxel) < std::numeric_limits<long>::max(), msg1);

	auto device = points_tensor.device();
	auto voxel_indice_tensor = torch::full({N}, -1, at::device(device).dtype(torch::kLong).requires_grad(false));
	long *voxel_indice = voxel_indice_tensor.data_ptr<long>();
	voxel_indice_kernel_launcher(N, C, voxel, lx, ly, lz, hx, hy, hz, points, voxel_indice);

	at::Tensor sort_voxel_indice_tensor, sort_index_tensor;
	std::tie(sort_voxel_indice_tensor, sort_index_tensor) = voxel_indice_tensor.sort();

	auto mask_tensor = torch::zeros({N}, at::device(device).dtype(torch::kBool).requires_grad(false));
	const long *sort_voxel_indice = sort_voxel_indice_tensor.data_ptr<long>();
	const long *sort_index = sort_index_tensor.data_ptr<long>();
	bool *mask = mask_tensor.data_ptr<bool>();
	unique_points_kernel_launcher(N, sort_voxel_indice, sort_index, mask);

	return mask_tensor;
}

at::Tensor voxel_unique_down_sample_v1(at::Tensor points_tensor, float voxel, int VX, int VY, int VZ) {
	// remove points outside of point cloud range [lx ly lz hx hy hz]
	CHECK_INPUT(points_tensor);

	int N = points_tensor.size(0);
	int C = points_tensor.size(1);
	const float *points = points_tensor.data_ptr<float>();

	std::string msg = "due to limits of int, the number of points ";
	msg += "must less than std::numeric_limits<int>::max() = 2e9";
	assert_expr_msg(N < std::numeric_limits<int>::max(), msg);

	std::string msg1 = "due to limits of long, thre voxel and pc range for grid";
	msg1 += "must less than std::numeric_limits<long>::max() = 2e9";
	assert_expr_msg(VX * VY * VZ < std::numeric_limits<long>::max(), msg1);

	auto device = points_tensor.device();
	auto voxel_indice_tensor = torch::full({N}, -1, at::device(device).dtype(torch::kLong).requires_grad(false));
	long *voxel_indice = voxel_indice_tensor.data_ptr<long>();
	voxel_indice_v1_kernel_launcher(N, C, voxel, VX, VY, VZ, points, voxel_indice);

	at::Tensor sort_voxel_indice_tensor, sort_index_tensor;
	std::tie(sort_voxel_indice_tensor, sort_index_tensor) = voxel_indice_tensor.sort();

	auto mask_tensor = torch::zeros({N}, at::device(device).dtype(torch::kBool).requires_grad(false));
	const long *sort_voxel_indice = sort_voxel_indice_tensor.data_ptr<long>();
	const long *sort_index = sort_index_tensor.data_ptr<long>();
	bool *mask = mask_tensor.data_ptr<bool>();
	unique_points_kernel_launcher(N, sort_voxel_indice, sort_index, mask);

	return mask_tensor;
}

int furthest_point_sampling_single_wrapper(int M, at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {
  CHECK_INPUT(points_tensor);
  CHECK_INPUT(temp_tensor);
  CHECK_INPUT(idx_tensor);

  int N = points_tensor.size(0);
  int C = points_tensor.size(1);

  const float *points = points_tensor.data_ptr<float>();
  float *temp = temp_tensor.data_ptr<float>();
  int *idx = idx_tensor.data_ptr<int>();

  furthest_point_sampling_kernel_launcher(1, N, M, C, points, temp, idx);
  return 1;
}

int gather_points_wrapper(at::Tensor idx_tensor, at::Tensor points_tensor, at::Tensor out_tensor){
  CHECK_INPUT(idx_tensor);
  CHECK_INPUT(points_tensor);
  CHECK_INPUT(out_tensor);

  int M = out_tensor.size(0);
  int C = out_tensor.size(1);
  const float *points = points_tensor.data_ptr<float>();
  const int *idx = idx_tensor.data_ptr<int>();
  float *out = out_tensor.data_ptr<float>();

  gather_points_kernel_launcher(M, C, idx, points, out);
  return 1;
}

int gather_points_grad_wrapper(at::Tensor idx_tensor, at::Tensor grad_out_tensor, at::Tensor grad_points_tensor){
  CHECK_INPUT(idx_tensor);
  CHECK_INPUT(grad_out_tensor);
  CHECK_INPUT(grad_points_tensor);

  int M = grad_out_tensor.size(0);
  int C = grad_out_tensor.size(1);

  const float *grad_out = grad_out_tensor.data_ptr<float>();
  const int *idx = idx_tensor.data_ptr<int>();
  float *grad_points = grad_points_tensor.data_ptr<float>();

  gather_points_grad_kernel_launcher(M, C, idx, grad_out, grad_points);
  return 1;
}

int group_points_wrapper_stack(int B, int M, int C, int nsample,
                               at::Tensor features_tensor, at::Tensor features_batch_cnt_tensor,
                               at::Tensor idx_tensor, at::Tensor idx_batch_cnt_tensor, at::Tensor out_tensor) {

  CHECK_INPUT(features_tensor);
  CHECK_INPUT(features_batch_cnt_tensor);
  CHECK_INPUT(idx_tensor);
  CHECK_INPUT(idx_batch_cnt_tensor);
  CHECK_INPUT(out_tensor);

  const float *features = features_tensor.data_ptr<float>();
  const int *idx = idx_tensor.data_ptr<int>();
  const int *features_batch_cnt = features_batch_cnt_tensor.data_ptr<int>();
  const int *idx_batch_cnt = idx_batch_cnt_tensor.data_ptr<int>();
  float *out = out_tensor.data_ptr<float>();

  group_points_kernel_launcher_stack(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, out);
  return 1;
}

int group_points_grad_wrapper_stack(int B, int M, int C, int N, int nsample,
                                    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor idx_batch_cnt_tensor,
                                    at::Tensor features_batch_cnt_tensor, at::Tensor grad_features_tensor) {

  CHECK_INPUT(grad_out_tensor);
  CHECK_INPUT(idx_tensor);
  CHECK_INPUT(idx_batch_cnt_tensor);
  CHECK_INPUT(features_batch_cnt_tensor);
  CHECK_INPUT(grad_features_tensor);

  const float *grad_out = grad_out_tensor.data_ptr<float>();
  const int *idx = idx_tensor.data_ptr<int>();
  const int *idx_batch_cnt = idx_batch_cnt_tensor.data_ptr<int>();
  const int *features_batch_cnt = features_batch_cnt_tensor.data_ptr<int>();
  float *grad_features = grad_features_tensor.data_ptr<float>();

  group_points_grad_kernel_launcher_stack(B, M, C, N, nsample, grad_out, idx, idx_batch_cnt, features_batch_cnt, grad_features);
  return 1;
}


