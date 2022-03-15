#include "sample_group_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("query_points_in_boxes", &query_points_in_boxes, "query_points_in_boxes");

m.def("furthest_point_sampling_single_wrapper", &furthest_point_sampling_single_wrapper, "furthest_point_sampling_single_wrapper");

m.def("voxel_average_down_sample", &voxel_average_down_sample, "voxel_average_down_sample");
m.def("voxel_average_down_sample_v1", &voxel_average_down_sample_v1, "voxel_average_down_sample_v1");
m.def("voxel_unique_down_sample", &voxel_unique_down_sample, "voxel_unique_down_sample");
m.def("voxel_unique_down_sample_v1", &voxel_unique_down_sample_v1, "voxel_unique_down_sample_v1");

m.def("furthest_point_sampling_single_wrapper", &furthest_point_sampling_single_wrapper, "furthest_point_sampling_single_wrapper");

m.def("gather_points_wrapper", &gather_points_wrapper, "gather_points_wrapper");
m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper, "gather_points_grad_wrapper");

m.def("group_points_wrapper_stack", &group_points_wrapper_stack, "group_points_wrapper_stack");
m.def("group_points_grad_wrapper_stack", &group_points_grad_wrapper_stack, "group_points_grad_wrapper_stack");
}