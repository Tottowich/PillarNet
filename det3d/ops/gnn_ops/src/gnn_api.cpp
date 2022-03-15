
#include "ball_query_gpu.h"
#include "ball_group_gpu.h"
#include "scatter_ops_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("ball_indice_wrapper", &ball_indice_wrapper_stack, "ball_gather_points_wrapper_stack");
m.def("ball_query_wrapper", &ball_query_wrapper_stack, "ball_query_wrapper_stack");

m.def("gather_feature_wrapper", &gather_feature_wrapper, "gather_feature_wrapper");
m.def("gather_feature_grad_wrapper", &gather_feature_grad_wrapper, "gather_feature_grad_wrapper");

m.def("scatter_max_wrapper", &scatter_max_wrapper, "scatter_max_wrapper");
m.def("scatter_max_grad_wrapper", &scatter_max_grad_wrapper, "scatter_max_grad_wrapper");

m.def("scatter_avg_wrapper", &scatter_avg_wrapper, "scatter_avg_wrapper");
m.def("scatter_avg_grad_wrapper", &scatter_avg_grad_wrapper, "scatter_avg_grad_wrapper");
}
