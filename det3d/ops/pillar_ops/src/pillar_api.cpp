//
// Created by sgs on 2021/7/20.
//

#include "scatter_ops_gpu.h"
#include "group_ops_gpu.h"
#include "interpolate_ops_gpu.h"
#include "query_ops_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("create_pillar_indice_pairs_stack_wrapper", &create_pillar_indice_pairs_stack_wrapper, "create_pillar_indice_pairs_stack_wrapper");
m.def("create_pillar_indice_pairsv2_stack_wrapper", &create_pillar_indice_pairsv2_stack_wrapper, "create_pillar_indice_pairsv2_stack_wrapper");
m.def("create_pillar_indice_pairsv2a_stack_wrapper", &create_pillar_indice_pairsv2a_stack_wrapper, "create_pillar_indice_pairsv2a_stack_wrapper");

m.def("create_pillar_indices_wrapper", &create_pillar_indices_wrapper, "create_pillar_indices_wrapper");
m.def("create_pillar_indices_stack_wrapper", &create_pillar_indices_stack_wrapper, "create_pillar_indices_stack_wrapper");
m.def("create_bev_indices_wrapper", &create_bev_indices_wrapper, "create_bev_indices_wrapper");

m.def("update_indice_pairs_wrapper", &update_indice_pairs_wrapper, "update_indice_pairs_wrapper");
m.def("create_indice2bev_kernel_wrapper", &create_indice2bev_kernel_wrapper, "create_indice2bev_kernel_wrapper");

m.def("flatten_indice_pairs_wrapper", &flatten_indice_pairs_wrapper, "flatten_indice_pairs_wrapper");
m.def("gather_feature_wrapper", &gather_feature_wrapper, "gather_feature_wrapper");
m.def("gather_feature_grad_wrapper", &gather_feature_grad_wrapper, "gather_feature_grad_wrapper");

m.def("getInterpIndices2D", &getInterpIndices2D, "getInterpIndices2D");
m.def("getInterpIndices3D", &getInterpIndices3D, "getInterpIndices3D");

m.def("indiceInterpForward", &indiceInterpForward, "indiceInterpForward");
m.def("indiceInterpBackward", &indiceInterpBackward, "indiceInterpBackward");

m.def("update_index_kernel_wrapper", &update_index_kernel_wrapper, "update_index_kernel_wrapper");
m.def("scatter_max_wrapper", &scatter_max_wrapper, "scatter_max_wrapper");
m.def("scatter_max_grad_wrapper", &scatter_max_grad_wrapper, "scatter_max_grad_wrapper");
}