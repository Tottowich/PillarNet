//
// Created by sgs on 2021/7/13.
//

#include "query_ops_gpu.h"


int create_pillar_indices_stack_wrapper(float bev_size, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
                                        at::Tensor pillar_mask_tensor) {
	CHECK_INPUT(xyz_tensor);
	CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(pillar_mask_tensor);

    int N = xyz_tensor.size(0);
    int C = xyz_tensor.size(1);
	assert(C == 3);
    int B = pillar_mask_tensor.size(0);
    int H = pillar_mask_tensor.size(1);
    int W = pillar_mask_tensor.size(2);

    const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data_ptr<int>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    bool *pillar_mask = pillar_mask_tensor.data_ptr<bool>();

    create_pillar_indices_stack_kernel_launcher(N, B, H, W, bev_size, xyz, xyz_batch_cnt, pillar_mask);

    return 1;
}

int create_pillar_indice_pairs_stack_wrapper(float radius, float bev_size,
											 at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
                                             at::Tensor bev_mask_tensor, at::Tensor indice_pairs_tensor) {
	CHECK_INPUT(xyz_tensor);
	CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(bev_mask_tensor);
    CHECK_INPUT(indice_pairs_tensor);

    int N = xyz_tensor.size(0);
    int C = xyz_tensor.size(1);
	assert(C == 3);
    int B = bev_mask_tensor.size(0);
    int H = bev_mask_tensor.size(1);
    int W = bev_mask_tensor.size(2);
    int K = indice_pairs_tensor.size(1);

    const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data_ptr<int>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    bool *bev_mask = bev_mask_tensor.data_ptr<bool>();
    int *indice_pairs = indice_pairs_tensor.data_ptr<int>();

    create_pillar_indice_pairs_stack_kernel_launcher(N, B, H, W, K, radius, bev_size, xyz, xyz_batch_cnt,
													 bev_mask, indice_pairs);
    return 1;
}

int create_pillar_indice_pairsv2_stack_wrapper(float radius, float bev_size, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
                                               at::Tensor pillar_bev_indices_tensor, at::Tensor indice_pairs_tensor) {
	CHECK_INPUT(xyz_tensor);
	CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(pillar_bev_indices_tensor);
    CHECK_INPUT(indice_pairs_tensor);

    int N = xyz_tensor.size(0);
    int C = xyz_tensor.size(1);
	assert(C == 3);
    int B = pillar_bev_indices_tensor.size(0);
    int H = pillar_bev_indices_tensor.size(1);
    int W = pillar_bev_indices_tensor.size(2);
    int K = indice_pairs_tensor.size(1);

    const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data_ptr<int>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    const int *pillar_bev_indices = pillar_bev_indices_tensor.data_ptr<int>();
    int *indice_pairs = indice_pairs_tensor.data_ptr<int>();

    create_pillar_indice_pairsv2_stack_kernel_launcher(N, B, H, W, K, radius, bev_size, xyz, xyz_batch_cnt,
													   pillar_bev_indices, indice_pairs);
    return 1;
}

int create_pillar_indice_pairsv2a_stack_wrapper(float bev_size, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
                                                at::Tensor pillar_bev_indices_tensor, at::Tensor indice_pairs_tensor) {
	CHECK_INPUT(xyz_tensor);
	CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(pillar_bev_indices_tensor);
    CHECK_INPUT(indice_pairs_tensor);

    int N = xyz_tensor.size(0);
    int C = xyz_tensor.size(1);
    assert(C == 3);
    int B = pillar_bev_indices_tensor.size(0);
    int H = pillar_bev_indices_tensor.size(1);
    int W = pillar_bev_indices_tensor.size(2);

    const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data_ptr<int>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    const int *pillar_bev_indices = pillar_bev_indices_tensor.data_ptr<int>();
    int *indice_pairs = indice_pairs_tensor.data_ptr<int>();

    create_pillar_indice_pairsv2a_stack_kernel_launcher(N, B, H, W, bev_size, xyz, xyz_batch_cnt,
														pillar_bev_indices, indice_pairs);
    return 1;
}

int create_bev_indices_wrapper(at::Tensor pillar_indices_tensor, at::Tensor bev_indices_tensor) {
	// pillar_indices_tensor: (M, 3) [byx]     bev_indices_tensor: (B, H, W)
    CHECK_INPUT(pillar_indices_tensor);
    CHECK_INPUT(bev_indices_tensor);

//    int B = bev_indices_tensor.size(0);
    int H = bev_indices_tensor.size(1);
    int W = bev_indices_tensor.size(2);
    int M = pillar_indices_tensor.size(0);

    const int *pillarIndices = pillar_indices_tensor.data_ptr<int>();
    int *bevIndices = bev_indices_tensor.data_ptr<int>();

    create_bev_indices_kernel_launcher(M, H, W, pillarIndices, bevIndices);

    return 1;
}

int create_pillar_indices_wrapper(at::Tensor bev_indices_tensor, at::Tensor pillar_indices_tensor) {
	// pillar_indices_tensor: (M, 3) [byx]     bev_indices_tensor: (B, H, W)
	CHECK_INPUT(bev_indices_tensor);
	CHECK_INPUT(pillar_indices_tensor);

	int B = bev_indices_tensor.size(0);
	int H = bev_indices_tensor.size(1);
	int W = bev_indices_tensor.size(2);

	const int *bevIndices = bev_indices_tensor.data_ptr<int>();
	int *pillarIndices = pillar_indices_tensor.data_ptr<int>();

	create_pillar_indices_kernel_launcher(B, H, W, bevIndices, pillarIndices);

	return 1;
}


int update_indice_pairs_wrapper(at::Tensor location_tensor, at::Tensor indice_pairs_tensor){
    CHECK_INPUT(location_tensor);
    CHECK_INPUT(indice_pairs_tensor);

    int N = indice_pairs_tensor.size(0);
    int K = indice_pairs_tensor.size(1);

    const int *location = location_tensor.data_ptr<int>();
    int *indicePairs = indice_pairs_tensor.data_ptr<int>();

    update_indice_pairs_kernel_launcher(N, K, location, indicePairs);
    return 1;
}


int create_indice2bev_kernel_wrapper(at::Tensor location_tensor, at::Tensor indice2bev_tensor){
    CHECK_INPUT(location_tensor);
    CHECK_INPUT(indice2bev_tensor);

    int B = location_tensor.size(0);
    int H = location_tensor.size(1);
    int W = location_tensor.size(2);

    const int *location = location_tensor.data_ptr<int>();
    int *indice2bev = indice2bev_tensor.data_ptr<int>();

    create_indice2bev_kernel_launcher(B * H * W, location, indice2bev);
    return 1;
}
