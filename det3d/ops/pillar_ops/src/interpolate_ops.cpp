//
// Created by sgs on 2021/7/18.
//
#include "interpolate_ops_gpu.h"


int getInterpIndices2D(int kernel, int H, int W, float by, float bx,
					   at::Tensor indices_tensor, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
                       at::Tensor indice_kernel_tensor, at::Tensor inverse_dist_tensor) {

    // indices_tensor: (M1+M2..., 3) [byx]  xyz_tensor: (N1+N2..., 3) [xyz]  xyz_batch_cnt_tensor: (N1, N2, ...)
    CHECK_INPUT(indices_tensor);
    CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(indice_kernel_tensor);
    CHECK_INPUT(inverse_dist_tensor);

    int B = xyz_batch_cnt_tensor.size(0);
    int M = indices_tensor.size(0);
    int N = indice_kernel_tensor.size(0);
    int K = indice_kernel_tensor.size(1);
    assert(K <= 4096);

    std::string msg = "due to limits of cuda hash, the volume of dense space "
                      "include batch size ";
    msg += "must less than std::numeric_limits<int>::max() = 2e9";
    assert_expr_msg(B * H * W < std::numeric_limits<int>::max(), msg);

    const int *indices = indices_tensor.data_ptr<int>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    const int *xyzBatchCnt = xyz_batch_cnt_tensor.data_ptr<int>();

    auto grid_indice_tensor = torch::full({B*H*W}, -1,
                                          torch::dtype(torch::kInt32).device(xyz_tensor.device()).requires_grad(false));
    int *gridIndice = grid_indice_tensor.data_ptr<int>();

    int *indiceKernel = indice_kernel_tensor.data_ptr<int>();
    float *inverseDist = inverse_dist_tensor.data_ptr<float>();

    indice_interp2d_kernel_launcher(M, N, B, H, W, K, kernel, by, bx, indices, gridIndice,
                                    xyzBatchCnt, xyz, indiceKernel, inverseDist);
    return 1;
}

int getInterpIndices3D(int kernel, int D, int H, int W, float vz, float vy, float vx,
					   at::Tensor indices_tensor, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
                       at::Tensor indice_kernel_tensor, at::Tensor inverse_dist_tensor) {
    CHECK_INPUT(indices_tensor);
    CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(indice_kernel_tensor);
    CHECK_INPUT(inverse_dist_tensor);

    int B = xyz_batch_cnt_tensor.size(0);
    int M = indices_tensor.size(0);
    int N = inverse_dist_tensor.size(0);
    int K = inverse_dist_tensor.size(1);
    assert(K <= 4096);

    std::string msg = "due to limits of cuda hash, the volume of dense space "
                      "include batch size ";
    msg += "must less than std::numeric_limits<long>::max()";
    assert_expr_msg(B * D * H * W < std::numeric_limits<int>::max(), msg);

    torch::Tensor grid_indice_tensor = torch::full({B*W*H*D}, -1,
                                                   torch::TensorOptions().dtype(torch::kInt32).device(xyz_batch_cnt_tensor.device()));
    int *gridIndice = grid_indice_tensor.data_ptr<int>();

    const int *indices = indices_tensor.data_ptr<int>();
    const int *xyzBatchCnt = xyz_batch_cnt_tensor.data_ptr<int>();
    const float *xyz = xyz_tensor.data_ptr<float>();

    int *indiceKernel = indice_kernel_tensor.data_ptr<int>();
    float *inverseDist = inverse_dist_tensor.data_ptr<float>();

    indice_interp3d_kernel_launcher(M, N, B, D, H, W, K, kernel, vz, vy, vx, indices,
                                    xyzBatchCnt, xyz, gridIndice, indiceKernel, inverseDist);

    return 1;
}

void indiceInterpForward(at::Tensor features_tensor, at::Tensor indice_kernel_tensor, at::Tensor inverse_dist_tensor,
                         at::Tensor out_tensor) {
    CHECK_INPUT(features_tensor);
    CHECK_INPUT(indice_kernel_tensor);
    CHECK_INPUT(inverse_dist_tensor);
    CHECK_INPUT(out_tensor);

    int C = features_tensor.size(1);
    int N = indice_kernel_tensor.size(0);
    int K = indice_kernel_tensor.size(1);

    const int *indiceKernel = indice_kernel_tensor.data_ptr<int>();
    const float *inverseDist = inverse_dist_tensor.data_ptr<float>();
    const float *features = features_tensor.data_ptr<float>();
    float *out = out_tensor.data_ptr<float>();

    interp_fwd_kernel_launcher(N, K, C, features, indiceKernel, inverseDist, out);
}

void indiceInterpBackward(at::Tensor grad_out_tensor, at::Tensor indice_kernel_tensor,
                          at::Tensor inverse_dist_tensor, at::Tensor grad_in_tensor) {
    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(indice_kernel_tensor);
    CHECK_INPUT(inverse_dist_tensor);
    CHECK_INPUT(grad_in_tensor);

    int C = grad_in_tensor.size(1);
    int N = indice_kernel_tensor.size(0);
    int K = indice_kernel_tensor.size(1);

    const int *indiceKernel = indice_kernel_tensor.data_ptr<int>();
    const float *inverseDist = inverse_dist_tensor.data_ptr<float>();
    const float *outGrad = grad_out_tensor.data_ptr<float>();
    float *inGrad = grad_in_tensor.data_ptr<float>();

    interp_bwd_kernel_launcher(N, K, C, outGrad, indiceKernel, inverseDist, inGrad);
}
