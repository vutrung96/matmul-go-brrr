#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

extern void basic_matmul_wrapper(const float* A, const float* B, float* C, int M, int N, int K);

torch::Tensor basic_matmul(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);

    torch::Tensor C = torch::zeros({M, K}, at::kCUDA);
    basic_matmul_wrapper(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("basic_matmul", &basic_matmul, "Matrix multiplication using basic kernel");
}
