#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

extern void mem_coalesce_wrapper(const float* A, const float* B, float* C, int M, int N, int K);

torch::Tensor mem_coalesce(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);

    torch::Tensor C = torch::zeros({M, K}, at::kCUDA);
    mem_coalesce_wrapper(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mem_coalesce", &mem_coalesce, "Matrix multiplication using basic kernel");
}
