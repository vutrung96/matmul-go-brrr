#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <ctime>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <torch/torch.h>
#include <torch/extension.h>

void _cublas_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    cublasHandle_t handle;
    cudaStream_t stream;
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasCreate_v2(&handle);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream_v2(handle, stream);

    // For matrix multiplication C = A * B where:
    // A is M×N, B is N×K, and C is M×K
    // 
    // Since cuBLAS uses column-major order and our matrices are in row-major order,
    // we compute B^T * A^T which is equivalent to (A * B)^T
    // 
    // For row-major data, we compute: C = (B^T * A^T)^T
    // This is done by swapping the matrices and using OP_N
    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                K, M, N,     // dimensions: K×M = K×N * N×M
                &alpha,
                B, K,        // B is treated as column-major N×K (or row-major K×N)
                A, N,        // A is treated as column-major M×N (or row-major N×M)
                &beta,
                C, K);       // C is treated as column-major K×M (or row-major M×K)
    
    // Don't forget to destroy the handle when done
    cublasDestroy_v2(handle);
    cudaStreamDestroy(stream);
}

torch::Tensor cublas_matmul(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int N = A.size(1);
    const int K = B.size(1);

    torch::Tensor C = torch::zeros({M, K}, A.options());
    _cublas_matmul(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

// Define the Python module
PYBIND11_MODULE(cublas, m) {
    m.def("cublas_matmul", &cublas_matmul, "Matrix multiplication using cuBLAS");
}
