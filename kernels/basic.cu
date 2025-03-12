#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <ctime>
#define BLOCK_SIZE 32

__global__ void basic_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row*N + i] * B[col + K*i];
        }
        C[row * K + col] = sum;
    }
}

void basic_matmul_wrapper(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    basic_matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
}