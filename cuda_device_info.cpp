#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        
        printf("Device %d: \"%s\"\n", device, deviceProp.name);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  CUDA Cores: %d\n", deviceProp.multiProcessorCount * 
               (deviceProp.major == 8 ? 128 : 
                deviceProp.major == 9 ? 128 : 
                deviceProp.major == 7 ? (deviceProp.minor == 0 ? 64 : 128) : 
                deviceProp.major == 6 ? 128 : 
                deviceProp.major == 5 ? 128 : 
                deviceProp.major == 3 ? 192 : 0));
        printf("  Warp Size: %d\n", deviceProp.warpSize);
        printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Max Blocks per Multiprocessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);
        printf("  Max Thread Dimensions: (%d, %d, %d)\n", 
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: (%d, %d, %d)\n", 
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Clock Rate: %.2f GHz\n", deviceProp.clockRate / 1e6);
        printf("  Memory Clock Rate: %.2f GHz\n", deviceProp.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d bits\n", deviceProp.memoryBusWidth);
        printf("  L2 Cache Size: %d KB\n", deviceProp.l2CacheSize / 1024);
        printf("  Total Constant Memory: %zu bytes\n", deviceProp.totalConstMem);
        printf("  Shared Memory per Block: %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Shared Memory per Multiprocessor: %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);
        printf("  Registers per Block: %d\n", deviceProp.regsPerBlock);
        printf("  Registers per Multiprocessor: %d\n", deviceProp.regsPerMultiprocessor);
        printf("  Concurrent Kernels: %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
        printf("  ECC Enabled: %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
        printf("  Compute Mode: %d\n", deviceProp.computeMode);
    }
    
    return 0;
}