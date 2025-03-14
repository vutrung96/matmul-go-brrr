import torch
import argparse
import time
import gc
from kernels.basic import basic
from kernels.cublas import cublas

def debug_kernel(kernel_name, size, num_runs=5):
    """
    Run a specific kernel multiple times and report execution times.
    
    Args:
        kernel_name: Name of the kernel to debug
        size: Size of test matrices (N for NxN)
        num_runs: Number of times to run the kernel
    """
    # Available kernels
    kernels = {
        'basic': basic.basic_matmul,
        'cublas': cublas.cublas_matmul
    }
    
    if kernel_name not in kernels:
        print(f"Error: Kernel '{kernel_name}' not found. Available kernels: {list(kernels.keys())}")
        return
    
    kernel_func = kernels[kernel_name]
    
    print(f"Running {kernel_name} kernel with {size}x{size} matrices {num_runs} times...")
    
    # Create test matrices
    a = torch.rand((size, size), device='cuda')
    b = torch.rand((size, size), device='cuda')
    
    # Warmup run
    print("Warmup run...")
    result = kernel_func(a, b)
    torch.cuda.synchronize()
    
    # Actual timed runs
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        
        result = kernel_func(a, b)
        
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
        
        print(f"Run {i+1}: {times[-1]:.6f} seconds")
    
    # Print summary
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("\nSummary:")
    print(f"  Average time: {avg_time:.6f} seconds")
    print(f"  Min time:     {min_time:.6f} seconds")
    print(f"  Max time:     {max_time:.6f} seconds")
    
    # Clean up
    del a, b, result
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug a specific kernel')
    parser.add_argument('kernel', type=str, help='Kernel to debug (basic, cublas)')
    parser.add_argument('--size', type=int, default=1024, help='Matrix size (default: 1024)')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs (default: 5)')
    
    args = parser.parse_args()
    
    debug_kernel(args.kernel, args.size, args.runs) 