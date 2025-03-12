import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import gc
import argparse

# Import your existing CUDA kernels
from kernels.cublas import cublas
from kernels.basic import basic

def verify_output(kernel_name, kernel_func, a, b, rtol=1e-5, atol=1e-5):
    """
    Verify that kernel output matches cuBLAS output within tolerance.
    
    Args:
        kernel_name: Name of the kernel being tested
        kernel_func: Function to call for matrix multiplication
        a, b: Input matrices
        rtol, atol: Relative and absolute tolerance for comparison
        
    Returns:
        Boolean indicating whether outputs match
    """
    # Get reference result from cuBLAS
    cublas_result = cublas.cublas_matmul(a, b)
    
    # Get result from the kernel being tested
    kernel_result = kernel_func(a, b)
    
    # Check if results match within tolerance
    is_close = torch.allclose(kernel_result, cublas_result, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"✓ {kernel_name} output matches cuBLAS within tolerance")
    else:
        # Calculate max absolute difference for debugging
        max_diff = torch.max(torch.abs(kernel_result - cublas_result))
        print(f"✗ {kernel_name} output DOES NOT match cuBLAS! Max difference: {max_diff}")
    
    return is_close

def benchmark_matmul(sizes, kernels, num_runs=10, verify=True):
    """
    Benchmark matrix multiplication for different sizes.
    
    Args:
        sizes: List of matrix sizes to benchmark (N for NxN matrices)
        kernels: Dictionary mapping kernel names to functions
        num_runs: Number of runs for each size to compute average
        verify: Whether to verify kernel outputs against cuBLAS
        
    Returns:
        Dictionary with results for each implementation
    """
    results = {
        'sizes': sizes,
        'cublas_times': []
    }
    
    # Initialize time arrays for each kernel
    for kernel_name in kernels:
        results[f'{kernel_name}_times'] = []
    
    for size in sizes:
        print(f"\nBenchmarking size: {size}x{size}")
        
        # Initialize time arrays for this size
        cublas_times = []
        kernel_times = {k: [] for k in kernels}
        
        # Generate random square matrices for verification
        if verify:
            verify_a = torch.rand((size, size), device='cuda')
            verify_b = torch.rand((size, size), device='cuda')
            
            # Verify each kernel's output
            for kernel_name, kernel_func in kernels.items():
                verify_output(kernel_name, kernel_func, verify_a, verify_b)
            
            # Clean up verification matrices
            del verify_a, verify_b
            torch.cuda.empty_cache()
        
        for run in range(num_runs):
            # Generate random square matrices
            a = torch.rand((size, size), device='cuda')
            b = torch.rand((size, size), device='cuda')
            
            # Benchmark cuBLAS (baseline)
            torch.cuda.synchronize()
            start = time.time()
            cublas.cublas_matmul(a, b)
            torch.cuda.synchronize()
            cublas_times.append(time.time() - start)
            
            # Benchmark each kernel
            for kernel_name, kernel_func in kernels.items():
                torch.cuda.synchronize()
                start = time.time()
                kernel_func(a, b)
                torch.cuda.synchronize()
                kernel_times[kernel_name].append(time.time() - start)
            
            # Clean up to avoid memory issues with large matrices
            del a, b
            gc.collect()
            torch.cuda.empty_cache()
        
        # Calculate averages, excluding the first run (warmup)
        results['cublas_times'].append(np.mean(cublas_times[1:]) if len(cublas_times) > 1 else cublas_times[0])
        
        for kernel_name in kernels:
            kernel_time = np.mean(kernel_times[kernel_name][1:]) if len(kernel_times[kernel_name]) > 1 else kernel_times[kernel_name][0]
            results[f'{kernel_name}_times'].append(kernel_time)
    
    return results

def plot_results(results, kernel_names):
    """
    Plot the benchmark results.
    
    Args:
        results: Dictionary with benchmark results
        kernel_names: List of kernel names to include in the plot
    """
    plt.figure(figsize=(12, 7))
    
    # Convert sizes to strings for better x-axis labels
    size_labels = [f"{s}x{s}" for s in results['sizes']]
    
    x = np.arange(len(size_labels))
    width = 0.8 / (len(kernel_names) + 1)  # +1 for cuBLAS
    
    # Plot cuBLAS times
    plt.bar(x - 0.4 + width/2, results['cublas_times'], width, label='cuBLAS')
    
    # Plot each kernel's times
    for i, kernel_name in enumerate(kernel_names):
        plt.bar(x - 0.4 + (i+1.5)*width, results[f'{kernel_name}_times'], width, label=f'{kernel_name}')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance Comparison')
    plt.xticks(x, size_labels, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(results['cublas_times']):
        plt.text(x[i] - 0.4 + width/2, v + 0.0001, f"{v:.5f}s", ha='center', fontsize=8, rotation=90)
    
    for k_idx, kernel_name in enumerate(kernel_names):
        for i, v in enumerate(results[f'{kernel_name}_times']):
            plt.text(x[i] - 0.4 + (k_idx+1.5)*width, v + 0.0001, f"{v:.5f}s", ha='center', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig('matmul_benchmark.png')
    plt.show()

def print_results(results, kernel_names):
    """
    Print the benchmark results in a table format.
    
    Args:
        results: Dictionary with benchmark results
        kernel_names: List of kernel names to include in the table
    """
    print("\n" + "="*80)
    header = f"{'Size':<12} {'cuBLAS (s)':<15}"
    for name in kernel_names:
        header += f" {name+' (s)':<15} {'vs cuBLAS':<10}"
    print(header)
    print("-"*80)
    
    for i, size in enumerate(results['sizes']):
        cublas_time = results['cublas_times'][i]
        line = f"{size}x{size:<8} {cublas_time:<15.6f}"
        
        for name in kernel_names:
            kernel_time = results[f'{name}_times'][i]
            speedup = cublas_time / kernel_time if kernel_time > 0 else float('inf')
            line += f" {kernel_time:<15.6f} {speedup:<10.2f}x"
        
        print(line)
    
    print("="*80)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark matrix multiplication kernels')
    parser.add_argument('--kernels', type=str, default='basic', 
                        help='Comma-separated list of kernels to benchmark (default: basic)')
    parser.add_argument('--sizes', type=str, default='1024,2048,4096',
                        help='Comma-separated list of matrix sizes to benchmark (default: 1024,2048,4096)')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs for each benchmark (default: 3)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification of kernel outputs against cuBLAS')
    
    args = parser.parse_args()
    
    # Parse kernel names
    kernel_names = [k.strip() for k in args.kernels.split(',')]
    
    # Create kernel dictionary
    available_kernels = {
        'basic': basic.basic_matmul
    }
    
    # Filter to requested kernels
    kernels_to_benchmark = {}
    for name in kernel_names:
        if name in available_kernels:
            kernels_to_benchmark[name] = available_kernels[name]
        else:
            print(f"Warning: Kernel '{name}' not found, skipping")
    
    if not kernels_to_benchmark:
        print("Error: No valid kernels specified")
        exit(1)
    
    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    
    # Run benchmarks
    results = benchmark_matmul(sizes, kernels_to_benchmark, num_runs=args.runs, verify=not args.no_verify)
    
    # Print and plot results
    print_results(results, list(kernels_to_benchmark.keys()))
    plot_results(results, list(kernels_to_benchmark.keys())) 