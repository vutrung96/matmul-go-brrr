import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.cpp_extension import load_inline
import gc

# Import your existing CUDA kernel
from kernels.cublas import cublas

def benchmark_matmul(sizes, num_runs=10):
    """
    Benchmark matrix multiplication for different sizes.
    
    Args:
        sizes: List of matrix sizes to benchmark (N for NxN matrices)
        num_runs: Number of runs for each size to compute average
        
    Returns:
        Dictionary with results for each implementation
    """
    results = {
        'sizes': sizes,
        'custom_times': [],
        'pytorch_times': []
    }

    
    for size in sizes:
        print(f"Benchmarking size: {size}x{size}")
        custom_times = []
        pytorch_times = []
        
        for _ in range(num_runs):
            # Generate random square matrices
            a = torch.rand((size, size), device='cuda')
            b = torch.rand((size, size), device='cuda')
            
            # Benchmark PyTorch built-in
            torch.cuda.synchronize()
            start = time.time()
            cublas.cublas_matmul(a, b)
            torch.cuda.synchronize()
            pytorch_times.append(time.time() - start)
            
            # Clean up to avoid memory issues with large matrices
            del a, b
            gc.collect()
            torch.cuda.empty_cache()
        
        # Calculate averages, excluding the first run (warmup)
        results['custom_times'].append(np.mean(custom_times[1:]))
        results['pytorch_times'].append(np.mean(pytorch_times[1:]))
    
    return results

def plot_results(results):
    """
    Plot the benchmark results.
    
    Args:
        results: Dictionary with benchmark results
    """
    plt.figure(figsize=(10, 6))
    
    # Convert sizes to strings for better x-axis labels
    size_labels = [f"{s}x{s}" for s in results['sizes']]
    
    x = np.arange(len(size_labels))
    width = 0.35
    
    plt.bar(x - width/2, results['custom_times'], width, label='Custom Kernel')
    plt.bar(x + width/2, results['pytorch_times'], width, label='PyTorch Built-in')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance Comparison')
    plt.xticks(x, size_labels, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(results['custom_times']):
        plt.text(i - width/2, v + 0.0001, f"{v:.5f}s", ha='center', fontsize=9)
    
    for i, v in enumerate(results['pytorch_times']):
        plt.text(i + width/2, v + 0.0001, f"{v:.5f}s", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('matmul_benchmark.png')
    plt.show()

def print_results(results):
    """
    Print the benchmark results in a table format.
    
    Args:
        results: Dictionary with benchmark results
    """
    print("\n" + "="*60)
    print(f"{'Size':<12} {'Custom Kernel (s)':<20} {'PyTorch (s)':<20} {'Speedup':<10}")
    print("-"*60)
    
    for i, size in enumerate(results['sizes']):
        custom = results['custom_times'][i]
        pytorch = results['pytorch_times'][i]
        speedup = pytorch / custom if custom > 0 else float('inf')
        
        print(f"{size}x{size:<8} {custom:<20.6f} {pytorch:<20.6f} {speedup:<10.2f}x")
    
    print("="*60)

if __name__ == "__main__":
    # Sizes to benchmark (N for NxN matrices)
    sizes = [2**14]
    
    # Run benchmarks
    results = benchmark_matmul(sizes, num_runs=1)
    
    # Print and plot results
    print_results(results)
    plot_results(results) 