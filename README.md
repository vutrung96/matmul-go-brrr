# matmul

To benchmark the performance of different matrix multiplication kernels, run:

```bash
python matmul_benchmark.py
```

This will generate a plot of the performance of the different kernels in matmul_benchmark.png.

We currently have two kernels:

- `basic`: A basic CUDA kernel that multiplies two matrices.
- `cublas`: A wrapper around cuBLAS.

