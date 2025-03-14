import os
from torch.utils.cpp_extension import load

# Get the current directory (where this file is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the current directory
cpp_source_path = os.path.join(current_dir, 'mem_coalesce.cpp')
cuda_source_path = os.path.join(current_dir, 'mem_coalesce.cu')
build_dir = os.path.join(current_dir, '.build')

mem_coalesce = load(
    name='mem_coalesce',
    sources=[cpp_source_path, cuda_source_path],
    extra_cflags=['-O2'],
    build_directory=build_dir,
)