import pycuda.driver as cuda
import pycuda.autoinit

device = cuda.Device(0)  # Use the first GPU

# Dictionary mapping attribute constants to their names and explanations
attributes = {
    cuda.device_attribute.MAX_THREADS_PER_BLOCK: 
        ("MAX_THREADS_PER_BLOCK", "Maximum number of threads per block"),
    
    cuda.device_attribute.MAX_BLOCK_DIM_X: 
        ("MAX_BLOCK_DIM_X", "Maximum block dimension in X direction"),
    
    cuda.device_attribute.MAX_BLOCK_DIM_Y: 
        ("MAX_BLOCK_DIM_Y", "Maximum block dimension in Y direction"),
    
    cuda.device_attribute.MAX_BLOCK_DIM_Z: 
        ("MAX_BLOCK_DIM_Z", "Maximum block dimension in Z direction"),
    
    cuda.device_attribute.MAX_GRID_DIM_X: 
        ("MAX_GRID_DIM_X", "Maximum grid dimension in X direction (number of blocks)"),
    
    cuda.device_attribute.MAX_GRID_DIM_Y: 
        ("MAX_GRID_DIM_Y", "Maximum grid dimension in Y direction (number of blocks)"),
    
    cuda.device_attribute.MAX_GRID_DIM_Z: 
        ("MAX_GRID_DIM_Z", "Maximum grid dimension in Z direction (number of blocks)"),
    
    cuda.device_attribute.TOTAL_CONSTANT_MEMORY: 
        ("TOTAL_CONSTANT_MEMORY", "Size of constant memory in bytes (fast, read-only memory)"),
    
    cuda.device_attribute.WARP_SIZE: 
        ("WARP_SIZE", "Number of threads in a warp (threads execute in groups of this size)"),
    
    cuda.device_attribute.MAX_PITCH: 
        ("MAX_PITCH", "Maximum pitch allowed for memory copies in bytes"),
    
    cuda.device_attribute.CLOCK_RATE: 
        ("CLOCK_RATE", "Clock frequency in kilohertz"),
    
    cuda.device_attribute.TEXTURE_ALIGNMENT: 
        ("TEXTURE_ALIGNMENT", "Alignment requirement for textures in bytes"),
    
    cuda.device_attribute.GPU_OVERLAP: 
        ("GPU_OVERLAP", "Device can simultaneously process CUDA kernels and memory transfers (1=Yes, 0=No)"),
    
    cuda.device_attribute.MULTIPROCESSOR_COUNT: 
        ("MULTIPROCESSOR_COUNT", "Number of streaming multiprocessors (SMs) on the device"),
    
    cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK: 
        ("MAX_SHARED_MEMORY_PER_BLOCK", "Maximum shared memory available per block in bytes"),
    
    cuda.device_attribute.MAX_REGISTERS_PER_BLOCK: 
        ("MAX_REGISTERS_PER_BLOCK", "Maximum number of 32-bit registers available per block"),
    
    cuda.device_attribute.KERNEL_EXEC_TIMEOUT: 
        ("KERNEL_EXEC_TIMEOUT", "Kernel execution timeout enabled (1=Yes, 0=No)"),
    
    cuda.device_attribute.INTEGRATED: 
        ("INTEGRATED", "Device is integrated with host memory (1=Yes, 0=No)"),
    
    cuda.device_attribute.CAN_MAP_HOST_MEMORY: 
        ("CAN_MAP_HOST_MEMORY", "Device can map host memory into CUDA address space (1=Yes, 0=No)"),
    
    cuda.device_attribute.COMPUTE_MODE: 
        ("COMPUTE_MODE", "Compute mode: 0=Default, 1=Exclusive, 2=Prohibited, 3=Exclusive Process"),
    
    cuda.device_attribute.MAXIMUM_TEXTURE1D_WIDTH: 
        ("MAXIMUM_TEXTURE1D_WIDTH", "Maximum 1D texture width"),
    
    cuda.device_attribute.MAXIMUM_TEXTURE2D_WIDTH: 
        ("MAXIMUM_TEXTURE2D_WIDTH", "Maximum 2D texture width"),
    
    cuda.device_attribute.MAXIMUM_TEXTURE2D_HEIGHT: 
        ("MAXIMUM_TEXTURE2D_HEIGHT", "Maximum 2D texture height"),
    
    cuda.device_attribute.MAXIMUM_TEXTURE3D_WIDTH: 
        ("MAXIMUM_TEXTURE3D_WIDTH", "Maximum 3D texture width"),
    
    cuda.device_attribute.MAXIMUM_TEXTURE3D_HEIGHT: 
        ("MAXIMUM_TEXTURE3D_HEIGHT", "Maximum 3D texture height"),
    
    cuda.device_attribute.MAXIMUM_TEXTURE3D_DEPTH: 
        ("MAXIMUM_TEXTURE3D_DEPTH", "Maximum 3D texture depth"),
    
    cuda.device_attribute.MAXIMUM_TEXTURE2D_ARRAY_WIDTH: 
        ("MAXIMUM_TEXTURE2D_ARRAY_WIDTH", "Maximum 2D texture array width"),
    
    cuda.device_attribute.MAXIMUM_TEXTURE2D_ARRAY_HEIGHT: 
        ("MAXIMUM_TEXTURE2D_ARRAY_HEIGHT", "Maximum 2D texture array height"),
    
    cuda.device_attribute.MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES: 
        ("MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES", "Maximum number of slices in a 2D texture array"),
    
    cuda.device_attribute.SURFACE_ALIGNMENT: 
        ("SURFACE_ALIGNMENT", "Alignment requirement for surfaces in bytes"),
    
    cuda.device_attribute.CONCURRENT_KERNELS: 
        ("CONCURRENT_KERNELS", "Device can execute multiple kernels concurrently (1=Yes, 0=No)"),
    
    cuda.device_attribute.ECC_ENABLED: 
        ("ECC_ENABLED", "Error Correction Code is enabled (1=Yes, 0=No)"),
    
    cuda.device_attribute.PCI_BUS_ID: 
        ("PCI_BUS_ID", "PCI bus ID of the device"),
    
    cuda.device_attribute.PCI_DEVICE_ID: 
        ("PCI_DEVICE_ID", "PCI device ID of the device"),
    
    cuda.device_attribute.TCC_DRIVER: 
        ("TCC_DRIVER", "Device is using TCC driver model (1=Yes, 0=No)"),
    
    cuda.device_attribute.MEMORY_CLOCK_RATE: 
        ("MEMORY_CLOCK_RATE", "Memory clock frequency in kilohertz"),
    
    cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH: 
        ("GLOBAL_MEMORY_BUS_WIDTH", "Global memory bus width in bits"),
    
    cuda.device_attribute.L2_CACHE_SIZE: 
        ("L2_CACHE_SIZE", "Size of L2 cache in bytes"),
    
    cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR: 
        ("MAX_THREADS_PER_MULTIPROCESSOR", "Maximum number of threads per multiprocessor"),
    
    cuda.device_attribute.ASYNC_ENGINE_COUNT: 
        ("ASYNC_ENGINE_COUNT", "Number of asynchronous engines (for concurrent data transfers)"),
    
    cuda.device_attribute.UNIFIED_ADDRESSING: 
        ("UNIFIED_ADDRESSING", "Device shares a unified address space with the host (1=Yes, 0=No)"),
    
    cuda.device_attribute.PCI_DOMAIN_ID: 
        ("PCI_DOMAIN_ID", "PCI domain ID of the device"),
    
    cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR: 
        ("COMPUTE_CAPABILITY_MAJOR", "Major compute capability version number"),
    
    cuda.device_attribute.COMPUTE_CAPABILITY_MINOR: 
        ("COMPUTE_CAPABILITY_MINOR", "Minor compute capability version number"),
    
    cuda.device_attribute.STREAM_PRIORITIES_SUPPORTED: 
        ("STREAM_PRIORITIES_SUPPORTED", "Device supports stream priorities (1=Yes, 0=No)"),
    
    cuda.device_attribute.GLOBAL_L1_CACHE_SUPPORTED: 
        ("GLOBAL_L1_CACHE_SUPPORTED", "Device supports caching global memory in L1 cache (1=Yes, 0=No)"),
    
    cuda.device_attribute.LOCAL_L1_CACHE_SUPPORTED: 
        ("LOCAL_L1_CACHE_SUPPORTED", "Device supports caching local memory in L1 cache (1=Yes, 0=No)"),
    
    cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: 
        ("MAX_SHARED_MEMORY_PER_MULTIPROCESSOR", "Maximum shared memory per multiprocessor in bytes"),
    
    cuda.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR: 
        ("MAX_REGISTERS_PER_MULTIPROCESSOR", "Maximum number of 32-bit registers per multiprocessor"),
    
    cuda.device_attribute.MANAGED_MEMORY: 
        ("MANAGED_MEMORY", "Device supports allocating managed memory (1=Yes, 0=No)"),
    
    cuda.device_attribute.MULTI_GPU_BOARD: 
        ("MULTI_GPU_BOARD", "Device is on a multi-GPU board (1=Yes, 0=No)"),
    
    cuda.device_attribute.PAGEABLE_MEMORY_ACCESS: 
        ("PAGEABLE_MEMORY_ACCESS", "Device supports coherently accessing pageable memory (1=Yes, 0=No)"),
    
    cuda.device_attribute.CONCURRENT_MANAGED_ACCESS: 
        ("CONCURRENT_MANAGED_ACCESS", "Device can coherently access managed memory concurrently with the CPU (1=Yes, 0=No)"),
    
    cuda.device_attribute.COMPUTE_PREEMPTION_SUPPORTED: 
        ("COMPUTE_PREEMPTION_SUPPORTED", "Device supports compute preemption (1=Yes, 0=No)"),
    
    cuda.device_attribute.CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: 
        ("CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM", "Device can access host registered memory at the same virtual address as the CPU (1=Yes, 0=No)"),
}

# Try to add newer attributes if available
try:
    attributes[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN] = (
        "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", 
        "Maximum shared memory per block with opt-in in bytes"
    )
    
    attributes[cuda.device_attribute.PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES] = (
        "PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES", 
        "Device accesses pageable memory via host's page tables (1=Yes, 0=No)"
    )
    
    attributes[cuda.device_attribute.DIRECT_MANAGED_MEM_ACCESS_FROM_HOST] = (
        "DIRECT_MANAGED_MEM_ACCESS_FROM_HOST", 
        "Host can directly access managed memory on the device without migration (1=Yes, 0=No)"
    )
    
    attributes[cuda.device_attribute.MAX_PERSISTING_L2_CACHE_SIZE] = (
        "MAX_PERSISTING_L2_CACHE_SIZE", 
        "Maximum L2 cache size that can be persisted in bytes"
    )
    
    attributes[cuda.device_attribute.MAX_BLOCKS_PER_MULTIPROCESSOR] = (
        "MAX_BLOCKS_PER_MULTIPROCESSOR", 
        "Maximum number of thread blocks that can be resident on a multiprocessor"
    )
    
    attributes[cuda.device_attribute.GENERIC_COMPRESSION_SUPPORTED] = (
        "GENERIC_COMPRESSION_SUPPORTED", 
        "Device supports compressing memory (1=Yes, 0=No)"
    )
    
    attributes[cuda.device_attribute.RESERVED_SHARED_MEMORY_PER_BLOCK] = (
        "RESERVED_SHARED_MEMORY_PER_BLOCK", 
        "Shared memory reserved by CUDA driver per block in bytes"
    )
    
    attributes[cuda.device_attribute.READ_ONLY_HOST_REGISTER_SUPPORTED] = (
        "READ_ONLY_HOST_REGISTER_SUPPORTED", 
        "Device supports read-only host memory registration (1=Yes, 0=No)"
    )
    
    attributes[cuda.device_attribute.MEMORY_POOLS_SUPPORTED] = (
        "MEMORY_POOLS_SUPPORTED", 
        "Device supports CUDA memory pools (1=Yes, 0=No)"
    )
except AttributeError:
    # Some attributes might not be available in older PyCUDA versions
    pass

# Print basic device info
print(f"Device: {device.name()}")
print(f"Compute Capability: {device.compute_capability()[0]}.{device.compute_capability()[1]}")
print(f"Total Memory: {device.total_memory() / (1024**2):.2f} MB")
print("\nDevice Attributes:")
print("-" * 80)

# Calculate theoretical memory bandwidth
try:
    mem_clock = device.get_attribute(cuda.device_attribute.MEMORY_CLOCK_RATE) / 1000  # MHz
    bus_width = device.get_attribute(cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH)
    bandwidth = 2 * mem_clock * (bus_width / 8) / 1000  # GB/s (factor of 2 for DDR)
    print(f"Theoretical Memory Bandwidth: {bandwidth:.2f} GB/s (calculated from memory clock and bus width)")
except:
    pass

# Calculate approximate CUDA cores (based on compute capability and SM count)
try:
    sm_count = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
    cc_major = device.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR)
    cc_minor = device.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MINOR)
    
    # Cores per SM based on architecture
    if cc_major == 8:  # Ampere/Ada Lovelace
        cores_per_sm = 128
    elif cc_major == 7:  # Volta/Turing
        cores_per_sm = 64 if cc_minor == 0 else 128
    elif cc_major == 6:  # Pascal
        cores_per_sm = 128
    elif cc_major == 5:  # Maxwell
        cores_per_sm = 128
    elif cc_major == 3:  # Kepler
        cores_per_sm = 192
    else:
        cores_per_sm = 0
    
    if cores_per_sm > 0:
        cuda_cores = sm_count * cores_per_sm
        print(f"Estimated CUDA Cores: {cuda_cores} ({sm_count} SMs × {cores_per_sm} cores/SM)")
except:
    pass

print("-" * 80)

# Print all attributes with explanations
for attr, (name, explanation) in attributes.items():
    try:
        value = device.get_attribute(attr)
        
        # Format special values with units or explanations
        formatted_value = value
        
        if "CLOCK_RATE" in name:
            formatted_value = f"{value / 1000:.2f} MHz"
        elif "MEMORY" in name and "PER" in name and value > 1024:
            formatted_value = f"{value:,} bytes ({value / 1024:.1f} KB)"
        elif name == "L2_CACHE_SIZE" or name == "MAX_PERSISTING_L2_CACHE_SIZE":
            formatted_value = f"{value:,} bytes ({value / (1024*1024):.2f} MB)"
        elif name == "COMPUTE_MODE":
            modes = {0: "DEFAULT", 1: "EXCLUSIVE", 2: "PROHIBITED", 3: "EXCLUSIVE_PROCESS"}
            formatted_value = f"{value} ({modes.get(value, 'UNKNOWN')})"
        elif value in [0, 1] and not any(x in name for x in ["ID", "COUNT", "SIZE", "WIDTH", "HEIGHT", "DEPTH"]):
            formatted_value = f"{value} ({'Yes' if value == 1 else 'No'})"
        
        print(f"{name}: {formatted_value}")
        print(f"    {explanation}")
    except cuda.LogicError:
        # Skip attributes that aren't supported on this device/driver
        pass 