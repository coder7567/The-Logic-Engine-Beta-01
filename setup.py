from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# We will conditionally build the CUDA extension only if CUDA is available,
# to avoid crashes during initial project setup on machines lacking nvcc.
try:
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        raise ImportError("CUDA is not available.")
        
    ext_modules = [
        CUDAExtension('logic_engine_cuda', [
            'src/cuda/attention_kernel.cu',
            'src/cuda/memory_allocator.cpp',
        ]),
    ]
except Exception as e:
    print(f"Skipping CUDA extension build: {e}")
    ext_modules = []

setup(
    name='logic_engine',
    version='0.1.0',
    packages=['logic_engine'],
    package_dir={'': 'src/python'},
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)
