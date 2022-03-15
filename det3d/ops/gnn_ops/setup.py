from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gnn',
    ext_modules=[
        CUDAExtension('gnn_cuda', [
            'src/gnn_api.cpp',
            'src/ball_query.cpp',
            'src/ball_query_gpu.cu',
            'src/ball_group.cpp',
            'src/ball_group_gpu.cu',
            'src/scatter_ops.cpp',
            'src/scatter_ops_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda/include'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
