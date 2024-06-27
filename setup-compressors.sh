#!/bin/bash

# cusz-stock and cusz-interp
cmake -S cusz-interp -B cusz-interp/build \
    -D PSZ_BACKEND=cuda \
    -D PSZ_BUILD_EXAMPLES=off \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cusz-interp/build -- -j

# fzgpu
pushd fzgpu
make -j
popd

# szx-cuda
cmake -S szx-cuda -B szx-cuda/build \
    -D SZx_BUILD_CUDA=on \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build szx-cuda/build -- -j

# cuszp
cmake -S cuszp -B cuszp/build \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cuszp/build -- -j

# zfp-cuda
cmake -S zfp-cuda -B zfp-cuda/build \
    -D ZFP_WITH_CUDA=on \
    -D CUDA_SDK_ROOT_DIR=$(dirname $(which nvcc))/.. \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build zfp-cuda/build -- -j