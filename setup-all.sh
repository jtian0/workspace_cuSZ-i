#!/bin/bash

if [ $# -eq 1 ]; then
    if [[ "$1" = "purge" ]]; then
        echo "purging build files..."
        rm -fr \
            cusz-interp/build \
            fzgpu/claunch_cuda.o fzgpu/fz-gpu \
            cuszp/build/examples/bin \
            szx-cuda/build \
            zfp-cuda/build \
        exit 0
    else
        echo "specified CUDA version $1."
    fi
else
    echo "sh setup-all.sh [OPTION]"
    echo "  * \"purge\"  to reset this workspace"
    echo "  * \"11\"     to initialize the artifacts for CUDA 11"
    echo "  * \"12\"     to initialize the artifacts for CUDA 12"
    exit 1
fi

export PATH=$(pwd)/cusz-interp/build:$PATH
export PATH=$(pwd)/fzgpu:$PATH
export PATH=$(pwd)/cuszp/build/examples/bin:$PATH
export PATH=$(pwd)/szx-cuda/build:$PATH
export PATH=$(pwd)/zfp-cuda/build:$PATH

export PATH=$(pwd)/analyzer/build/examples:$PATH
export LD_LIBRARY_PATH=$(pwd)/analyzer/build/qcat:$LD_LIBRARY_PATH

export NVCOMP_VER="3.0.5"
NVCOMP_DIR11="nvcomp${NVCOMP_VER}-cuda11"
NVCOMP_DIR12="nvcomp${NVCOMP_VER}-cuda12"

export LD_LIBRARY_PATH=$(pwd)/${NVCOMP_DIR11}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/${NVCOMP_DIR12}/lib:$LD_LIBRARY_PATH

sh setup-compressors.sh
sh setup-analyzer.sh
python setup-nvcomp.py $1