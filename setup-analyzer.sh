#!/bin/bash

# qcat
cmake -S analyzer -B analyzer/build \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build analyzer/build -- -j

export PATH=$(pwd)/analyzer/build/examples:$PATH
export LD_LIBRARY_PATH=$(pwd)/analyzer/build/qcat:$LD_LIBRARY_PATH