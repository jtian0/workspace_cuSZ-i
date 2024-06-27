#!/bin/bash

# qcat
cmake -S analyzer -B analyzer/build \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build analyzer/build -- -j
