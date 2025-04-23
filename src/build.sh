#!/bin/bash

# parameter parsing <llama or main or powerinfer>
if [ $# -eq 0 ]; then
    echo "No arguments provided. Please provide a target."
    echo "Usage: $0 <llama.cpp, main, powerinfer>"
    exit 1
fi

TARGET=$1
CMAKE_SRC_DIR=$(pwd)

# cmake build
if [ "$TARGET" == "llama.cpp" ]; then
    echo "Building llama.cpp"
    # check build/llama.cpp exist and create it if not
    if [ ! -d "build/llama" ]; then
        mkdir -p build/$TARGET
    fi
    pushd $TARGET && cmake -B $CMAKE_SRC_DIR/build/llama.cpp -DGGML_CUDA=ON && cmake --build $CMAKE_SRC_DIR/build/llama.cpp --config Release
    popd
elif [ "$TARGET" == "main" ]; then
    echo "Building main"
    mkdir -p build
    cd build
    cmake ..
    make -j 8
    cd ..
elif [ "$TARGET" == "powerinfer" ]; then
    echo "Building powerinfer"
    mkdir -p build
    cd build
    cmake ..
    make -j 8
    cd ..
else
    echo "Invalid target. Please provide a valid target."
fi
