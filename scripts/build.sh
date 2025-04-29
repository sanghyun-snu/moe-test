#!/bin/bash

rm -rf ../build/*
# find ../build -mindepth 1 -not -path "../build/src/ggml*" -exec rm -rf {} +

cmake -S .. -B ../build
cmake --build ../build