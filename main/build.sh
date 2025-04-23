#!/bin/bash

cmake -S . -B test -DGGML_USE_CUDA=ON
cmake --build test