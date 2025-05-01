cmake_minimum_required(VERSION 3.14)
project(ggml_demo_root C CXX)

set(CMAKE_CUDA_ARCHITECTURES 86)

enable_language(CUDA)

# ggml path
set(GGML_BASE_LIB ${CMAKE_SOURCE_DIR}/lib/libggml-base.so)
set(GGML_CORE_LIB ${CMAKE_SOURCE_DIR}/lib/libggml.so)
set(GGML_CPU_LIB  ${CMAKE_SOURCE_DIR}/lib/libggml-cpu.so)
set(GGML_CUDA_LIB ${CMAKE_SOURCE_DIR}/lib/libggml-cuda.so)

set(GGML_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml/include)
set(GGML_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml/src)

# lz4 path
set(LZ4_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/lz4/lib)
set(LZ4_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/lz4/lib)
set(LZ4_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib) # should copy lz4.so in this directory
set(LZ4_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/liblz4.so)

# json path
set(JSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/json/single_include/nlohmann)
# set(CMAKE_BUILD_RPATH        ${CMAKE_SOURCE_DIR}/lib)
# set(CMAKE_INSTALL_RPATH      ${CMAKE_SOURCE_DIR}/lib)
# set(CMAKE_SKIP_BUILD_RPATH   FALSE)
# set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# 1. ggml-core
add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES
    IMPORTED_LOCATION ${GGML_CORE_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${GGML_INCLUDE_DIR}
)

# 2. ggml-base
add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base PROPERTIES
    IMPORTED_LOCATION ${GGML_BASE_LIB}
)

# 3. ggml-cpu
add_library(ggml-cpu SHARED IMPORTED)
set_target_properties(ggml-cpu PROPERTIES
    IMPORTED_LOCATION ${GGML_CPU_LIB}
)

# 4. ggml-cuda
add_library(ggml-cuda SHARED IMPORTED)
set_target_properties(ggml-cuda PROPERTIES
    IMPORTED_LOCATION ${GGML_CUDA_LIB}
)

add_library(lz4-lib SHARED IMPORTED)
set_target_properties(lz4-lib PROPERTIES
    IMPORTED_LOCATION ${LZ4_LIB}
)


# 5. add test directory
add_subdirectory(test)
