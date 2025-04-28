cmake_minimum_required(VERSION 3.14)
project(ggml_demo C CXX)

set(CMAKE_CUDA_ARCHITECTURES 86)

# CUDA 필수
enable_language(CUDA)

# ggml 경로 (상대 경로 또는 절대 경로 모두 가능)
set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml)
set(GGML_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml/src/ggml-cuda)
set(GGML_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml/include)


# 소스 파일 지정
set(SOURCES
    test.cpp
)

# 실행 파일 생성
add_executable(ggml_demo ${SOURCES})

# include 경로 추가
target_include_directories(ggml_demo PRIVATE ${GGML_DIR})
target_include_directories(ggml_demo PRIVATE ${GGML_INCLUDE_DIR})

# ggml에서 CUDA를 쓰기 때문에 math와 cuda 관련 라이브러리 필요
target_link_libraries(ggml_demo PRIVATE m cuda cublas)

# find and link pthread properly
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(ggml_demo PRIVATE Threads::Threads)

# (optionally) force ggml to build only the core with your preferred flags:
set(GGML_STANDALONE OFF      CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF    CACHE BOOL "" FORCE)   # or ON if you want .so
set(GGML_BUILD_TESTS OFF     CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF  CACHE BOOL "" FORCE)
# ... any other GGML_XXX options before you add_subdirectory()

# Tell CMake where to find ggml
add_subdirectory(${GGML_DIR} EXCLUDE_FROM_ALL)

# Your executable:
target_link_libraries(ggml_demo PRIVATE ggml)
