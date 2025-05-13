# 공통 빌드 설정 함수
function(add_ggml_test_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_include_directories(${target_name} PRIVATE ${GGML_INCLUDE_DIR} ${GGML_SRC_DIR} ${GGML_CUDA_DIR} ${GGML_CPU_DIR})
    target_link_libraries(${target_name} 
        PRIVATE
        ggml
        ggml-base
        ggml-cpu
        ggml-cuda
        m
        cuda
        cublas
        Threads::Threads
    )
    set_target_properties(${target_name} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN/../lib"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test
    )
endfunction()

function(add_cuda_test_executable target_name)
    add_executable(${target_name} ${ARGN})
    target_include_directories(${target_name} PRIVATE)
    target_link_libraries(${target_name} 
        PRIVATE
        m
        cuda
        cublas
        Threads::Threads
        CUDA::cudart 
        CUDA::cublas
        ${NVML_LIB}
    )
    set_target_properties(${target_name} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN/../lib"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test
    )
endfunction()

function(add_lz4_executable target_name source_file)
    # 실행 파일 생성
    add_executable(${target_name} ${source_file})

    # include 디렉토리 설정 (LZ4용)
    target_include_directories(${target_name} PRIVATE ${LZ4_INCLUDE_DIR} ${JSON_INCLUDE_DIR})

    # 필요한 라이브러리 링크
    target_link_libraries(${target_name}
        PRIVATE
        m                 # 수학 라이브러리 (필요시)
        Threads::Threads # 스레드 (필요시)
        lz4-lib          # lz4 라이브러리
        OpenMP::OpenMP_CXX
    )

    # 실행 파일 출력 디렉터리 설정
    set_target_properties(${target_name} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN/../lib"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test
    )
endfunction()

# 여기서 함수 호출만 하면 됨!
add_ggml_test_executable(test_runner test.cpp)
add_ggml_test_executable(test_graph graph.cpp)
add_ggml_test_executable(test_block block.cpp)
add_lz4_executable(test_lz4 lz4.cpp)
add_lz4_executable(test_csr csr.cpp)

add_lz4_executable(test_cache cacheline.cpp)
add_cuda_test_executable(test_gpu_cache cache-gpu.cu)
add_cuda_test_executable(test_warp warp.cu)

add_cuda_test_executable(memcpy memory/memcpy.cpp)
add_cuda_test_executable(direct memory/direct.cpp)


add_cuda_test_executable(gpu-test gpu/gpu-test.cpp gpu/gpu-bench.cpp)
