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

# 여기서 함수 호출만 하면 됨!
add_ggml_test_executable(test_runner test.cpp)
add_ggml_test_executable(test_graph graph.cpp)
add_ggml_test_executable(test_block block.cpp)
