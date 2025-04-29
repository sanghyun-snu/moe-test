add_executable(main_app main.cpp)

target_include_directories(main_app PRIVATE ${GGML_INCLUDE_DIR})
target_link_libraries(main_app PRIVATE ggml m cuda cublas Threads::Threads)

# 실행파일 빌드 위치
set_target_properties(main_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/main
)
