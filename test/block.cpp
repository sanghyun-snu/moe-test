#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
int main() {
    // 1. Context 초기화
    struct ggml_init_params params = {
        .mem_size   = 64 * 1024 * 1024,  // 64MB 메모리 풀
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // 2. 입력 텐서 A, weight B, bias 텐서 생성
    const int input_dim = 4;
    const int output_dim = 3;

    struct ggml_tensor * A = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_dim);   // [input_dim]
    struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, output_dim); // [input_dim x output_dim]
    struct ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_dim); // [output_dim]

    // 3. 값 채우기
    float * data_A = ggml_get_data_f32(A);
    float * data_B = ggml_get_data_f32(B);
    float * data_bias = ggml_get_data_f32(bias);

    for (int i = 0; i < input_dim; i++) {
        data_A[i] = i + 1;   // 1, 2, 3, 4
    }
    for (int i = 0; i < input_dim * output_dim; i++) {
        data_B[i] = 0.1f * (i + 1); // 0.1, 0.2, ..., 1.2
    }
    for (int i = 0; i < output_dim; i++) {
        data_bias[i] = 0.5f; // bias = 0.5
    }

    // 4. 연산 정의
    struct ggml_tensor * matmul = ggml_mul_mat(ctx, B, A); // (output_dim x 1)
    struct ggml_tensor * add_bias = ggml_add(ctx, matmul, bias); // broadcasting
    struct ggml_tensor * relu_out = ggml_relu(ctx, add_bias);

    // 5. 그래프 생성 및 구축
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, relu_out);

    // 6. 그래프 실행
    
    // 계획 만들기
    struct ggml_cplan plan = ggml_graph_plan(graph, /*n_threads=*/0, /*threadpool=*/NULL);

    // 그래프 계산
    ggml_graph_compute(graph, &plan);

    // 7. 결과 출력
    printf("Result ReLU(A @ B + bias):\n");
    for (int i = 0; i < output_dim; i++) {
        printf("%8.4f ", ggml_get_data_f32(relu_out)[i]);
    }
    printf("\n");

    // 8. 그래프 구조 출력
    printf("\n=== Computation Graph ===\n");
    printf("Total nodes: %d\n", graph->n_nodes);
    for (int i = 0; i < graph->n_nodes; ++i) {
        struct ggml_tensor * node = graph->nodes[i];
        printf("Node %2d: op = %-8s | shape = (%lld)\n",
            i,
            ggml_op_name(node->op),
            node->ne[0]);
    }

    // 9. 메모리 해제
    ggml_free(ctx);

    return 0;
}
