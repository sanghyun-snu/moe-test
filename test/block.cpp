#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

// #define GGML_MAX_DIMS 4
// #define GGML_MAX_SRC  4  // 실제 라이브러리에서 정의된 값에 맞춰주세요

int main() {
    // 1. Context 초기화
    struct ggml_init_params params = {
        .mem_size   = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 2. 입력 텐서 생성
    const int input_dim  = 4;
    const int output_dim = 3;
    const int M = 2, N = 3, K = 4; // 행렬 곱셈을 위한 차원

    struct ggml_tensor * A    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * B    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);

    // 값 채우기
    float * data_A    = ggml_get_data_f32(A);
    float * data_B    = ggml_get_data_f32(B);
    float * data_bias = ggml_get_data_f32(bias);
    for (int i = 0; i < input_dim; i++)    data_A[i]    = i + 1;
    for (int i = 0; i < input_dim * output_dim; i++) data_B[i] = 0.1f * (i + 1);
    for (int i = 0; i < output_dim; i++)   data_bias[i] = 0.5f;

    // 3. 연산 정의
    struct ggml_tensor * matmul     = ggml_mul_mat(ctx, B, A);
    printf("matmul result shape: \n");
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        printf("%lld ", (long long)matmul->ne[d]);
        if (d+1 < GGML_MAX_DIMS) printf(", ");
    }
    printf("\n\n");

    struct ggml_tensor * add_bias   = ggml_add    (ctx, matmul, bias);
    struct ggml_tensor * relu_out   = ggml_relu   (ctx, add_bias);

    // 4. 그래프 생성 및 구축
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, relu_out);

    // 5. 그래프 실행 준비
    struct ggml_cplan plan = ggml_graph_plan(graph, /*n_threads=*/0, /*threadpool=*/NULL);
    ggml_graph_compute(graph, &plan);

    // 6. 결과 출력
    printf("Result ReLU(A @ B + bias):\n");
    for (int i = 0; i < M*N; i++) {
        printf("%8.4f ", ggml_get_data_f32(relu_out)[i]);
    }
    printf("\n");

    // 7. Computation Graph 출력
    printf("\n=== Computation Graph ===\n");

    // 7-1) Leaf 텐서들
    printf("\n-- Leafs (%d) --\n", graph->n_leafs);
    for (int i = 0; i < graph->n_leafs; ++i) {
        struct ggml_tensor * leaf = graph->leafs[i];
        printf("Leaf %2d: %p | shape = (", i, (void*)leaf);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            printf("%lld", (long long)leaf->ne[d]);
            if (d+1 < GGML_MAX_DIMS) printf(", ");
        }
        printf(")\n");
    }

    // 7-2) Node 텐서들
    printf("\n-- Nodes (%d) --\n", graph->n_nodes);
    for (int i = 0; i < graph->n_nodes; ++i) {
        struct ggml_tensor * node = graph->nodes[i];
        printf("Node %2d: op = %-8s | shape = (", i, ggml_op_name(node->op));
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            printf("%lld", (long long)node->ne[d]);
            if (d+1 < GGML_MAX_DIMS) printf(", ");
        }
        printf(")\n");

        // 7-3) 이 노드가 사용한 입력(src)들
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            struct ggml_tensor * in = node->src[j];
            if (!in) break;  // NULL 포인터일 때 종료
            printf("    Input %d: %p | shape = (", j, (void*)in);
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                printf("%lld", (long long)in->ne[d]);
                if (d+1 < GGML_MAX_DIMS) printf(", ");
            }
            printf(")\n");
        }
    }

    // 8. 메모리 해제
    ggml_free(ctx);
    return 0;
}
