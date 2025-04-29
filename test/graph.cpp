#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include <stdio.h>

int main() {
    // 1. Context 초기화
    struct ggml_init_params params = {
        .mem_size   = 64 * 1024 * 1024,  // 64MB 메모리 풀
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // 2. 텐서 생성
    const int M = 2;
    const int K = 3;
    const int N = 4;

    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M); // [K x M]
    struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N); // [K x N]

    // 3. 값 채우기
    float * data_A = ggml_get_data_f32(A);
    float * data_B = ggml_get_data_f32(B);

    for (int i = 0; i < M*K; i++) {
        data_A[i] = i + 1;   // 1~6
    }
    for (int i = 0; i < K*N; i++) {
        data_B[i] = (i + 1) * 0.1f; // 0.1 ~ 1.2
    }

    // 4. 연산 정의 (C = A @ B)
    struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);

    // 5. 그래프 생성 및 구축
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, C);

    // 6. 그래프 실행
    int n_threads = 1; // 스레드 수
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads, /*threadpool=*/NULL);  // plan 짤 때 n_threads, threadpool 넘기기
    ggml_graph_compute(graph, &plan);                 // 그래프와 계획을 넘긴다

    // 7. 결과 출력
    printf("Result C (shape: %lld x %lld):\n", C->ne[1], C->ne[0]);
    for (int i = 0; i < C->ne[1]; i++) { // row
        for (int j = 0; j < C->ne[0]; j++) { // col
            printf("%8.4f ", ggml_get_data_f32(C)[i*C->ne[0] + j]);
        }
        printf("\n");
    }

    // 8. 내부 그래프 구조 출력
    printf("\n=== Computation Graph ===\n");
    printf("Total nodes: %d\n", graph->n_nodes);
    for (int i = 0; i < graph->n_nodes; ++i) {
        struct ggml_tensor * node = graph->nodes[i];
        printf("Node %2d: op = %-8s | shape = (%lld, %lld)\n",
            i,
            ggml_op_name(node->op),
            node->ne[0], node->ne[1]);
    }

    // 9. 메모리 해제
    ggml_free(ctx);

    return 0;
}
