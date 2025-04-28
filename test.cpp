// example_matmul_plan.c
#include <stdio.h>
#include <stdlib.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

int _main() {
    // 1) 컨텍스트 생성
    ggml_init_params ip = {
        .mem_size   = 16ull * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(ip);

    // 2) 2×2 텐서 A, B 생성 및 초기화
    const int N = 2;
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);
    struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);
    float * pA = (float *)A->data;
    float * pB = (float *)B->data;
    // A = [1 2; 3 4], B = [5 6; 7 8]
    pA[0]=1; pA[1]=2; pA[2]=3; pA[3]=4;
    pB[0]=5; pB[1]=6; pB[2]=7; pB[3]=8;

    // (선택) A,B를 파라미터로 표시
    ggml_set_f32(A, 0);  // ggml_set_* 로도 파라미터 지정 가능
    ggml_set_f32(B, 0);

    // 3) C = A·B 연산 추가
    struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);

    // 4) 실행 그래프(build) 생성
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, C);

    // ============================================================
    // 방법 A) 직접 플랜 짜서 실행
    // ============================================================
    // 5a) 플랜 계산 (work_size, n_threads, threadpool 등 설정)
    struct ggml_cplan plan = ggml_graph_plan(
        graph,
        GGML_DEFAULT_N_THREADS,  // 또는 원하는 쓰레드 수
        NULL                     // 자체 스레드풀 사용
    );
    // 6a) work_data 버퍼 할당
    if (plan.work_size > 0) {
        plan.work_data = reinterpret_cast<uint8_t*>(std::malloc(plan.work_size));

    }
    // 7a) 그래프 실행
    if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed\n");
        return 1;
    }
    // 8a) work_data 해제
    free(plan.work_data);

    // ============================================================
    // 방법 B) 컨텍스트에 붙여서 한번에 실행
    // ============================================================
    // 5b~8b) ggml_graph_compute_with_ctx 내부에서 plan 생성부터 실행, 해제까지 알아서 처리
    if (ggml_graph_compute_with_ctx(ctx, graph, GGML_DEFAULT_N_THREADS) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute_with_ctx failed\n");
        return 1;
    }

    // 9) 결과 출력 (행렬 C)
    float * pC = (float *)C->data;
    printf("C = [ %f, %f ; %f, %f ]\n",
           pC[0], pC[1],
           pC[2], pC[3]);
    // 예상: [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8]
    //      = [19, 22; 43, 50]

    // 10) 정리
    ggml_free(ctx);
    return 0;
}

int main() {
    // 1) ggml 컨텍스트 생성 (no_alloc=true: tensor data는 나중에 backend에서 할당)
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(ip);

    // 2) 2D 텐서 생성: ne[0]=nx, ne[1]=ny
    const int nx = 2, ny = 3;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, ny);

    // 3) data 포인터에 직접 쓰기 (no backend here)
    //    a->nb[0] = sizeof(float)
    //    a->nb[1] = a->nb[0] * nx
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            size_t offset = y * a->nb[1] + x * a->nb[0];
            *(float *)((char *)a->data + offset) = (float)(x + y);
        }
    }

    // 4) flat 메모리 순서대로 출력
    printf("Flat memory contents:\n");
    for (int i = 0; i < nx*ny; i++) {
        float *p = &((float *)a->data)[i];
        printf("%6.2f ", *p);
    }
    printf("\n\n");

    // 5) (x,y) 인덱스 순서대로 출력
    printf("By (x,y) index:\n");
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            size_t offset = y * a->nb[1] + x * a->nb[0];
            float  v      = *(float *)((char *)a->data + offset);
            printf("%6.2f ", v);
        }
        printf("\n");
    }
    // a 는 ggml_tensor*
    enum ggml_type    type    = a->type;     // 텐서의 데이터 타입
    int64_t          ne0     = a->ne[0];     // 차원 0 크기 (가장 빠르게 변하는 축)
    int64_t          ne1     = a->ne[1];     // 차원 1 크기

    // 블록 크기 (type 당 sub-block 요소 수)
    int64_t blk_elems = ggml_blck_size(GGML_TYPE_Q4_0);     // 예: GGML_TYPE_F32 인 경우 내부 블록 크기

    // row_size: 한 “행”(fastest-varying 축의 전체 요소) 의 바이트 수
    size_t  row_bytes = ggml_row_size(type, ne0); // ne0=가로(열) 개수

    printf("block elems = %lld\n", (long long)blk_elems);
    printf("row bytes  = %zu\n", row_bytes);

    // 6) 정리
    ggml_free(ctx);
    return 0;
}