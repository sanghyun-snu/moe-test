#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// 오류 체크 매크로
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                    \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

// 실험 설정
const size_t MAX_SIZE = 64 * 1024 * 1024;   // 최대 전송 크기 16 MB
const size_t MIN_SIZE = 1 * 1024;           // 최소 전송 크기 1 KB
const int    REPEATS  = 100;                // 반복 횟수

int main() {
    printf("chunk_size [KB], pageable BW [GB/s], pinned BW [GB/s]\n");

    // 디바이스 메모리 할당
    void* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, MAX_SIZE));

    // CUDA 이벤트 생성 (타이밍용)
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 실험할 청크 크기들: 1KB, 2KB, 4KB, ..., 최대 16MB
    for (size_t size = MIN_SIZE; size <= MAX_SIZE; size *= 2) {
        // 1) pageable 메모리
        void* h_page = std::malloc(size);

        // 타이밍 측정
        CUDA_CHECK(cudaEventRecord(start, 0));
        for (int i = 0; i < REPEATS; ++i) {
            // 동기 복사
            CUDA_CHECK(cudaMemcpy(d_buf, h_page, size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(h_page, d_buf, size, cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_page = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_page, start, stop));

        // 2) pinned 메모리
        void* h_pinned = nullptr;
        CUDA_CHECK(cudaHostAlloc(&h_pinned, size, cudaHostAllocDefault));

        CUDA_CHECK(cudaEventRecord(start, 0));
        for (int i = 0; i < REPEATS; ++i) {
            CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(h_pinned, d_buf, size, cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_pinned = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_pinned, start, stop));

        // 대역폭 계산 (왕복 포함: H2D + D2H)
        double gb_page   = (double)size * 2 * REPEATS / (ms_page / 1000) / 1e9;
        double gb_pinned = (double)size * 2 * REPEATS / (ms_pinned / 1000) / 1e9;

        printf("%10zu, %17.2f, %15.2f\n", size/1024, gb_page, gb_pinned);

        // 정리
        std::free(h_page);
        CUDA_CHECK(cudaFreeHost(h_pinned));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_buf));

    return 0;
}
