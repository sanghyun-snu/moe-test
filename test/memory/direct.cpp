#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err));                                       \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

int main() {
    const int dev0 = 0, dev1 = 1;
    const size_t N = 1<<26;                     // 예: 64M elements
    const size_t bytes = N * sizeof(float);

    // 1. P2P 지원 여부 확인
    int can01 = 0, can10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can01, dev0, dev1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can10, dev1, dev0));
    printf("GPU0→GPU1 P2P support = %d\n", can01);
    printf("GPU1→GPU0 P2P support = %d\n", can10);
    if (!can01 || !can10) {
        fprintf(stderr, "피어 액세스가 지원되지 않습니다.\n");
        return 1;
    }

    // 2. 피어 액세스 활성화
    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(dev1, 0));
    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(dev0, 0));

    // 3. 디바이스 메모리 할당
    float *d0 = nullptr, *d1 = nullptr;
    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaMalloc(&d0, bytes));
    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaMalloc(&d1, bytes));

    // 초기화 (GPU0 메모리에 1.0f 채우기)
    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaMemset(d0, 0, bytes));  // 예시: 0으로 초기화

    // 4. P2P 복사 (동기)
    CUDA_CHECK(cudaMemcpyPeer(d1, dev1, d0, dev0, bytes));

    // (원한다면 비동기로도)
    /*
    cudaStream_t s;
    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaStreamCreate(&s));
    CUDA_CHECK(cudaMemcpyPeerAsync(d1, dev1, d0, dev0, bytes, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaStreamDestroy(s));
    */

    printf("데이터가 GPU0 → GPU1으로 복사되었습니다.\n");

    // 정리
    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaFree(d0));
    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaFree(d1));

    return 0;
}
