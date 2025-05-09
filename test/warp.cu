#include <stdio.h>
#include <cuda_runtime.h>

__global__ void warpReducePrint() {
    int laneId = threadIdx.x & 0x1f;
    // Seed starting value as inverse lane ID
    int value = 31 - laneId;

    // Print 초기값
    printf("Thread %2d start value = %2d\n", threadIdx.x, value);

    // Use XOR mode to perform butterfly reduction
    for (int offset = 16; offset >= 1; offset >>= 1) {
        int other = __shfl_xor_sync(0xffffffff, value, offset, 32);
        value += other;
        // Print 각 단계별 내부 상태
        printf("  Thread %2d after offset %2d has value = %2d (added from lane %2d (%2d) ^ %2d = %2d (%2d)\n",
               threadIdx.x, offset, value, laneId, value-other, offset, laneId^offset, other);
    }

    // 최종값 출력 (lane 0만 출력해도 되지만, 확인 차원에서 모두 출력)
    printf("Thread %2d final reduced value = %2d\n", threadIdx.x, value);
}

int main() {
    // 한 블록에 32 스레드만 실행
    warpReducePrint<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
