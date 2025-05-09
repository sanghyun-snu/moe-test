#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE (1024 * 1024 * 32)
#define THREADS_PER_BLOCK 256

__global__ void stride_access_kernel(char* data, int stride, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx * stride; i < size; i += stride * gridDim.x * blockDim.x) {
        data[i]++;
    }
}

double measure_stride_time(char* d_array, int stride, int size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stride_access_kernel<<<128, THREADS_PER_BLOCK>>>(d_array, stride, size);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms * 1e6;  // convert to ns
}

int main() {
    char* d_array;
    cudaMalloc(&d_array, ARRAY_SIZE * sizeof(char));
    cudaMemset(d_array, 0, ARRAY_SIZE);

    for (int stride = 1; stride <= 1024; stride *= 2) {
        double time_ns = measure_stride_time(d_array, stride, ARRAY_SIZE);
        printf("Stride: %d, Time: %.0f ns\n", stride, time_ns);
    }

    cudaFree(d_array);
    return 0;
}
