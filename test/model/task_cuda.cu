#include "tasks.h"
#include <cuda_runtime.h>
#include <cmath>

// Constructor
Task::Task(int _id, const std::vector<int>& _deps, int _N,
           double cpuCost, double gpuCost)
    : id(_id), deps(_deps), cost_cpu(cpuCost), cost_gpu(gpuCost),
      N(_N), d_A(nullptr), d_B(nullptr), d_C(nullptr)
{
    h_A.resize(N);
    h_B.resize(N);
    h_C.resize(N);
    for(int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
}

void Task::run_cpu() {
    for(int i = 0; i < N; ++i) {
        h_C[i] = h_A[i] + h_B[i];
    }
}

__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) C[i] = A[i] + B[i];
}

void Task::run_gpu_async(int device, cudaStream_t stream) {
    cudaSetDevice(device);
    size_t bytes = N * sizeof(float);
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpyAsync(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice, stream);
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    vecAddKernel<<<blocks, threads, 0, stream>>>(d_A, d_B, d_C, N);
    cudaMemcpyAsync(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost, stream);
}

void Task::cleanup_gpu(int device) {
    cudaSetDevice(device);
    if(d_A) cudaFree(d_A);
    if(d_B) cudaFree(d_B);
    if(d_C) cudaFree(d_C);
    d_A = d_B = d_C = nullptr;
}

bool Task::verify() const {
    return std::fabs(h_C[0] - (h_A[0] + h_B[0])) < 1e-6f;
}
