#ifndef TASKS_H
#define TASKS_H

#include <vector>
#include <cuda_runtime.h>

struct Task {
    int id;
    std::vector<int> deps; 
    double cost_cpu;
    double cost_gpu;

    // Host-side buffers
    std::vector<float> h_A;
    std::vector<float> h_B;
    std::vector<float> h_C;
    // Device-side buffers
    float *d_A{nullptr};
    float *d_B{nullptr};
    float *d_C{nullptr};
    int    N;

    Task(int _id, const std::vector<int>& _deps, int _N,
         double cpuCost, double gpuCost);

    void run_cpu();
    void run_gpu_async(int device, cudaStream_t stream);
    void cleanup_gpu(int device);
    bool verify() const;
};

#endif // TASKS_H
