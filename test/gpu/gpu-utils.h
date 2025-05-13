#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvml.h>

#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <atomic>

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)                    \
                << " (" << __FILE__ << ":" << __LINE__ << ")\n";                \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

#define CHECK_NVML(call)                                                        \
  do {                                                                          \
    nvmlReturn_t r = call;                                                      \
    if (r != NVML_SUCCESS) {                                                    \
      std::cerr << "NVML Error: " << nvmlErrorString(r)                         \
                << " (" << __FILE__ << ":" << __LINE__ << ")\n";                \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

extern void benchmarkTransfer(size_t size_bytes, int gpu_id);
extern double runConcurrent(size_t size_bytes, bool use_two_gpus);
extern void benchmarkTransferNoPrint(size_t size_bytes, int gpu_id);
extern void sampleUtilization(int gpu_id, std::atomic<bool>& run_flag);

#endif // GPU_UTILS_H
