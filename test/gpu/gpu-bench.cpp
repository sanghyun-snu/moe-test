#include "gpu-utils.h"


// ─────────────── H2D/D2H 전송 벤치마크 ───────────────
void benchmarkTransfer(size_t size_bytes, int gpu_id) {
    CHECK_CUDA(cudaSetDevice(gpu_id));

    // 호스트 버퍼 (pinned)
    void* h_buf = nullptr;
    CHECK_CUDA(cudaHostAlloc(&h_buf, size_bytes, cudaHostAllocDefault));

    // 디바이스 버퍼
    void* d_buf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buf, size_bytes));

    // 이벤트 생성
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int REPEATS = 100;
    float ms_h2d = 0.f, ms_d2h = 0.f;

    // H2D
    CHECK_CUDA(cudaEventRecord(start, 0));
    for(int i = 0; i < REPEATS; i++){
      CHECK_CUDA(cudaMemcpyAsync(d_buf, h_buf, size_bytes,
                                 cudaMemcpyHostToDevice, 0));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms_h2d, start, stop));

    // D2H
    CHECK_CUDA(cudaEventRecord(start, 0));
    for(int i = 0; i < REPEATS; i++){
      CHECK_CUDA(cudaMemcpyAsync(h_buf, d_buf, size_bytes,
                                 cudaMemcpyDeviceToHost, 0));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms_d2h, start, stop));

    double bw_h2d = size_bytes * REPEATS / (ms_h2d/1000.0) / 1e9;
    double bw_d2h = size_bytes * REPEATS / (ms_d2h/1000.0) / 1e9;

    std::cout
      << "[GPU " << gpu_id << "] "
      << size_bytes/1024 << " KiB : "
      << "H2D " << bw_h2d << " GB/s, "
      << "D2H " << bw_d2h << " GB/s\n";

    // 정리
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_buf));
}
// ─────────────── H2D/D2H 전송 벤치마크 (로그 없이 시간만 측정) ───────────────
void benchmarkTransferNoPrint(size_t size_bytes, int gpu_id) {
    CHECK_CUDA(cudaSetDevice(gpu_id));

    void* h_buf = nullptr;
    CHECK_CUDA(cudaHostAlloc(&h_buf, size_bytes, cudaHostAllocDefault));
    void* d_buf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buf, size_bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int REPEATS = 100;
    float ms_h2d = 0.f, ms_d2h = 0.f;

    // H2D
    CHECK_CUDA(cudaEventRecord(start, 0));
    for(int i = 0; i < REPEATS; i++){
      CHECK_CUDA(cudaMemcpyAsync(d_buf, h_buf, size_bytes,
                                 cudaMemcpyHostToDevice, 0));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms_h2d, start, stop));

    // D2H
    CHECK_CUDA(cudaEventRecord(start, 0));
    for(int i = 0; i < REPEATS; i++){
      CHECK_CUDA(cudaMemcpyAsync(h_buf, d_buf, size_bytes,
                                 cudaMemcpyDeviceToHost, 0));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms_d2h, start, stop));

    // cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_buf));
}
// ────────────── GPU 이용률 샘플러 (백그라운드) ──────────────
void sampleUtilization(int gpu_id, std::atomic<bool>& run_flag) {
    nvmlDevice_t dev;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(gpu_id, &dev));

    while(run_flag.load()) {
        nvmlUtilization_t util;
        CHECK_NVML(nvmlDeviceGetUtilizationRates(dev, &util));
        std::cout << "[GPU " << gpu_id << "] Util GPU:"
                  << util.gpu << "%, Mem:" << util.memory << "%\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
// ────────────── 동시 전송 시간 측정 ──────────────
double runConcurrent(size_t size_bytes, bool use_two_gpus) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    // GPU0
    threads.emplace_back(benchmarkTransferNoPrint, size_bytes, 0);
    // GPU1 (옵션)
    if (use_two_gpus) {
        threads.emplace_back(benchmarkTransferNoPrint, size_bytes, 1);
    }
    for (auto &t : threads) t.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}