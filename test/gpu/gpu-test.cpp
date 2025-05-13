// gpu_concurrent_bench.cpp

#include "gpu-utils.h"


int main(){
  // 1) NVML 초기화
  CHECK_NVML(nvmlInit());

  // 2) GPU 개수 확인
  int ngpu = 0;
  CHECK_CUDA(cudaGetDeviceCount(&ngpu));
  std::cout << "Detected " << ngpu << " GPU(s)\n\n";

  // 3) 샘플링 스레드 시작
  std::atomic<bool> run_flag{true};
  std::vector<std::thread> util_threads;
  for(int i = 0; i < ngpu; i++){
      util_threads.emplace_back(sampleUtilization, i, std::ref(run_flag));
  }

  // 4) 동시 전송 벤치마크 (로그 없이 시간만 수집)
  std::vector<std::pair<double,double>> results;
  std::vector<size_t> sizes = {1<<10, 1<<20, 10*(1<<20)};
  for(auto sz : sizes){
      double t1 = runConcurrent(sz, false);  // 1 GPU
      double t2 = runConcurrent(sz, true);   // 2 GPUs
      results.emplace_back(t1, t2);
  }

  // 5) 샘플링 중지 및 join
  run_flag = false;
  for(auto &t : util_threads) t.join();

  std::cout << "Number of threads: " << std::thread::hardware_concurrency() << "\n";

  // 6) 결과 테이블 출력
  std::cout << std::left << std::setw(12) << "Size"
            << std::right << std::setw(15) << "1GPU_time(s)"
            << std::setw(15) << "2GPU_time(s)\n";
  for(size_t i = 0; i < sizes.size(); i++){
      std::string s = std::to_string(sizes[i]/1024) + " KiB";
      std::cout << std::left << std::setw(12) << s
                << std::right << std::fixed << std::setprecision(6)
                << std::setw(15) << results[i].first
                << std::setw(15) << results[i].second
                << "\n";
  }

  // 7) NVML 종료
  CHECK_NVML(nvmlShutdown());
  return 0;
}