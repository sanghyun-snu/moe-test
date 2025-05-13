struct Buffer {
    void* ptr;
    size_t size;
    // host↔device 위치 정보 추적용 플래그...
  };
  
struct Task {
    int      id;
    vector<int> deps;          // 선행 Task ID 리스트
    Buffer* input;
    Buffer* output;
    double   cost_cpu;         // ms 단위 예상 비용
    double   cost_gpu;
    virtual void run_cpu() = 0;
    virtual void run_gpu(cudaStream_t s) = 0;
};

struct Resource {
    enum Type { CPU, GPU } type;
    int        device_id;      // GPU면 0,1,…; CPU면 -1
    cudaStream_t stream;       // GPU 전용
    thread_pool* pool;         // CPU 전용
};
  