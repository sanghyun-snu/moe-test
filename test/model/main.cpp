#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <atomic>

#include "tasks.h"

struct Resource {
    enum Type { CPU, GPU } type;
    int device_id;       // GPU: >=0, CPU: -1
    cudaStream_t stream; // valid only for GPU
};

int main() {
    // 1) Define tasks
    const int VECTOR_SIZE = 1 << 20;
    std::vector<Task*> tasks = {
        new Task(0, {}, VECTOR_SIZE, 5.0, 1.0),
        new Task(1, {}, VECTOR_SIZE, 1.0, 5.0),
        new Task(2, {0,1}, VECTOR_SIZE, 3.0, 3.0)
    };
    const int T = tasks.size();

    // 2) Initialize resources
    std::vector<Resource> resources;
    unsigned int cpu_threads = std::max(1u, std::thread::hardware_concurrency() - 1);
    for(unsigned int i = 0; i < cpu_threads; ++i) {
        resources.push_back({Resource::CPU, -1, nullptr});
    }
    int ngpu = 0;
    cudaGetDeviceCount(&ngpu);
    for(int dev = 0; dev < ngpu; ++dev) {
        cudaStream_t s;
        cudaSetDevice(dev);
        cudaStreamCreate(&s);
        resources.push_back({Resource::GPU, dev, s});
    }

    // 3) Prepare synchronization
    std::mutex mtx;
    std::condition_variable cv;
    std::unordered_map<int,int> dep_cnt;
    std::atomic<int> done_count{0};
    std::queue<Task*> ready;

    for(auto t : tasks) {
        dep_cnt[t->id] = t->deps.size();
        if(dep_cnt[t->id] == 0) ready.push(t);
    }

    auto onTaskDone = [&](Task* t) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            ++done_count;
            for(auto u : tasks) {
                for(int d : u->deps) {
                    if(d == t->id && --dep_cnt[u->id] == 0) {
                        ready.push(u);
                    }
                }
            }
        }
        cv.notify_one();
    };

    // 4) Scheduling loop
    int next_gpu = cpu_threads;
    while(done_count < T) {
        Task* curr = nullptr;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&]{ return !ready.empty(); });
            curr = ready.front();
            ready.pop();
        }
        bool use_gpu = (curr->cost_gpu < curr->cost_cpu) && (ngpu > 0);
        int ridx = 0;
        if(use_gpu) {
            ridx = next_gpu;
            next_gpu = cpu_threads + ((next_gpu - cpu_threads + 1) % ngpu);
        } else {
            static unsigned int next_cpu = 0;
            ridx = next_cpu++ % cpu_threads;
        }
        Resource& R = resources[ridx];

        if(R.type == Resource::CPU) {
            std::thread([curr,&onTaskDone](){
                curr->run_cpu();
                onTaskDone(curr);
            }).detach();
        } else {
            int dev = R.device_id;
            curr->run_gpu_async(dev, R.stream);
            cudaEvent_t ev;
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            cudaEventRecord(ev, R.stream);
            std::thread([curr,ev,dev,&onTaskDone](){
                cudaSetDevice(dev);
                cudaEventSynchronize(ev);
                curr->cleanup_gpu(dev);
                onTaskDone(curr);
                cudaEventDestroy(ev);
            }).detach();
        }
    }

    // 5) Cleanup resources
    for(unsigned int i = cpu_threads; i < resources.size(); ++i) {
        cudaStreamDestroy(resources[i].stream);
    }

    // 6) Verification & free
    bool ok = true;
    for(auto t : tasks) {
        ok &= t->verify();
        delete t;
    }
    std::cout << "All tasks done. Verification: " << (ok ? "SUCCESS" : "FAIL") << std::endl;
    return 0;
}
