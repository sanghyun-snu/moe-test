#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>

class ThreadPool {
public:
    ThreadPool(size_t numThreads);
    ~ThreadPool();

    // 작업 추가
    template <class F>
    void enqueue(F&& f);

private:
    // 스레드가 작업을 처리할 함수
    void worker();

    std::vector<std::thread> workers;  // 스레드들
    std::queue<std::function<void()>> tasks;  // 대기 중인 작업들

    std::mutex queueMutex;  // 작업 큐를 보호하는 뮤텍스
    std::condition_variable condition;  // 작업이 있을 때까지 기다리기 위한 조건 변수
    std::atomic<bool> stop;  // 스레드 종료 플래그
};

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    // 스레드들 생성
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back(&ThreadPool::worker, this);
    }
}

ThreadPool::~ThreadPool() {
    stop = true;
    condition.notify_all();  // 스레드가 종료될 수 있도록 알림
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();  // 모든 스레드가 종료될 때까지 기다림
        }
    }
}

void ThreadPool::worker() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queueMutex);
            condition.wait(lock, [this] { return !tasks.empty() || stop; });  // 작업이 있거나 종료 플래그가 true일 때까지 기다림

            if (stop && tasks.empty()) {
                return;
            }

            task = std::move(tasks.front());  // 큐에서 작업 꺼내기
            tasks.pop();
        }

        task();  // 작업 실행
    }
}

template <class F>
void ThreadPool::enqueue(F&& f) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.push(std::forward<F>(f));  // 작업 큐에 추가
    }
    condition.notify_one();  // 하나의 스레드에게 작업을 할당하도록 알림
}

void exampleTask(int id) {
    std::cout << "Task " << id << " is being processed by thread " << std::this_thread::get_id() << "\n";
}

int main() {
    ThreadPool pool(4);  // 4개의 스레드를 가진 풀 생성

    // 작업 추가
    for (int i = 0; i < 10; ++i) {
        pool.enqueue([i] { exampleTask(i); });
    }

    return 0;
}
