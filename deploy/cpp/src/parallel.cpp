#include "dogperc/parallel.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace dogperc {

namespace {

// one job per run_chunks call: workers that wake late hold the old job and find its chunks taken
struct Job {
  const std::function<void(size_t)>* fn;
  size_t chunks;
  std::atomic<size_t> next{0}, done{0};
};

class Pool {
 public:
  ~Pool() { resize(0); }

  void resize(int workers) {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
    workers_.clear();
    stop_ = false;
    for (int i = 0; i < workers; ++i) workers_.emplace_back([this] { loop(); });
  }
  int workers() const { return static_cast<int>(workers_.size()); }

  void run(size_t chunks, const std::function<void(size_t)>& fn) {
    auto job = std::make_shared<Job>();
    job->fn = &fn;
    job->chunks = chunks;
    {
      std::lock_guard<std::mutex> lk(m_);
      job_ = job;
      ++gen_;
    }
    cv_.notify_all();
    work(*job);
    std::unique_lock<std::mutex> lk(m_);
    done_cv_.wait(lk, [&] { return job->done.load() == job->chunks; });
    job_.reset();
  }

 private:
  void work(Job& j) {
    for (;;) {
      const size_t c = j.next++;
      if (c >= j.chunks) return;
      (*j.fn)(c);
      if (++j.done == j.chunks) {
        std::lock_guard<std::mutex> lk(m_);
        done_cv_.notify_all();
      }
    }
  }
  void loop() {
    uint64_t seen = 0;
    {
      std::lock_guard<std::mutex> lk(m_);
      seen = gen_;
    }
    for (;;) {
      std::shared_ptr<Job> j;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
        if (stop_) return;
        seen = gen_;
        j = job_;
      }
      if (j) work(*j);
    }
  }

  std::mutex m_;
  std::condition_variable cv_, done_cv_;
  std::vector<std::thread> workers_;
  std::shared_ptr<Job> job_;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

std::atomic<int> g_threads{1};
std::mutex g_pool_mutex;          // one run_chunks at a time, and no resize under a run

Pool& pool() {
  static Pool p;
  return p;
}

}  // namespace

void set_num_threads(int n) { g_threads = std::max(1, n); }

int num_threads() { return g_threads.load(); }

namespace detail {

void run_chunks(size_t chunks, const std::function<void(size_t)>& fn) {
  std::lock_guard<std::mutex> lk(g_pool_mutex);
  Pool& p = pool();
  if (p.workers() != num_threads() - 1) p.resize(num_threads() - 1);
  p.run(chunks, fn);
}

}  // namespace detail

}  // namespace dogperc
