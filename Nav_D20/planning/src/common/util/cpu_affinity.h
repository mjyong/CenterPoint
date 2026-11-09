#pragma once
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace planning::common::util {

inline bool SetCpuAffinity(const std::vector<int> &cores) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int core : cores) {
    CPU_SET(core, &cpuset);
  }
  pthread_t thread = pthread_self();
  if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
    return false;
  }
  return true;
}

inline bool SetCpuAffinity(int core) {
  return SetCpuAffinity(std::vector<int>{core});
}

inline bool SetRealtimePriority(int priority = 50) {
  struct sched_param param;
  param.sched_priority = priority; // 1-99, 越大优先级越高
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
    return false;
  }
  return true;
}

} // namespace planning::common::util