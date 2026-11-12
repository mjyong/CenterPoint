/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 19:51:06
 * @LastEditors: zhaotian tian.zhao@seres.cn
 * @LastEditTime: 2024-04-08 11:16:56
 */
#pragma once
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
namespace planning {
namespace common {
namespace util {
class TimeKeeper {
public:
  TimeKeeper(const std::string &title) : title_(title) {}
  void RecordTime(const std::string &name) {
    auto now = std::chrono::high_resolution_clock::now();
    auto nano_time_point =
        std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
    auto epoch = nano_time_point.time_since_epoch();
    uint64_t now_nano =
        std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();

    time_stamps_.emplace_back(now_nano);
    names_.emplace_back(name);
  }

  static double TimeNowSec() {
    auto now = std::chrono::high_resolution_clock::now();
    auto nano_time_point =
        std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
    auto epoch = nano_time_point.time_since_epoch();
    uint64_t now_nano =
        std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();
    return static_cast<double>(now_nano) / 1000000000UL;
  }

  std::string DumpString() const {
    std::stringstream ss;
    if (time_stamps_.size() <= 1) {
      ss << "time stamps size not enough!";
      return ss.str();
    }
    ss << "\n========Printing time for [" << title_ << "]========\n";
    for (size_t i = 0; i < time_stamps_.size() - 1; ++i) {
      ss << "--------[" << names_[i] << "] cost "
         << time_ms(time_stamps_[i], time_stamps_[i + 1]) << " ms.\n";
    }
    if (time_stamps_.size() > 2) {
      ss << "--------Total time cost: "
         << time_ms(time_stamps_.front(), time_stamps_.back()) << " ms.\n";
    }
    ss << "========End printing time for [" << title_ << "]========\n\n";
    return ss.str();
  }

  static std::string timeNowSecStr() {
    auto now = std::chrono::high_resolution_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm *localTime = std::localtime(&currentTime);
    std::ostringstream outputStream;
    outputStream << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");
    return outputStream.str();
  }

  void Clear() {
    time_stamps_.clear();
    names_.clear();
  }

private:
  static double time_s(const uint64_t &begin, const uint64_t &end) {
    return static_cast<double>(end - begin) / 1000000000UL;
  }

  static double time_ms(const uint64_t &begin, const uint64_t &end) {
    return static_cast<double>(end - begin) / 1000000UL;
  }

private:
  std::string title_;
  std::vector<std::string> names_;
  std::vector<uint64_t> time_stamps_;
};
} // namespace util
} // namespace common
} // namespace planning