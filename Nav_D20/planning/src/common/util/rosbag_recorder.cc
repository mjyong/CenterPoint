#include "src/common/util/rosbag_recorder.h"
#include "src/common/util/time_keeper.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <vector>

namespace planning {
namespace common {
namespace util {
namespace fs = std::filesystem;
std::unique_ptr<rosbag2_cpp::Writer> RosbagRecorder::writer_;
double RosbagRecorder::duration_;
double RosbagRecorder::prev_check_time_;
std::string RosbagRecorder::bag_file_name_;
std::string RosbagRecorder::bag_dir_;

static std::string DatetimeToString(time_t time) {
  tm *tm_ = localtime(&time);
  int year, month, day, hour, minute, second;
  year = tm_->tm_year + 1900;
  month = tm_->tm_mon + 1;
  day = tm_->tm_mday;
  hour = tm_->tm_hour;
  minute = tm_->tm_min;
  second = tm_->tm_sec;
  char yearStr[5], monthStr[3], dayStr[3], hourStr[3], minuteStr[3],
      secondStr[3];
  sprintf(yearStr, "%d", year);
  sprintf(monthStr, "%d", month);
  sprintf(dayStr, "%d", day);
  sprintf(hourStr, "%d", hour);
  sprintf(minuteStr, "%d", minute);
  if (minuteStr[1] == '\0') {
    minuteStr[2] = '\0';
    minuteStr[1] = minuteStr[0];
    minuteStr[0] = '0';
  }
  sprintf(secondStr, "%d", second);
  if (secondStr[1] == '\0') {
    secondStr[2] = '\0';
    secondStr[1] = secondStr[0];
    secondStr[0] = '0';
  }
  char s[20];
  sprintf(s, "%s-%s-%s-%s-%s_%s", yearStr, monthStr, dayStr, hourStr, minuteStr,
          secondStr);
  std::string str(s);
  return str;
}

RosbagRecorder::RosbagRecorder() {
  bag_dir_ = "/tmp/planning_" + DatetimeToString(time(nullptr));
  duration_ = 60.0;
  rcpputils::fs::remove_all(bag_dir_);
  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = bag_dir_;
  storage_options.storage_id = "sqlite3";
  storage_options.max_bagfile_duration = duration_;

  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";
  writer_ = std::make_unique<rosbag2_cpp::Writer>(
      std::make_unique<rosbag2_cpp::writers::SequentialWriter>());
  writer_->open(storage_options, converter_options);
  prev_check_time_ = TimeKeeper::TimeNowSec();
}
RosbagRecorder::~RosbagRecorder() {
  if (writer_) {
    writer_->close();
  }
}

void RosbagRecorder::RestrictCheck() {
  if (fabs(TimeKeeper::TimeNowSec() - prev_check_time_) > (duration_ + 2.0)) {
    prev_check_time_ = TimeKeeper::TimeNowSec();
    try {
      std::vector<fs::directory_entry> files;
      for (const auto &entry : fs::directory_iterator(bag_dir_)) {
        if (entry.is_regular_file()) {
          files.push_back(entry);
        }
      }
      if (files.size() <= 2) {
        return;
      }
      std::sort(files.begin(), files.end(),
                [](const fs::directory_entry &a, const fs::directory_entry &b) {
                  return fs::last_write_time(a) > fs::last_write_time(b);
                });
      for (size_t i = 2; i < files.size(); ++i) {
        try {
          fs::remove(files[i].path());
        } catch (const fs::filesystem_error &ex) {
          AERROR << "Failed to remove old rosbag files: " << ex.what();
        }
      }
    } catch (const fs::filesystem_error &ex) {
      AERROR << "Failed to remove old rosbag files: " << ex.what();
    }
  }
  return;
}

} // namespace util
} // namespace common
} // namespace planning