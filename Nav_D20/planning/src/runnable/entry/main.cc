/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 09:14:41
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2025-01-10 14:02:52
 */
#include "src/common/util/config_helper.h"
#include "src/common/util/cpu_affinity.h"
#include "src/common/util/diagnostic.h"
#include "src/runnable/entry/planning_entry.h"
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
using planning::common::util::ConfigHelper;
int main(int argc, char *argv[]) {

  // planning::common::util::SetCpuAffinity({2, 3});
  // mlockall(MCL_CURRENT | MCL_FUTURE);
  // planning::common::util::SetRealtimePriority(60);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = false;
  FLAGS_stderrthreshold = 0;
  FLAGS_max_log_size = 200;
  FLAGS_log_dir = "/tmp";

  planning::PlanningEntry planning_entry;
  ConfigHelper::Instance();

  planning::common::util::Diagnostic::Instance()->Init(
      "planning", ConfigHelper::GetConfDir() + "/../../data/fault_info.csv");

  const std::string config_file =
      ConfigHelper::GetConfDir() + std::string("/planning_conf/");

  bool ret = false;
  planning::config::CommunicationConf communication_conf;
  ret = ConfigHelper::GetProtoFromFile(
      config_file + "communication_conf.pb.txt", &communication_conf);
  ACHECK(ret != false);

  planning::config::PreprocessConf preprocess_conf;
  ret = ConfigHelper::GetProtoFromFile(config_file + "preprocess_conf.pb.txt",
                                       &preprocess_conf);
  ACHECK(ret != false);

  planning::config::MissionConf mission_conf;
  ret = ConfigHelper::GetProtoFromFile(config_file + "mission_conf.pb.txt",
                                       &mission_conf);
  ACHECK(ret != false);

  planning_entry.Init(communication_conf, preprocess_conf, mission_conf);
  while (planning_entry.communication_entry_ptr()->ok()) {
    planning_entry.Proc();
  }
  return 0;
}
