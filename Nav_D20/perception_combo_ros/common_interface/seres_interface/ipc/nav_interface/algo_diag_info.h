#ifndef D10_NAV_ALGODIAG_INFO_H
#define D10_NAV_ALGODIAG_INFO_H
#include "common_interface/msg/header.h"
#include "common_interface/msg/time.h"
#include "utils/seres_string.h"
#include "utils/seres_vector.h"

#pragma pack(4)

namespace seres_ipc {
namespace nav_interface {
namespace msg {
// 故障码枚举
constexpr static uint32_t OK = 0;
// 定位错误
constexpr static uint32_t LOC_ODOMETRY_DIVERGENCE = 1;        // 里程计残差不收敛
constexpr static uint32_t LOC_TOO_FEW_MATCHES = 2;            // 匹配点数过少
constexpr static uint32_t LOC_LOW_MATCH_SCORE = 3;            // 匹配得分底
constexpr static uint32_t LOC_MAP_MATCH_SCORE_LOW = 4;        // 地图匹配得分太低
constexpr static uint32_t LOC_MAP_MISMATCH = 5;               // 地图误匹配
constexpr static uint32_t MAP_OPTIMIZATION_DIVERGENCE = 6;    // 优化不收敛
constexpr static uint32_t MAP_TOO_FEW_MATCHES = 7;            // 匹配点数过少
constexpr static uint32_t MAP_LOW_MATCH_SCORE = 8;            // 匹配得分低
constexpr static uint32_t MAP_LOOP_CLOSURE_FAULT = 9;         // 回环异常
constexpr static uint32_t MAP_BACKEND_OPTIMIZATION_FAIL = 10; // 后段优化不收敛
                                                              // 规控错误
constexpr static uint32_t PNC_LOC_TIMEOUT = 11;               // 定位信息超时
constexpr static uint32_t PNC_LOCAL_MAP_TIMEOUT = 12;         // 局部占用栅格地图超时
constexpr static uint32_t PNC_GLOBAL_MAP_LOAD_FAIL = 13;      // 全局地图加载失败
constexpr static uint32_t PNC_GLOBAL_PLAN_FAIL = 14;          // 全局路径规划失败
constexpr static uint32_t PNC_TASK_FILE_PARSE_FAIL = 15;      // 任务文件解析失败
constexpr static uint32_t PNC_TASK_INFO_PARSE_FAIL = 16;      // 任务信息解析失败
constexpr static uint32_t PNC_GIMBAL_TIMEOUT = 17;            // 云台状态信息超时
constexpr static uint32_t PNC_POSE_DEVIATION_LARGE = 18;      // 当前位置与全局参考路径偏差过大
constexpr static uint32_t PNC_LOCAL_REF_PATH_ERROR = 19;      // 局部参考路径提取错误
constexpr static uint32_t PNC_CONTROL_COMPUTE_FAIL = 20;      // 控制指令解算异常
constexpr static uint32_t PNC_LOCAL_PLAN_SOLVER_FAIL = 21;    // 局部规划求解失败
constexpr static uint32_t PNC_LOCAL_PLAN_STOP_BY_OBS = 22;    // 局部规划停障
constexpr static uint32_t PNC_EMERGENCY_STOP = 23;            // 因障碍栅格急停
                                                              // 传感器数据异常
constexpr static uint32_t SENSOR_DATA_ANOMALY_TIMESTAMP = 24; // 传感器时间戳异常
constexpr static uint32_t SENSOR_DATA_ANOMALY_FREQUENCY = 25; // 数据频率异常
constexpr static uint32_t SENSOR_DATA_ANOMALY_CONTENT = 26;   // 数据内容异常

struct DiagInfo {
  // 诊断故障码
  uint32_t diag_code;
  // 时间
  uint64_t time;
  // 描述
  Seres::string<32> brief;
};

struct AlgoDiagInfo {
  common_interface::msg::Header header;
  // 故障信息列表
  Seres::vector<DiagInfo, 256> diag_info;
};
} // namespace msg
} // namespace nav_interface
} // namespace seres_ipc

#pragma pack()
#endif