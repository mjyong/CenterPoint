/*
 * @Author: eryuan.liu eryuan.liu@seres.cn
 * @Date: 2026-08-24 00:00:00
 * @LastEditors: eryuan.liu eryuan.liu@seres.cn
 * @LastEditTime: 2026-08-24 00:00:00
 */
#pragma once
#include <cstdint>

namespace planning {

struct BatteryState {
  uint8_t charge_state = 0;      // 1=充电, 2=放电, 4=充满, 0=未知
  double rsoc = 0.0;             // 保留电量百分比 %
  double voltage = 0.0;          // 总电压 V
  double current = 0.0;          // 电流 A (充电为负, 放电为正)
  double timestamp = 0.0;
  bool valid = false;
};

struct MotionState {
  uint32_t motion = 0;           // 当前执行的 motion ID (MOTION_*)
  uint8_t state = 0;             // 1=REJECTED, 2=DOING, 3=DONE, 4=ABORTED
  double timestamp = 0.0;
  bool valid = false;
};

} // namespace planning
