# `20261115_x6_all_1` 在线测试分析看板与复盘归档

对应包：`/public/zjst/robot/cp_data/20261116_dog/20261115_x6_all_1`（地图 `ad68517c` 图上 **6 号人行横道** / 规划 `crosswalk 0`，全长 `394.1 s`）。

## 看板文件说明

- **`x6-online-test-analysis.canvas.tsx`**：Cursor Canvas 原生交互式看板源码（可拷入 `~/.cursor/projects/<workspace>/canvases/` 在 Cursor 侧栏直接渲染并一键跳转源码）。
- **`x6_online_dashboard.html`**：零依赖独立网页版交互式看板（双击或浏览器直接打开即可查看总览图表、61 条航迹分类筛选、五阶段规划复盘与 Foxglove 指南）。
- **`planning_viz_overview.png`**：`planning_mcap_viz.py` 生成的全局路径与状态时间序列总览图。
- **`snap_t064.0s.jpg` / `snap_t305.5s.jpg` / `snap_t353.0s.jpg` / `snap_t385.0s.jpg`**：四路鱼眼相机 + BEV 激光雷达与在线 3D 框同步投影关键帧。

## 核心结论与对应代码修复

1. **3D 点云车辆识别（`Nav_D20/perception_combo_ros/node_src/lidar_3d_detection`）**：
   - `10.00 Hz` 稳定零丢帧（`3,941` 帧），端到端耗时中位数 `26.5 ms`（`p99 = 39.7 ms`）。
   - 主路口近中距（`3–35 m`）横穿真车 **14 辆次 100% 检出**；仅包末尾右前方 `19–41 m` 慢速侧向接近大车（`ID 1368 -> 1374`）发生掉分断轨，另狗正后方约 `76°` 存在机身/操作员遮挡盲区（`ID 1356 -> 1365`）。
   - 帧级疑似误检占比 **`5.4%`**（`468 / 8,603` 框·帧），集中在 `34–52 m` 建筑立面/绿化带边缘瞬态闪框；3 辆停放重车因过往大车遮挡超过 `max_age = 1.5 s` 裂变为 `18` 个 ID。
2. **路口让车规划（`Nav_D20/planning`）**：
   - 真车横穿期间 `wait` / `cross_wait` 门控响应正常。
   - **修复状态机边界死区 Bug**（`crosswalk_yield.cc` + `crosswalk_sim.cc` `RunEdgeJitter`）：狗压入口线 `0.3–7.2 cm` 站立时，旧逻辑一进 `cross` 即记 `passed_.insert(0)`，`t = 352.0 s` 里程计微抖几厘米出界 3 帧再抖回界内 `7 mm` 后因 `passed_.count(0)` 跳过该横道，导致最后 `42 s` 锁死在 `none` 不再让车；现改为走过 `max_unreleased_depth` 或距出口 `<= 0.5 m` 才记入 `passed_`，且浅层 `unreleased` 同步 `passed_.erase`。
   - **补齐整包 MCAP 转换**（`bag_on_map.py` + `bag_to_mcap.py`）：注册 `foxglove_msgs/msg/CompressedVideo` 与 `hesai_ros_driver/msg/LossPacket` 并支持 `-o/--output`，整包 23 个非空话题 `282,505` 条消息 `0 LEFT OUT`。
