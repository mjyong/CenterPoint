# 我们在 Nav_D20 上的改动

`Nav_D20/` 是机器人团队的代码（定位 lightning、感知、规划），我们在上面改了几处。**下次从机器人团队同步代码时，先看这一页**：

- 2026-11-12 那次同步（04d3a34）把下面大部分改动冲掉了，T47 复核时又重新打上；
- 能并进机器人团队那边最好，就不用每次重打。

每条写了改在哪、为什么、怎么验。重打时用 git 的三方合并：`git diff <改动提交>^ <改动提交> -- <文件> | git apply --3way`。冲突要手动看：
- 机器人团队把那段代码删了或注释了，就不用再改；
- 否则按"为什么"那一栏重新改。

| 改动 | 文件 | 为什么 | 怎么验 | 提交 |
|---|---|---|---|---|
| 高频预测的加速度按重力归一化 | `slam/lightning-lm-master/src/core/lio/laser_mapping.cc`（`ProcessIMU`） | kygi665 的 IMU 加速度单位是 g（静止时约 1.0），原来直接当 m/s² 用。两次激光定位之间，`/lightning/odom` 每秒多掉约 8.8 m/s²：激光定位 10 Hz 时每周期高度掉约 4 cm、速度偏约 0.9 m/s；慢下来时会一路掉（T41：z 到 −46 m） | `tools/check_planning_mcap.py` 不报 odom；T42 | 1eff593 |
| `/tf` 只发一棵树 | `planning/src/common/util/tf_buffer.{h,cc}`、`runnable/communication/d20_ptz_bridge/d20_ptz_bridge.cc`（捷径 `publish=false`） | 原来每条变换连逆变换一起发，云台还多一条 `map -> ptz_*`，一个坐标系有两个父、还成环，Foxglove、RViz 里来回跳 | 查包工具的 TF 检查；T41 | 78567f5 |
| 调试话题的时间戳和标记寿命 | `planning/src/runnable/communication/ros_debug_bridge/ros_debug_bridge.{h,cc}`（`DebugStamp`、DELETEALL、终点标记空坐标系不发）、`visualization_util.h`（寿命 0.5 s） | 局部代价地图原来用本机时钟打戳，回放旧 bag 时找不到那一刻的 TF；标记寿命 0.05 s，回放时一闪一闪 | 查包工具；T41 | 78567f5、1eff593 |
| 代价地图调试话题的方向；标记打时间戳 | `planning/src/runnable/communication/ros_debug_bridge/ros_debug_bridge.{h,cc}`（`toOccupancyGrid(..., "obstacle", 255.0, 0.0, ...)`，局部和全局两处；`StampMarkers`） | 代价地图的 obstacle 层是 255 = 空闲、0 = 障碍（`CostMapProduction::Proc`、`GlobalMapManager::LoadPGMMap`），原来按 (0, 1) 转：空闲全成了 100、障碍成了 0，Foxglove 里路显示成障碍（T47 两次录包里感知的 11.7 万个障碍格在代价地图里全是 0）。标记的时间戳原来是 0，挂 `base_link` 的（碰撞点）被 Foxglove 摆在第一帧 TF 的位置 | 查包工具不再报 `inverted`；`tests/dog/test_planning_viz.py` | 本次 |
| A* 起点挪到路上 | `planning/src/algorithm/grid_path_planner/grid_path_planner.{h,cc}`、`src/proto/grid_path_planner_conf.proto`（`allow_collision_start`、`collision_start_search_radius`）、D20 的 `preprocess_conf.pb.txt` 打开 | CJGC 是 1.7 m 宽的路网图，定位偏几分米起点就在路外，A* 报 `Start point is in collision` | T47：`Start adjusted`，进了 active | 4687a87、本次 |
| 人行横道 | `planning/third_party/osm_map/osm_parser.cc`（注册 `crosswalk`）、`runnable/preprocess/global_map_manager/global_map_manager.{h,cc}`（`crosswalks()`；`map.osm` 不在就用 `semantic.osm`） | 新的现场地图（ad68517c、a70c6db6）在 `2d/<id>/semantic.osm` 里标了人行横道，原来读不进来；门控在人行横道起作用（docs/PLANNING_INTEGRATION.md） | 切到新地图，日志 `GlobalMapManager: N crosswalks`；T48 | 本次 |
| OpenCV 5 兼容 | `grid_path_planner.cc`、`cost_map_production.cc`、`slam/lightning-lm-master/src/core/maps/tiled_map.cc` | 服务器的编译环境是 OpenCV 5，2D 几何函数挪到了 `opencv2/geometry/2d.hpp`；有版本判断，OpenCV 4 下不起作用 | 服务器上能编 | aff7108、本次 |
| 调试脚本 | `planning/script/run_planning_stack_debug.sh`（`--once` 时 bag 放完就收）、`replay_planning_check.sh`、`planning/tools/check_planning_mcap.py`、`planning/tools/planning_mcap_viz.py` 和布局 `planning/script/planning_viz.json`（录包转成 Foxglove 里不闪的版本）、`slam/lightning-lm-master/scripts/{loc_debug_config.sh,check_map_tilt.py}` | 本机回放、查包、看录包 | `tests/dog/test_planning_mcap.py`、`test_planning_viz.py`、`test_loc_map_tilt.py` | 多次 |
