# D20 定位（lightning）入门

2026-11-12 重写，合并了原来的 `OVERVIEW.md`。对应 `Nav_D20/slam/lightning-lm-master`（机器人团队狗上在跑的版本，04d3a34 同步）。规划见 `Nav_D20/planning/docs/ONBOARDING.md`；我们打的补丁在 `Nav_D20/PATCHES.md`。仓库根目录的 `docs/LOCALIZATION.md` 讲的是我们自己做训练数据用的离线位姿融合，不是这个。

先看什么：
1. 第 0、1 节：一帧 IMU、一帧点云分别经过哪些代码；
2. 在 Foxglove 里打开回放转出来的 `viz/planning_viz.mcap`，用布局 `Nav_D20/planning/script/loc_viz.json`（第 10 节）："定位流程"面板就是第 1 节那张表，每一刻填着当时的数；
3. 地图的目录结构（第 3 节），定位和规划各用里面哪个文件。

## 0. 一页看懂

`lightning-lm-master` 是高翔开源的 [Lightning-LM](https://github.com/gaoxiang12/lightning-lm) 的一份拷贝（不是 submodule）。三层叠在一起：

| 层 | 是什么 |
|---|---|
| FasterLIO | `src/core/lio`：激光 + IMU 紧耦合里程计（ESKF、IVox），前端 |
| Lightning-LM | 分块地图、回环、g2p5 栅格、miao 图优化、在线 / 离线入口 |
| D20 改动 | `config/default_d20.yaml`、kygi665 组合导航、地图模板（`SaveMapByTemplate`）、`mode_switch_node.cc`（2026-02-28）、健康上报 |

定位模式下不建图，把每帧点云配准到**事先建好的地图**上；RTK 固定解的 INS 当观测融进去。

```text
IMU  /rte/rdap/rtk/imus/data_raw（kygi665，100 Hz；加速度单位 g，角速度度/秒）
  └─► ESKF 预测 ──────────────────────────────► /lightning/odom（100 Hz，base_link 在 map 里）+ TF
顶激光 /robot/topic/rdap/top_point_cloud2（禾赛 XT32，10 Hz）
  └─► 预处理 → 去畸变 → 降采样 → 和地图的 IVox 点到面配准（IEKF 更新）
         │                        ▲ 地图按位姿分块加载（3d/*.pcd）
         ├─► 健康检查：匹配点太少 / 残差太大，连续 10 s → 自动重初始化
         └─► /lightning/current_scan_world（这一帧点云放进 map）
INS  /rte/rdap/rtk/inspvax（kygi665 INSPVAX）
  └─► 经纬高 → map（ENU，原点在配置里）→ RTK 固定解才融进 ESKF；初始化也用它
```

进程：仓库的 `Nav_D20/README.md` 写的是 `run_loc_online`；8 号狗上实际在跑的是 `mode_switch_node`（T47：它一个进程占 517% CPU），它启动后先定位，再按 Action 切建图、重定位，听 `/map/switch`（第 7 节）。两者里面的定位都是同一条链：

```text
run_loc_online / mode_switch_node
  -> LocSystem       src/core/system/loc_system.cc：ROS 外壳，订激光 / IMU / INS，发里程计，初始位姿，栅格；节点名 lightning_slam
  -> Localization    src/core/localization/localization.cpp：点云交给 LIO、INS 送进 ESKF、按 IMU 频率发里程计
  -> LaserMapping    src/core/lio/laser_mapping.cc：去畸变、IMU 预测、点到面配准、健康检查
  -> TiledMap + IVox src/core/maps/tiled_map.cc、src/core/ivox3d/：按位姿加载附近的地图块当局部地图
```

IMU、激光、INS 各一个回调组，三线程执行器并行。

定位路径上**不跑**的代码（读代码时可以先跳过）：

| 代码 | 什么时候才跑 |
|---|---|
| `SlamSystem`、`loop_closing`、`g2p5`、`miao` | 建图（`run_slam_online` 或 `/nav/mapping`） |
| `super_lio_frontend` | `use_super_lio: true` 才换前端，D20 是 false |
| `localization/lidar_loc/`、`pose_graph/` | 上游的"NDT 全局匹配 + 位姿图"定位，`localization.cpp` 没有调用 |
| `PoseSmoother` 的更新、`SetLidarOdomCallback` | 注释掉了 |
| `run_*_offline`、`split_map` | 离线工具 |

## 1. 一帧怎么跑

| 步 | 做什么 | 代码 | 可视化里看哪 |
|---|---|---|---|
| ① IMU 预测 | 每来一帧 IMU（角速度从度/秒换成弧度/秒）：`LaserMapping::ProcessIMU` 用 ESKF 往前推一步；`Localization::ProcessIMUMsg` 取预测状态（`GetIMUState`），用 `T_body_imu` 换到 `base_link`，**每帧 IMU 发一次** `/lightning/odom` 和 TF——所以是 100 Hz，两帧点云之间全靠 IMU 推；激光配准成功以后才开始发 | `loc_system.cc`、`localization.cpp`、`laser_mapping.cc`、`eskf.cc` | 流程面板 ①；高度、横滚、俯仰曲线 |
| ② 点云预处理 | `ProcessLidarMsg` → `pointcloud_preprocess.cc`（`lidar_type: 4` 按 XT32 解析，盲区 0.5 m，每 6 个点取 1 个）→ 放进队列 | `localization.cpp`、`pointcloud_preprocess.cc` | |
| ③ LIO 配准 | `LaserMapping::Run`：`SyncPackages`（一帧点云 + 这段时间的 IMU）→ `ImuProcess::Process`（去畸变）→ 0.5 m 体素降采样 → 离上次加载地图的地方超过 100 m 就按位姿加载地图块、异步重建 IVox → `kf_.Update(LIDAR)`：点到面的迭代 EKF，最多 4 次 | `laser_mapping.cc`、`imu_processing.hpp`、`tiled_map.cc`、`ivox3d/` | 当前帧点云按"离地图多远"着色 |
| ④ 健康 | 匹配上的点 `effect_num` < 100 或平均残差 > 0.15 m 算一帧失败（日志"定位质量下降"）；连续 100 帧（约 10 s）→"定位故障"，下一帧自动重初始化：缓冲里有 RTK 固定解就用最新 INS 位姿，否则退回上次好的位姿再做 NDT | `LaserMapping::Run` 末尾、`Localization::LidarOdomProcCloud` | 事件；"定位有效"状态条 |
| ⑤ INS 融合 | 找时间最近的 INS，RTK 固定解（`ins_status` 3 且 `pos_type` 56）才当观测融进 ESKF（`ProcessINS`） | `Localization::LidarOdomProcCloud` | INS 轨迹（紫 = RTK 固定）、"定位减 INS"曲线 |
| ⑥ 输出 | `/lightning/current_scan_world`（有订阅者才发）；`/lightning/odom` 的 `covariance[0]` 挪作发布时的系统时间、`covariance[1]` = 定位有效（1 有效、0 故障）；每秒一条 `/nav/location`、一行 `/tmp/lightning_trajectory.txt`；健康事件发 `/rhr/topic/event_info` | `loc_system.cc`、`health_reporter.h` | |

## 2. 启动和初始化

1. 读配置（环境变量 `config_path` 指的 yaml，D20 是 `config/default_d20.yaml`）和地图；
2. 等初始位姿（`LocSystem` 的 INS 回调，`with_ins: true`）：
   - INS 是 RTK 固定解：直接用 INS 位姿，**跳过 NDT**；
   - RTK 浮点解：用 INS 位姿，再做 NDT；
   - 等满 30 s（`INS_INIT_TIMEOUT_SECONDS`）还没有：用配置的 `system.init_t` / `init_r`；
3. `LaserMapping::Run` 第一次：按初始位姿加载附近的地图块（附近没有块就用地图的起点功能点），建 IVox；需要时做多级 NDT 偏航搜索（`YawSearch`）精化位姿；
4. 接着 IMU 初始化（`fasterlio::INIT_TIME`），然后进入第 1 节的循环。

INS 直出（`system.ins_only_mode: true`，本机调试用）：不跑 LIO，把 INS 位姿直接发到 `/lightning/odom`；连续 3 s 没有固定解按丢失处理。0921 这包全程 RTK 固定解，用它可以把规划的调试和激光定位分开。

## 3. 地图

一张现场地图是一个目录（`Nav_D20/map/<uuid>.zip` 解开；狗上在 `/opt/seres/nav/data/<uuid>`）。现在建图写出的模板：

```text
<uuid>/
  global_map.yaml               地图编号、楼层；每层的 map_image_file、map_yaml_file、关键帧
  2d/<起始时间戳>/map.yaml       分辨率 0.1 m、原点、阈值
  2d/<起始时间戳>/map.png        2D 栅格（ad68517c 还有 map.pgm）；路网图，255 = 能走
  2d/<起始时间戳>/semantic.osm   语义层：人行横道等（新地图才有）
  3d/global.pcd                 整张点云（ad68517c：474 万点）
  3d/index.txt                  分块索引：定位加载用这个，没有就不算有效地图
  3d/<块>.pcd                   分块点云
  session/keyframes/<id>.pcd|yaml、manifest.yaml、factor_graph.yaml   建图的关键帧和因子图
```

旧布局是根目录直接放 `global.pcd`、`index.txt`、`map.yaml`、`map.png` / `map.pgm`，两种都认。

| 文件 | 谁用 |
|---|---|
| `3d/index.txt` + `3d/<块>.pcd` | 定位配准（`TiledMap` 按位姿加载，`maps.load_map_size` 2） |
| `3d/global.pcd` | 定位启动时降到 0.5 m 体素发 `/lightning/map_cloud`（3 次）；可视化的地图点云 |
| `2d/<id>/map.png` + `map.yaml` | 定位发 `/map`；规划切图后读它当全局图 |
| `2d/<id>/semantic.osm` | 规划（`GlobalMapManager::crosswalks()`） |

- 规划默认不读这里：它用自己的 `robot_factory/data/astar_pgm/CJGC.pgm`，收到 `/map/switch` 才换成这里的 `map.png`。ad68517c 的路网比 CJGC 多人行横道 6 往西那段。
- 配置的 `system.map_path` 没有有效索引时（本机回放时 `/opt/seres/...` 不存在），回退到配置上一级的 `map/0c3f8a2d-...` 并打日志，`/nav/location` 报地图失败。**这张图不能用来定位**：它是用 0921 这包的 15 s 直行建的，对齐 INS 后整张图倾斜约 51°（`python3 scripts/check_map_tilt.py <地图目录>` 查得出来）。本机用 `LOC_MAP_PATH=<现场地图目录>`。
- g2p5 只把地面（−0.7 m）以上 0.5–1.0 m 的点投进 0.2 m 的栅格，假设雷达水平；栅格只给导航显示和存盘，不参与定位。

## 4. 坐标系和外参

- `map`：ENU，原点在配置的 `system.ins_lat`、`ins_lon`、`ins_height`（生效的是超级工厂那组：29.7314°N、106.7839°E、217.2 m；五云湖那组注释掉了）。INS 的经纬高用 GeographicLib `LocalCartesian` 换过来，朝向取 `-azimuth`（度）绕 z 轴（novatel 是 `90 - azimuth`）。
- `base_link`：狗身。`system.extrinsic_T/R` 是 IMU 到 `base_link`：IMU 在 (0.139, 0.021, 0.134) m，绕 z 转 −90°（D20 A2）。
- 雷达到 IMU：`fasterlio.extrinsic_T/R`：(−0.021, 0.18, 0.009) m，绕 z 转 180°。
- **kygi665 的加速度单位是 g**（静止时约 1.0）。高频预测原来当 m/s² 用，两次配准之间每秒多掉 8.8 m/s²，T41 回放时 z 掉到 −46 m。我们按重力归一化了（`laser_mapping.cc` 的 `ProcessIMU`，T42 验证）。

## 5. 话题

| 方向 | 话题 / Action / 服务 | 类型 | 说明 |
|---|---|---|---|
| 订 | `/robot/topic/rdap/top_point_cloud2` | `sensor_msgs/PointCloud2` | 顶激光（`common.lidar_topic`） |
| 订 | `/rte/rdap/rtk/imus/data_raw` | `sensor_msgs/Imu` | 加速度 g，角速度度/秒 |
| 订 | `/rte/rdap/rtk/inspvax` | `kygi665_msgs/INSPVAX` | `common.kygi665_ins_topic`，`ins_type: kygi665` |
| 发 | `/lightning/odom` | `nav_msgs/Odometry` | IMU 频率；`covariance[0]` 系统时间、`[1]` 定位有效 |
| 发 | TF `map -> base_link` | | 和 odom 同一时刻（`pub_tf: true`） |
| 发 | `/map` | `nav_msgs/OccupancyGrid` | reliable + transient_local；读 `map.png`（`publish_occ_grid: true`） |
| 发 | `/lightning/map_cloud` | `PointCloud2` | 启动时 3 次（`publish_map_cloud: true`） |
| 发 | `/lightning/current_scan_world`、`current_scan_body` | `PointCloud2` | 当前帧，有订阅者才发；很大 |
| 发 | `/lightning/ins_path` | `nav_msgs/Path` | 最近 1000 个 INS 位姿、每条 INS 发一次：不要录 |
| 发 | `/nav/location` | `nav2_interface/Location` | 1 Hz：地图 UUID、map 位姿、经纬高（RTK 固定 / 浮点时状态 2，否则 −1）、速度；`twist.angular.x` 挪作系统时间 |
| 发 | `/rhr/topic/event_info`、`alarm_info` | `rhr_interface` | 定位丢失（去抖 1 s）、重定位失败（10 s）、地图匹配失败、重定位成功 |
| 订 | `/map/switch` | `std_msgs/String` | 仅 `mode_switch_node`（第 7 节） |
| Action | `/nav/mapping`、`/nav/relocation` | `nav2_interface/action` | 仅 `mode_switch_node` |
| 服务 | `/lightning/save_map` | `lightning/srv/SaveMap` | 仅 `run_slam_online` |

`with_ui: true` 时还有 RViz 用的 `/lightning/current_scan`、`/lightning/keyframes`、`/lightning/lio_path`、`/lightning/keyframe_path`；建图时有 `/lightning/ins_pose`。规划只用 `/lightning/odom` 和 TF；局部障碍是感知的 `/map/occupancy_grid`，和这里的全局 `/map` 不是一回事。

## 6. 配置要点（`config/default_d20.yaml`）

| 键 | 值 | 意思 |
|---|---|---|
| `system.map_path` | `/opt/seres/nav/data/ad68517c-...` | 地图目录 |
| `system.ins_lat/lon/height` | 超级工厂 | `map` 的原点 |
| `system.init_t/init_r` | | 没有 INS 时的初始位姿 |
| `system.with_ins`、`ins_only_mode` | true、false | 融 INS；INS 直出 |
| `system.use_super_lio` | false | 前端用 FasterLIO |
| `system.pub_tf`、`publish_map_cloud`、`publish_occ_grid` | true | 发 TF、地图点云、`/map` |
| `fasterlio.lidar_type`、`scan_line` | 4（禾赛）、32 | |
| `fasterlio.blind`、`point_filter_num` | 0.5 m、6 | 盲区、每 6 个点取 1 个 |
| `fasterlio.filter_size_scan/map`、`ivox_grid_resolution` | 0.5 m | 降采样和 IVox 格子 |
| `fasterlio.max_iteration` | 4 | IEKF 迭代 |
| `fasterlio.extrinsic_T/R`、`system.extrinsic_T/R` | | 雷达到 IMU、IMU 到 `base_link` |
| `maps.load_map_size` | 2 | 加载周围几块地图 |
| `loop_closing.*`、`g2p5.*` | | 建图用（第 7 节） |

健康阈值写在代码里（`laser_mapping.h`）：`min_effect_feat_` 100、`max_avg_residual_` 0.15 m、`max_fail_count_` 100、`map_update_dist_` 100 m。

## 7. 建图和模式切换

`mode_switch_node` 启动即定位；建图、定位、切图中（IDLE）三种模式互斥：

| 入口 | 做什么 |
|---|---|
| Action `/nav/mapping` | 停掉定位，建图。已经在建图、重定位或切图时拒绝。结束或取消都会存图到 `data/<map_uuid>`（空就生成 UUID），把 `system.map_path` 写回 yaml，用最后一帧优化位姿拉起定位 |
| Action `/nav/relocation` | 停掉定位再拉起，用 goal 里的位姿初始化，不换地图。`map_uuid` 必须等于当前地图，否则拒绝（先走 `/map/switch`） |
| 话题 `/map/switch`（`std_msgs/String`，reliable + transient_local） | 字符串是地图目录，要有 `index.txt` 或 `3d/index.txt`。和当前 UUID 相同、或正在建图 / 重定位时忽略。等最多约 5 s 的 INS，再 `SwitchMapAndReinit`：换 `TiledMap`、清空 LIO、设新的初始位姿；成功后把新路径写回 yaml |

建图的数据流（`run_slam_online` 直接跑 `SlamSystem`，`src/core/system/slam.cc`）：

```text
激光 + IMU + INS
  -> FasterLIO 关键帧（位移约 2 m 或转角约 10°）
  -> LoopClosing：每隔 loop_kf_gap 个关键帧用 NDT 找回环
  -> 回环成功后 miao 优化位姿，g2p5 重画栅格
  -> 存图前全局优化：INS 只约束位置（ins_prior_position_only），不锁 z（with_height: false，多层、楼梯能用）
  -> SaveMapByTemplate 按第 3 节的模板写出
```

## 8. 目录和可执行文件

| 目录 | 作用 |
|---|---|
| `lightning-lm-master` | 定位和建图本体，ROS 2 包名 `lightning` |
| `interface/kygi665_msgs` | kygi665 组合导航消息（`INSPVAX` 等） |
| `interface/nav2_interface` | 导航 Action / 消息：`Mapping`、`Relocation`、`Location`、`NavState` |
| `interface/common_interface` | `GeometryPose`、`GeograpicPose` 等 |
| `novatel_oem7_msgs` | 旧组合导航消息（`ins_type: novatel` 时） |
| `livox_ros_driver2-master` | Livox 驱动和 `CustomMsg`；D20 顶激光走 `PointCloud2`，用不上 |
| `basic` | 日志、流形、环形缓冲 |
| `build_l4t.sh`、`toolchain-aarch64.cmake` | 交叉编译到 Jetson |

可执行文件装在 `lib/lightning/`，都从环境变量 `config_path` 读 yaml（不解析 `--config`）：

| 程序 | 干什么 |
|---|---|
| `run_loc_online` | 按 yaml 加载地图并定位；本机回放用这个 |
| `mode_switch_node` | 定位 + `/nav/mapping` + `/nav/relocation` + `/map/switch`；8 号狗上在跑 |
| `run_slam_online` | 在线建图，服务存图 |
| `run_slam_offline`、`run_loc_offline`、`run_frontend_offline`、`run_loop_offline` | 离线吃 bag：建图、定位、只跑前端、只跑回环 |
| `split_map` | 把一整块 PCD 切成分块地图（输入输出路径写死在源文件里） |

Launch（都把 `config_path` 设成包里的 `default_d20.yaml`）：`ros2 launch lightning loc_launch.py`（定位 + RViz）、`mapping_launch.py`（建图）、`model_switch_launch.py`（`mode_switch_node`）。

## 9. 在这台机器上跑

编译环境和规划共用 `/public/zjst/robot/planning_humble_env`（见规划文档第 8 节），二进制是 `colcon/install/lightning/lib/lightning/run_loc_online`。

```bash
# 只起定位（要自己播 bag）
bash Nav_D20/slam/lightning-lm-master/scripts/run_loc_online_debug.sh
# 换地图 / INS 直出：不用改配置，脚本会写一份临时配置（scripts/loc_debug_config.sh）
LOC_MAP_PATH=$PWD/data/loc_maps/ad68517c-e845-42a4-b5dd-a6b7c6d2af13 bash .../run_loc_online_debug.sh
LOC_INS_ONLY=1 bash .../run_loc_online_debug.sh
# 连感知、规划一起回放 0921，录包、转可视化（推荐）
LOC_MAP_PATH=... bash Nav_D20/planning/script/replay_planning_check.sh <名字> 80.0 126.9
```

`ROS_DOMAIN_ID` 默认 42，不要用狗的域。回放脚本把定位要看的话题录下来：`/lightning/odom`、`/rte/rdap/rtk/inspvax`、健康事件进主录包；当前帧和地图点云太大，录在旁边的临时包里，只给转换用。定位日志在 `logs/loc.log`。

## 10. 看定位

回放脚本转出来的 `viz/planning_viz.mcap` 里有定位的几层（`Nav_D20/planning/tools/loc_viz.py`），在 Foxglove 里用布局 `Nav_D20/planning/script/loc_viz.json`：

| 看哪 | 是什么 | 看出什么 |
|---|---|---|
| 3D：`/viz/loc/map_cloud` | 定位地图的点云，按高度着色 | 现场长什么样 |
| 3D：`/viz/loc/scan` | 当前帧，按 LIO 的位姿放进地图；颜色是离地图最近点的距离（绿 = 贴上，红 = 对不上） | 配准好不好：墙、树、路沿对不对得上 |
| 3D：`/viz/loc/ins` | INS 轨迹（紫 = RTK 固定，灰 = 不是）和现在的位置 | 定位和 INS 差多少 |
| 曲线：贴上的比例、距离中位 | 每帧多少点离地图 0.2 m 以内 | 掉下去 = 地图和现场对不上（施工、停的车）或位姿偏了 |
| 曲线：定位减 INS | 两者的水平距离 | |
| 曲线：高度、横滚、俯仰 | | 高度一路掉 = IMU 单位那类问题 |
| 状态条：定位有效、INS RTK 固定 | `/lightning/odom` 的 `covariance[1]`、INS 状态 | |
| 标签页"定位流程" | 第 1 节的六步，每步这一刻的数和代码位置 | 对着代码看 |
| 标签页"事件" | 定位故障 / 恢复、RTK 丢了、偏离 INS、点云不贴地图、健康事件、`loc.log` 里的警告 | |

注意：
- 沿着一个平面滑动看不出来：整帧沿地面平移，地面点仍然"贴着"地面，只有墙、树这些立着的东西变红。
- 当前帧和 INS 只有 T49 起的回放才录（以前没录）。T47 的两份只有地图点云和定位有效标志。

## 11. 已知问题

| 现象 | 原因 | 状态 |
|---|---|---|
| 高频 odom 一路往下掉 | kygi665 加速度单位 g | 已修（T42） |
| 内置地图定位不上、每 10 s 重初始化 | 地图倾斜 51° | 用现场地图（T43 起） |
| INS 直出时规划起点在路外 | INS 位置离 1.7 m 宽的路边只有几分米 | 规划起点挪动（T47） |
| 开头一帧位姿在原点 | 初始化之前的输出 | 可视化里不画，事件里记一条 |
