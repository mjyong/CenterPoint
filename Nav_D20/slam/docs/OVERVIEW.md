# D20 定位（slam）

`Nav_D20/slam` 是狗上的定位仓库。规划侧怎么用这些话题，写在 `planning/docs/ONBOARDING.md`。

## 从哪来的

`lightning-lm-master` 是高翔的开源激光定位建图 [Lightning-LM](https://github.com/gaoxiang12/lightning-lm) 的一份拷贝，不是本仓库的 submodule。`package.xml` 的维护者仍是 gaoxiang，README 末尾还指着这个 GitHub 仓库。拷进赛力斯 D20 工程时留下过本机路径 `.../SaiLiSi/dog_robot/D20/lightning-lm-master_slam_vi/src/lightning-lm-master`（见该 README 建图示例）。

里面三层来源叠在一起：

| 层 | 是什么 |
|----|--------|
| FasterLIO | `src/core/lio`，头文件宏还是 `FASTER_LIO`。Lightning-LM 把它当成前端（README 里的 AA-FasterLIO） |
| Lightning-LM | 高翔在前端外面加的：分块地图、回环、g2p5 栅格、miao 优化、在线/离线入口。文件头 `Created by xiang` |
| D20 改动 | `default_d20.yaml`、KYGI665 组合导航、`SaveMapByTemplate`、`mode_switch_node.cc`（文件头 2026-02-28） |

`livox_ros_driver2-master` 是 Livox 官方驱动，`novatel_oem7_msgs` 是 NovAtel 官方消息。D20 顶激光不用 Livox，INS 用的是旁边自写的 `kygi665_msgs`。

## 定位实际跑什么

仓库里写明的狗上定位进程是 `run_loc_online`（`Nav_D20/README.md`、规划 onboarding）。`default_d20.yaml` 里 `use_super_lio: false`，所以只走 FasterLIO。

```text
run_loc_online
  -> LocSystem                 订激光 / IMU / INS，发 /lightning/odom
  -> Localization              把一帧点云交给 LIO，把有效 INS 送进 ESKF
  -> LaserMapping              去畸变、IMU 预测、点到面匹配
  -> TiledMap + IVox           按当前位姿加载附近地图块，当作匹配的局部地图
```

对应文件：`src/app/run_loc_online.cc`、`src/core/system/loc_system.cc`、`src/core/localization/localization.cpp`、`src/core/lio/laser_mapping.cc`、`src/core/maps/tiled_map.cc`、`src/core/ivox3d/`。配置就是 `config/default_d20.yaml`。

发出去、规划在用的只有 `/lightning/odom`（`map` 下车体位姿，跟 IMU 频率）和 TF `map → base_link`。`/nav/location` 每秒一次，给任务侧看状态。`publish_occ_grid` 和 `publish_map_cloud` 在这份 yaml 里是 false，`/map` 和地图点云这条发布不走。

定位路径上不跑这些：

| 代码 | 什么时候才跑 |
|------|----------------|
| `SlamSystem`、`loop_closing`、`g2p5`、`miao` | 建图（`run_slam_online` 或 `/nav/mapping`） |
| `super_lio_frontend` | `use_super_lio: true` 才换前端，D20 是 false |
| `lidar_loc/`、`pose_graph/` | 上游的 NDT + 位姿图定位。`localization.cpp` 没有调用 |
| `PoseSmoother` 的更新 | `localization.cpp` 里是注释掉的 |
| `mode_switch_node` | 另一个可执行文件，负责建图 / 重定位 / `/map/switch`。onboarding 没有把它写成狗上常驻进程 |
| `run_*_offline`、`split_map` | 离线工具 |

关键就四件事：`default_d20.yaml` 的话题和外参、`laser_mapping` 的匹配、`TiledMap` 读到的那张地图、INS 固定解用来初始化。地图目录没有 `3d/index.txt` 时会退回仓库里倾斜约 51° 的 `map/0c3f8a2d-...`，定位结果会被标成无效。

| 角色 | 仓库目录 | 进程 |
|------|----------|------|
| 感知 | `Nav_D20/perception_combo_ros` | `lidar_seg`、`grid_map_memory_node` |
| 定位 | `Nav_D20/slam` | `run_loc_online` 或 `mode_switch_node` |
| 规划 | `Nav_D20/planning` | `planning` |

D20 实际传感器在 `lightning-lm-master/config/default_d20.yaml`：禾赛 XT32 顶激光、组合导航 IMU、KYGI665 INS。配置里未注释的外参是 D20 A2。

## 目录里有什么

| 目录 | 作用 |
|------|------|
| `lightning-lm-master` | 定位和建图本体，ROS 2 包名 `lightning`。上游是 [Lightning-LM](https://github.com/gaoxiang12/lightning-lm)，D20 在上面接了 INS、地图模板、模式切换 |
| `interface/kygi665_msgs` | KYGI665 组合导航消息（`INSPVAX` 等） |
| `interface/nav2_interface` | 导航 Action / 消息：`Mapping`、`Relocation`、`Location`、`NavState` |
| `interface/common_interface` | `GeometryPose`、`GeograpicPose` 等公共消息 |
| `novatel_oem7_msgs` | 旧组合导航消息。`common.ins_type` 为 `novatel` 时使用 |
| `livox_ros_driver2-master` | Livox 驱动和 `CustomMsg`。D20 顶激光走 `PointCloud2`，在线订阅里 Livox 回调是注释掉的 |
| `basic` | 日志、流形、环形缓冲，给同仓库其它包用 |
| `build_l4t.sh` | 用 `/l4t/jetson_arm64_sysroot` 交叉编译到 Jetson（aarch64） |
| `toolchain-aarch64.cmake` | 上面这条交叉编译的 toolchain |

`lightning-lm-master` 里和 D20 有关的代码：

| 路径 | 作用 |
|------|------|
| `src/app/run_loc_online.cc` | 在线定位入口。配置路径读环境变量 `config_path` |
| `src/app/run_slam_online.cc` | 在线建图入口，地图名固定 `new_map` |
| `src/app/mode_switch_node.cc` | 启动后先按配置定位；Action 切到建图或重定位；听 `/map/switch` |
| `src/app/run_*_offline.cc` | 离线建图、定位、前端、回环，吃 rosbag |
| `src/core/system/loc_system.cc` | 定位的 ROS 外壳：订阅、发布、初始位姿、栅格 |
| `src/core/system/slam.cc` | 建图的 ROS 外壳：关键帧、回环、存图 |
| `src/core/localization/localization.cpp` | 在线定位算法：预处理、LIO、INS 观测、按 IMU 频率发里程计 |
| `src/core/lio` | FasterLIO 前端：去畸变、ESKF、IVox，定位时按位姿从分块地图加载局部点云 |
| `src/core/loop_closing` | 建图回环：NDT 找回环，miao 做位姿图，可加 INS 位置先验 |
| `src/core/g2p5` | 把关键帧点云投成 2D 占据栅格。栅格只给导航显示和存盘，不参与定位 |
| `src/core/maps` | 点云按块存取。定位时只加载车附近的块 |
| `src/core/miao` | 轻量图优化，回环和建图收尾的全局优化用它 |
| `config/default_d20.yaml` | D20 话题、外参、地图路径、回环和栅格参数 |
| `map/0c3f8a2d-...` | 仓库内置回退地图。配置里的地图目录没有 `index.txt` 时用它 |
| `launch/*.py` | 把 `config_path` 设成 `default_d20.yaml` 再拉对应可执行文件和 RViz |

`src/core/localization/lidar_loc` 和 `pose_graph` 是上游那套「NDT 全局匹配 + 位姿图」定位。当前 `localization.cpp` 的在线路径没有调用它们。在线定位是 FasterLIO 对着分块地图做匹配，有效 INS 作为 ESKF 的位置观测。

## 定位时数据怎么走

`run_loc_online` 和 `mode_switch_node` 里的定位都进 `LocSystem`。节点名是 `lightning_slam`。IMU、激光、INS 各一个回调组，三线程执行器并行。

```text
/robot/topic/rdap/top_point_cloud2     PointCloud2，禾赛 XT32
/rte/rdap/rtk/imus/data_raw            sensor_msgs/Imu
/rte/rdap/rtk/inspvax                  kygi665_msgs/INSPVAX
        |
        v
点云：盲区 0.5 m、抽点、按 lidar_type=4 解析成 XT32
        |
        v
FasterLIO（use_super_lio: false）
  IMU 预测 + 点到面匹配
  局部地图来自 TiledMap，按当前位姿加载邻近块到 IVox
  时间最近且有效的 INS 作为 ESKF 观测
        |
        v
每个 IMU 样本发一次 /lightning/odom
  frame_id=map，child=base_link
  位姿先在 IMU 系，再乘 system.extrinsic 的逆，变到车体
同时发 TF map -> base_link
每秒写一行 /tmp/lightning_trajectory.txt
每秒发一次 /nav/location
```

`/lightning/odom` 的频率跟 IMU，不是跟激光。位姿是 LIO 里 IMU 预测的状态（`GetIMUState`），激光匹配成功后才会开始发。`PoseSmoother` 的平滑更新在 `localization.cpp` 里是注释掉的，发出去的是预测位姿本身。

里程计 `pose.covariance` 被挪作状态：

| 下标 | 内容 |
|------|------|
| `[0]` | 发布时的系统时间（秒） |
| `[1]` | `1` 定位未失败，`0` 定位失败（`IsLocFailed`） |

IMU 回调里把角速度乘了 `π/180`，话题上的角速度按度/秒读入，内部和发出的 twist 是弧度/秒。

初始位姿：

1. `system.with_ins: true` 时等 INS。RTK 固定解（pos_type 56）用来设初始位姿，并跳过 NDT。
2. 等满 `30 s` 仍没有可用 INS，退回 `system.init_t` / `system.init_r`。
3. 定位过程中 LIO 触发重初始化，且缓冲里有 RTK 固定解，用最新 INS 位姿盖回去。

`system.ins_lat/lon/height` 是局部 ENU 原点。`default_d20.yaml` 里生效的是超级工厂那一组；五云湖那组是注释。

`system.ins_only_mode: true` 时不跑 LIO，直接把 INS 位姿发到 `/lightning/odom`。连续 3 秒没有固定解会按丢失处理。本机回放全程 RTK 固定解的包时用这个模式，把规划调试和激光定位分开。

## 建图时数据怎么走

`run_slam_online` 直接跑 `SlamSystem`。`mode_switch_node` 收到 `/nav/mapping` 后先停掉定位，再跑同一套建图，避免两个线程同时吃激光和 IMU。

```text
激光 + IMU + INS
    -> FasterLIO 关键帧（位移约 2 m 或转角约 10°）
    -> LoopClosing：每隔 loop_kf_gap 个关键帧做 NDT 回环
    -> 回环成功后 miao 优化位姿，g2p5 重绘栅格
    -> 运动超过约 3 m 或 30° 时，Mapping feedback 带一帧 2D 图
```

D20 配置里回环、3D UI、g2p5 都开着。`loop_closing.with_height` 是 false，不锁 Z，多层和楼梯还能用。`with_ins_prior: true` 且 `ins_prior_position_only: true`：存图前的全局优化只用 INS 约束位置。

存图走 `SaveMapByTemplate`。`run_slam_online` 另外提供服务 `/lightning/save_map`（`lightning/srv/SaveMap`）。`mode_switch_node` 在 Action 结束或取消时自己存，目录是 `data/<map_uuid>`。`map_uuid` 为空就生成一个 UUID。空关键帧不会写文件，Action 返回后回到配置里的默认地图。

存完之后 `mode_switch_node` 把 `system.map_path` 写回 yaml，再用最后一帧优化位姿拉起定位。

## 模式切换

`mode_switch_node` 启动即定位。三种模式互斥：建图、定位、切图中的 IDLE。

| 入口 | 行为 |
|------|------|
| Action `/nav/mapping` | 停定位，建图。已有建图、正在重定位、正在切图时拒绝。取消也会存图 |
| Action `/nav/relocation` | 停掉定位再拉起，用 goal 里的位姿做初始化，不换地图。`map_uuid` 必须等于当前已加载地图，否则拒绝，先走 `/map/switch` |
| 话题 `/map/switch`（`std_msgs/String`，reliable + transient_local） | 字符串是地图目录。目录下要有 `index.txt` 或 `3d/index.txt`。与当前 UUID 相同、或正在建图/重定位时忽略。成功后把新路径写回 yaml |

重定位和切图都会先等最多约 5 秒的 INS，再 `SwitchMapAndReinit`：换 `TiledMap`，LIO 清空后设新先验。

## 地图目录

定位认两种布局，有 `index.txt` 或 `3d/index.txt` 才算有效。

当前建图写出的模板：

```text
<data>/<uuid>/
  global_map.yaml
  2d/<起始时间戳>/map.png
  2d/<起始时间戳>/map.yaml      分辨率、原点、occupied/free 阈值
  3d/global.pcd                 整图，给查看和发布
  3d/index.txt                  分块索引，定位加载用这个
  3d/<块>.pcd
  session/keyframes/<id>.pcd
  session/keyframes/<id>.yaml   pose_lio、pose_opt
  session/manifest.yaml
  session/factor_graph.yaml
```

旧布局是根目录直接放 `global.pcd`、`index.txt`、`map.yaml`、`map.png` 或 `map.pgm`。加载时两种都认。

配置的 `system.map_path` 没有有效索引时，回退到配置文件上一级的 `map/0c3f8a2d-41eb-416d-8fca-aaad500e2be9`，并打日志。这时 `/nav/location` 的几何状态是地图失败，定位结果当无效。这张内置图来自很短的一段直行，地面相对水平倾斜大约 51°，不能拿来做定位。检查：

```bash
python3 Nav_D20/slam/lightning-lm-master/scripts/check_map_tilt.py \
  Nav_D20/slam/lightning-lm-master/map/0c3f8a2d-41eb-416d-8fca-aaad500e2be9
```

狗上配置写的地图是 `/opt/seres/nav/data/ad68517c-e845-42a4-b5dd-a6b7c6d2af13`。本机没有这个目录时就会落到上面的回退图。

g2p5 假设雷达水平。只把地面以上 `min_th_floor` 到 `max_th_floor` 的点投进栅格。D20 是地面高度 `-0.7 m`，高度带 `0.50–1.0 m`，分辨率 `0.2 m`。白是空闲，黑是占据，灰是未知。

## D20 实际订阅和发布

配置：`lightning-lm-master/config/default_d20.yaml`。话题名都在 `common` 段。

| 方向 | 话题 / Action / 服务 | 类型 | 说明 |
|------|----------------------|------|------|
| 订 | `/robot/topic/rdap/top_point_cloud2` | `sensor_msgs/PointCloud2` | 顶激光 |
| 订 | `/rte/rdap/rtk/imus/data_raw` | `sensor_msgs/Imu` | 角速度按度/秒 |
| 订 | `/rte/rdap/rtk/inspvax` | `kygi665_msgs/INSPVAX` | `ins_type: kygi665` 时 |
| 发 | `/lightning/odom` | `nav_msgs/Odometry` | `map` 下车体位姿，IMU 频率 |
| 发 | `/nav/location` | `nav2_interface/Location` | 1 Hz。地图 UUID、map 系位姿、经纬高、速度 |
| 发 | `/lightning/current_scan_world` | `PointCloud2` | 去畸变点云变到 map |
| 发 | `/lightning/map_cloud` | `PointCloud2` | 启动时把 `global.pcd` 降采样到 0.5 m 发 3 次 |
| 发 | `/map` | `nav_msgs/OccupancyGrid` | reliable + transient_local。读存好的 `map.png` |
| 发 | TF `map` → `base_link` | | 与 odom 同一时刻 |
| Action | `/nav/mapping` | `nav2_interface/action/Mapping` | 仅 `mode_switch_node` |
| Action | `/nav/relocation` | `nav2_interface/action/Relocation` | 仅 `mode_switch_node` |
| 订 | `/map/switch` | `std_msgs/String` | 仅 `mode_switch_node` |
| 服务 | `/lightning/save_map` | `lightning/srv/SaveMap` | 仅 `run_slam_online` |

`default_d20.yaml` 里 `publish_map_cloud` 和 `publish_occ_grid` 都是 false。这套配置启动后不发 `/lightning/map_cloud` 和 `/map`。规划文档按「定位会发 `/map`」来接；要发全局栅格，把 `publish_occ_grid` 改成 true。局部障碍仍是感知的 `/map/occupancy_grid`，和这里的全局 `/map` 不是同一个话题。

`/nav/location` 的几何状态：正在初始化、地图加载成功、地图失败（含回退到内置图）。经纬高状态在 RTK 固定解或浮点解时为 2，否则为 -1。`twist.angular.x` 被写成发布时的系统时间，不是横滚角速度。

建图调试话题还有 `/lightning/ins_pose`、`/lightning/ins_path`。RViz 可视化在 `with_ui: true` 时发 `/lightning/current_scan`、`/lightning/keyframes`、`/lightning/lio_path`、`/lightning/keyframe_path`。

## 可执行文件

`colcon` 包名 `lightning`，装到 `lib/lightning/`。都从环境变量 `config_path` 读 yaml，不解析 `--config`（`run_loc_online.cc` 里 gflags 解析是注释掉的）。

| 程序 | 干什么 |
|------|--------|
| `run_loc_online` | 按 yaml 加载地图并定位。狗上和本机回放用这个 |
| `run_slam_online` | 在线建图，服务存图 |
| `mode_switch_node` | 定位 + `/nav/mapping` + `/nav/relocation` + `/map/switch` |
| `run_slam_offline` | 离线遍历 bag 建图，结束写到 `data/new_map` |
| `run_loc_offline` | 离线定位 |
| `run_frontend_offline` | 只跑 LIO 前端 |
| `run_loop_offline` | 只跑回环 |
| `split_map` | 把一整块 PCD 切成 TiledMap。源文件里的输入输出路径是写死的 |

Launch（都会设 `config_path` 为包内 `config/default_d20.yaml`）：

```bash
ros2 launch lightning loc_launch.py            # run_loc_online + RViz
ros2 launch lightning mapping_launch.py        # run_slam_online，use_sim_time
ros2 launch lightning model_switch_launch.py   # mode_switch_node，use_sim_time
```

## 本机怎么跑

编译好的 Humble 在 `/public/zjst/robot/planning_humble_env`，不在仓库里。下面这条只在本机回放，不替代狗上的定位进程。

```bash
# run_loc_online：用 default_d20.yaml 订顶激光和 IMU，发 /lightning/odom。
# LOC_MAP_PATH 换成一张带 3d/index.txt 的地图；不设则沿用 yaml，本机通常会退回内置图。
# LOC_INS_ONLY=1 时不跑 LIO，直接发 INS 位姿。
bash Nav_D20/slam/lightning-lm-master/scripts/run_loc_online_debug.sh
```

和感知、规划一起回放：

```bash
# run_planning_stack_debug.sh：播 0921 bag，拉起地面分割、局部栅格、定位和 planning。
bash Nav_D20/planning/script/run_planning_stack_debug.sh
```

两个脚本都认 `LOC_MAP_PATH` 和 `LOC_INS_ONLY`。设了之后会复制一份临时 yaml，不改仓库里的 `default_d20.yaml`。
