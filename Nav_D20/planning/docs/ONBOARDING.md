# D20 规划 onboarding

狗上导航是三个独立进程，只通过 ROS 2 话题和 Action 通信：

| 角色 | 仓库目录 | 狗上安装位置 | 进程 |
|------|----------|--------------|------|
| 感知 | `Nav_D20/perception_combo_ros` | `/opt/seres/nav_perception`，服务 `nav_perception.service` | `lidar_seg`、`grid_map_memory_node` 等 |
| 定位 | `Nav_D20/slam/lightning-lm-master` | 定位节点 `run_loc_online` | 发布里程计和全局栅格 |
| 规划 | `Nav_D20/planning` | `/opt/seres/nav/planning`，服务 `planning.service` | 单个可执行文件 `planning` |

D20 车型由 `robot_factory/default_robot_vin` 决定，当前内容是 `d20/001`。改这一行会换成 R1 / r11 的配置，不要在狗上随便改。

**2026-11-12 起，`Nav_D20` 是机器人团队狗上在跑的新版**（04d3a34 同步）。和下文写的老版主要不同：
- A\* 挪到了 `GlobalMapManager`（预处理），配置在 `preprocess_conf.pb.txt` 的 `global_map_manager_conf.grid_path_planner_conf`。D20 的 `safe_margin` 从 0.3 改成了 0.1。
- Tracking 只取全局路径；D20 的局部规划是 MPPI（`mission_conf.pb.txt` 里 `tracking_conf.planner_type: MPPI`，`src/algorithm/mppi_planner/`），TEB 还在。
- 新增跟随、回充、技能等任务。
- 现场地图（`Nav_D20/map/*.zip`）的语义层叫 `semantic.osm`，里面标了人行横道。

我们在上面打的补丁（定位的 IMU 单位、`/tf` 树、调试话题的时间戳、A\* 起点挪到路上、读人行横道、OpenCV 5）列在 `Nav_D20/PATCHES.md`。下次同步先看那一页，同步完把补丁重新打上。

## 数据怎么走

```text
顶激光 / 补盲激光
    -> lidar_seg
    -> /patchworkpp/ground_seg_intensity
    -> grid_map_memory_node          局部障碍
    -> /map/occupancy_grid

定位 run_loc_online
    -> /lightning/odom               机器人在 map 下的位姿
    -> /map                          定位地图的全局 OccupancyGrid（transient_local）

任务侧 / RViz
    -> /goal_pose  或  Action /nav/navigation
    -> planning
    -> /cmd_vel
    -> /mc/motion_control            运控：MODE_WBMC + MOTION_GAIT_WALK，para = [vx, vy, yaw_rate]
```

规划主循环大约 20 Hz，顺序固定：

1. `CommunicationEntry` 收话题、检查输入是否超时
2. `PreprocessEntry` 维护全局静态图和车体周围代价地图
3. `MissionEntry` 跑当前任务（D20 上是 Tracking 或 Recharging）
4. `Publish` 把速度发给运控，并 `spin_some`

入口在 `src/runnable/entry/main.cc`。配置目录是 `$PLANNING_ROOT_DIR/robot_factory/<vin>/planning_conf/`。`PLANNING_ROOT_DIR` 由 `install` 里的环境钩子写成 `.../lib/planning`，这个目录下才有 `robot_factory/`。日志在 `/tmp`。

## D20 实际订阅和发布

配置文件：`robot_factory/d20/001/planning_conf/communication_conf.pb.txt`。没写进配置的管道不会创建。

| 方向 | 话题 / Action | 类型 | 谁产生 | 超时 |
|------|----------------|------|--------|------|
| 订 | `/lightning/odom` | `nav_msgs/Odometry` | 定位 | 0.2 s，只在 Tracking 时检查 |
| 订 | `/map/occupancy_grid` | `nav_msgs/OccupancyGrid` | 感知 `grid_map_memory_node` | 0.5 s，只在 Tracking 时检查 |
| 订 | `/map` | `nav_msgs/OccupancyGrid` | 定位，QoS 为 reliable + transient_local | 无 |
| 订 | `/map/switch` | `std_msgs/String` | 切图指令，定位和规划都会听 | 无 |
| 订 | `/goal_pose` | `geometry_msgs/PoseStamped` | RViz / 调试 | 无 |
| 订 | `/initialpose` | `PoseWithCovarianceStamped` | RViz 初始位姿 | 无 |
| 订 | `/reflector_position` | `PoseStamped` | 回充反光板 | 0.5 s |
| 订 | `/robot/topic/rdap/battery_states` | `rte_interface/BatteryStates` | 整机 | 1.0 s |
| 订 | `/mc/state` | `mc_interface/State` | 运控 | 1.0 s |
| 发 | `/cmd_vel` | `geometry_msgs/Twist` | 规划 | |
| 发 | `/mc/motion_control` | `mc_interface/Motion` | 规划，行走时 `para=[vx,vy,wz]` | |
| Action | `/nav/navigation` | `nav2_interface/action/Navigation` | 任务侧下发目标点 | |
| Action | `/nav/back_charge` | `nav2_interface/action/BackCharge` | 回充 | |
| Action | `/nav/ptz_camera` | 云台，`D20PtzBridge`，`assist: true` | | |

D20 这套配置没有跟随任务（没有 `action_following`，也没有障碍物目标订阅）。`sync_odom_occ` 是 false，里程计和局部栅格各自回调，不对齐时间戳。

输入超时：Tracking 处于 active 或 standstill 时，odom 或局部栅格断流持续 1 s（`timeout_pause_duration`）会 `pause`；恢复连续 3 s（`timeout_resume_duration`）后 `resume`。空闲时不检查。

## 一次去目标点

两条入口，最后都是把任务设成 Tracking 并 `start()`：

- `/goal_pose`：当前必须是 Idle，或当前任务已经 off。目标默认当 map 系；`header.frame_id` 不是 `map` 时会用内部 TF 变过去。`setTargetPose(..., constraint_theta=true)`。
- `/nav/navigation`：当前不是 Idle 会先 `stop()` 并 **拒绝** 这次 goal。接受后用 `poses[0]`，四元数任一非零就约束终点航向；`max_speed > 0.01` 会改 TEB 目标速度。

Tracking 状态机：`off -> passive -> standby -> active -> standstill -> off`。

- 进入 **standby**：用全局图做 A\*（`GridPathPlanner`，启发式欧式，`safe_margin` 0.3 m）。失败进 fault，Action 反馈 `NAV_STA_PLAN_PATH_FAILED`。
- **active**：从全局路径上截大约 5 m，变到 `base_link`，交给 TEB。TEB 用局部代价地图避障，输出未来 0.5 s 的速度。D20 速度上限大约 vx 2 m/s、vy 1 m/s、yaw rate π rad/s。
- 终点误差同时满足 1 s 后进 standstill，再过 `reset_idle_time`（2 s）回到 Idle：位置 0.2 m、航向 0.2 rad、速度 0.15。

全局图和局部图是分开的：

- 全局图：`GlobalMapManager` 读 `robot_factory/data/astar_pgm/CJGC.pgm` + `CJGC.yaml`（分辨率 0.1 m，原点约 `[-716, -321]`）。`enable_overlay: false`，感知障碍 **不会** 画进这张图，A\* 只看见静态图。
  - **CJGC 是一张路网图**（2026-11-12，T46 复核）：像素 255 是能走的路，只占 0.26%，是一条条约 1.7 m 宽的线；其余都是障碍。`LoadPGMMap` 把 255 读成 `obstacle` 层的 255，`cv::distanceTransform` 把 0 当障碍，所以只有路上能走。
  - A\* 要起点离路边至少 `safe_margin`（0.3 m），也就是只有路中间约 1.1 m 宽的一条。定位偏几分米就报 `Start point is in collision!`：0921 的起点按 INS 离路边只有 0.41 m，T43 两次回放都卡在这里。
  - 现在 D20 开了 `allow_collision_start`（新版在 `preprocess_conf.pb.txt` 里）：起点不合格时，在 1 m 内挪到最近的空地（和终点的 `allow_collision_goal` 一样）。局部代价地图也把路外当障碍，狗偏出路面时局部规划会报起点在障碍里。
- 局部图：`CostMapProduction`，车体前 10 m、后 5 m、左右各 5 m，分辨率 0.05 m，`enable_cuda: true`。TEB 避障看这张图，所以没有 `/map/occupancy_grid` 时全局路径还能搜，局部速度会不对。

`/map` 是定位发出来的另一张全局栅格。`enable_global_map_reload: false` 时，规划启动用的仍是磁盘上的 CJGC，不是定位刚发出的 `/map`。切图要走 `/map/switch`。

## 调试时看什么

D20 默认 `pub_flag: false`，RViz 里看不到规划路径。本机调试把 `communication_conf.pb.txt` 里 `RosDebugBridge` 的 `pub_flag` 改成 `true`（狗上的发布包不要带着这个改动）。也可以在运行中调用服务 `/planning/debug/set_pub_flag`。

打开之后这些话题才有内容：

| 话题 | 内容 |
|------|------|
| `/planning/debug/path_tracking` | 全局 / 局部路径 |
| `/planning/debug/goal_pose` | 目标 |
| `/planning/debug/cost_map_production/occmap` | 局部规划用的代价地图（2026-11-12 之前的录包是反的：空闲 100、障碍 0，见下） |
| `/planning/debug/global_map_manager/occmap` | A\* 用的全局图 |
| `/planning/debug/plotjuggler/string` | 一帧调试字符串，含 `mission_type`、`cmd_vx` |
| `/planning/debug/diagnostic` | 诊断 |

录下来先查一遍录包，再转成给人看的版本（下一节）：

```bash
python3 Nav_D20/planning/tools/check_planning_mcap.py <录包目录或 .mcap> --layout Nav_D20/planning/script/planning_io.json
```

它查四件事，2026-11-09 之前前三件都有问题，第四件是定位：
- `/tf` 是不是一棵树：`TfBuffer` 以前把每个变换连同它的逆一起发（`map -> base_link` 和 `base_link -> map`），云台还多发一条 `map -> ptz_*`，于是 `map`、`base_link`、`ptz_visible` 都有两个父坐标系，Foxglove 在它们之间来回跳；现在 `/tf` 只发 `map -> base_link`、`base_link -> ptz_*`。
- 时间戳对不对得上：`base_link` 里的局部代价地图以前用本机时钟打戳，回放旧 bag 时比里程计（也就是 `map -> base_link`）晚几周，Foxglove 找不到那个时刻的变换；现在用里程计的时间戳。
- 标记的寿命：以前 0.05 s，正好一个 20 Hz 周期，回放时一闪一闪、暂停时多半看不到；现在 0.5 s，个数会变的数组开头先发 DELETEALL。
- 代价地图的方向：代价地图的 obstacle 层是 255 = 空闲、0 = 障碍，调试话题原来按 (0, 1) 转，空闲全成了 100，路显示成障碍、障碍显示成空的。
  用感知的障碍格来认（规划把它们抄进代价地图当障碍）：T47 两次录包里 11.7 万个全是 0，报 `inverted`。`ros_debug_bridge` 已改成 (255, 0)。
  以前查出来的"狗脚下被占"也是反着读的：那些帧狗其实在路上。现在翻正以后再查，报的是狗站在规划地图的路外面。
- `/lightning/odom` 像不像一只在走的狗（速度、高度、跳变）。T41 回放 0921 时，定位的 IMU 高频预测把 kygi665 的加速度（单位 g）当成 m/s²，两次 LIO 更新之间每秒多掉 8.8 m/s²：17.5 s 里 z 掉到 −46 m、速度到 32 m/s、一跳 43 m。规划拿这个位置当起点，报 `Start point is in collision!`。`lightning-lm-master/src/core/lio/laser_mapping.cc` 已改成按重力归一化（同 `imu_processing.hpp`），T42 验证。

日志关键字：

- `Load robot config:` 确认读的是 `d20/001`
- `GlobalMapManager: PGM map loaded` 确认 CJGC 读到了
- `Set mission to Tracking` / `Navigation started` 确认目标进了状态机
- `mission_state:` 看 off / standby / active
- `CMD:` TEB 速度；没有这行就还没进 active
- `Data timeout exceeded, pausing mission` 里程计或局部栅格断了
- `teb_planner_ solver failed` 局部规划失败，速度被清零
- `Failed to load PGM` 路径或 `PLANNING_ROOT_DIR` 错了

本机回放时，`ROS_DOMAIN_ID` 要和感知、定位一致。Docker 启动脚本里写的是 `22`，以狗上实际环境为准。QoS 不一致会表现为节点活着但收不到图：`/map` 和 `/map/switch` 是 reliable + transient_local；局部栅格跟随感知 yaml 里的 `qos`（D20 点云侧是 best_effort）。

## 在 Foxglove 里看录包

原始录包直接放进 Foxglove 会一闪一闪，原因有三个：
- 局部代价地图（20 Hz）和感知的 `/map/occupancy_grid`（10 Hz）都挂在 `base_link` 上，Foxglove 按各自时间戳的 TF 摆。狗走路时横滚、俯仰 1–4°，两张图的时间戳差 0.1 s，两个倾斜的平面互相穿插；默认空闲格子是不透明的白色，哪张在上面每帧都在变。路径画在 `map` 的 z = 0，`base_link` 在 z ≈ −0.2 m，倾斜的栅格远端会冒上来盖住一段路径，再放开。
- 规划的标记时间戳是 0。在 `map` 里不影响；挂 `base_link` 的（碰撞点）被摆在第一帧 TF 的位置。
- 录包用服务器的时钟记时间（11 月），数据是 bag 的时间戳（9 月 21 日），时间轴、曲线、日志都和 bag、狗上的日志对不上。

所以录完转一份给人看的版本（`replay_planning_check.sh` 最后会自动转），在 Foxglove 里用 `script/planning_viz.json` 打开：

```bash
python3 Nav_D20/planning/tools/planning_mcap_viz.py work_dirs/planning_io_<名字> \
    [--map <切过去的地图目录或 zip>] [--osm <有 semantic.osm 的地图目录或 zip>]
# -> work_dirs/planning_io_<名字>/viz/planning_viz.mcap（几 MB，原包几百 MB）和 planning_viz.png（总览图）
```

- 全部换到 bag 的时间，摊平到地图平面（`/tf` 只剩 `map -> base_link` 的 x、y、朝向），每一层放在自己的高度上，不再互相抢。
- 代价地图方向自动认（`--cost-map-values auto`），反着的翻过来。
- 地图默认是规划的 CJGC（读 `CJGC.pgm`，和规划一样）；给地图目录或 zip 时读 `global_map.yaml` 第一层的 `map_image_file`（切图以后规划读的那张）。
- 定位刚起来时的那一跳（T47：开头一帧在原点，离真实位置 142 m）不画进轨迹，事件里会记一条。

| 话题 | 是什么 | 颜色 |
|---|---|---|
| `/viz/route_map` | 规划地图的路网（能走的路） | 灰蓝 |
| `/viz/crosswalks` | 人行横道，带编号和横穿长度 | 黄 |
| `/viz/cost_map` | 局部代价地图里的障碍（空闲透明） | 半透明红 |
| `/viz/perception_grid` | 感知栅格里的障碍 | 橙 |
| `/viz/trail` | 走过的路，按速度着色；停下 2 s 以上的地方画圈标时间 | 蓝→绿→黄→红 |
| `/viz/plan` | 全局路径（蓝）、局部路径（绿）和它的脚印、跟踪目标点、碰撞圆、终点（红） | |
| `/viz/robot` | 狗（青色，白三角是朝向）、实际速度箭头（浅蓝）、指令箭头和转向弧（橙）、状态文字 | |
| `/viz/state` | 每个规划周期一条 JSON：任务和状态、速度和指令、到终点、偏离全局路径、脚下障碍格数、最近障碍，还有规划调试字符串里的全部数 | 右边曲线 |
| `/viz/events` | 任务状态变化、输入超时、开始走和停下、规划或栅格断了多久、位姿跳变、狗站到路外、诊断，有规划 glog 时加上它的警告 | 右下日志 |

布局右边四张曲线（速度、转速、距离、脚下障碍格）、任务状态条、事件日志，时间轴和 3D 同步，3D 跟着狗（俯视）。

## 代码从哪改

| 想改的行为 | 先看 |
|------------|------|
| 话题名、超时、要不要调试可视化 | `robot_factory/d20/001/planning_conf/communication_conf.pb.txt` |
| 车体尺寸 | `robot_factory/d20/001/robot_param.pb.txt`（长 1.0 m，宽 0.4 m） |
| 全局搜索安全距离、终点判定、TEB 权重和速度上限 | `robot_factory/d20/001/planning_conf/mission_conf.pb.txt` |
| 局部代价地图范围和分辨率 | `robot_factory/d20/001/planning_conf/preprocess_conf.pb.txt` |
| 目标点怎么变成任务 | `src/runnable/communication/ros_bridge/ros_bridge.cc` 的 `OnGoalPose`，以及 `ros_bridge/tracking.cc` |
| A\* | `src/algorithm/grid_path_planner/`，新版由 `GlobalMapManager::PlanGlobalPath` 调 |
| 局部轨迹和速度 | D20 新版是 `src/algorithm/mppi_planner/`（TEB 在 `src/algorithm/teb_planner/`），调用在 `src/runnable/mission/tracking/tracking.cc` 的 `spin_s_active` |
| 人行横道 | `GlobalMapManager::crosswalks()`（`third_party/osm_map` 读 `semantic.osm`） |
| 发给运控的速度 | `RosBridge::Publish`，发出前会过 `VelocityCalibration` |
| 路口有车就停、车过去再走（接 3D 检测和预测） | 还没做，设计见仓库根目录 `docs/PLANNING_INTEGRATION.md`；停和走复用任务的 `pause()` / `resume()` |

## 在这台机器上跑起来

下面的命令都在 CenterPoint 仓库根目录执行（`Nav_D20/` 在仓库里；以前在 `d20car_predict/` 下）。
2026-11-09 之前仓库的 `.gitignore` 把 `Nav_D20` 里所有 `*.sh`、`*.yaml` 都漏掉了（启动脚本、A* 地图的 `CJGC.yaml`、定位和感知的配置），那之前的克隆缺这些文件，跑不起来。

规划是 ROS 2 Humble 的 C++ 节点，链接 acados、Ceres、g2o、OSQP、OsqpEigen、grid_map、OctoMap、PCL、CUDA。狗上的 aarch64 包仍走官方镜像 `10.42.11.106:5000/jetpack6.1-crosscompile-aarch64-x86:v2.2.1` 和 `bash build.sh -arm`。这台机器访问该仓库会超时，主机也没有 `/opt/ros`，所以本机调试不用镜像，编译环境单独放在 `/public/zjst/robot/planning_humble_env`。

这个前缀里有：

- micromamba 环境 `env/`：robostack-humble 的 ROS 2 Humble，加上 grid_map、octomap、PCL、OpenCV、Ceres、g2o、glog、Eigen、Protobuf。`CONDA_OVERRIDE_CUDA=""`，避免再拉一套 CUDA。
- 源码装进同一前缀的 acados 0.5.1、OSQP 0.6.3、OsqpEigen 0.10.0。规划源码用的是 OSQP 0.6 的 `c_int` / `OSQPData`，不能换回 conda 的 OSQP 1.0。
- `colcon/install/`：`seres_interface` 的接口包、`woosh_robot_msgs`、x86 的 `planning`，以及规划要接的 `lidar_seg`、`grid_map_memory_node`、`lightning`（`run_loc_online`）。定位用的 Livox 消息、NovAtel / KYGI665 消息和 `basic` 也在这里。本机没有 Livox SDK，所以没有编 Livox 驱动进程，只编了 `livox_ros_driver2` 的点云消息。
- 编译器是系统 gcc 10 和 `/usr/local/cuda` 的 nvcc。运行时要先 `micromamba activate`，让 conda 的 `libstdc++` 盖过系统的。

二进制：

```text
/public/zjst/robot/planning_humble_env/colcon/install/planning/lib/planning/planning
```

`install/setup.bash` 会把 `PLANNING_ROOT_DIR` 指到同级的 `lib/planning`，里面有 `robot_factory`。

改过 planning 源码后，在这个前缀里增量重编：

```bash
# 激活 /public 上的 Humble 前缀，增量重编 planning。
ENV=/public/zjst/robot/planning_humble_env/env
export MAMBA_ROOT_PREFIX=/public/zjst/robot/planning_humble_env/root CONDA_BUILD=0
set +u
eval "$(/public/zjst/robot/planning_humble_env/micromamba shell hook -s bash)"
micromamba activate "$ENV"
set -u
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
export LIBRARY_PATH="$ENV/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
cd Nav_D20/planning
set +u
source /public/zjst/robot/planning_humble_env/colcon/install/setup.bash
set -u
colcon build \
  --build-base /public/zjst/robot/planning_humble_env/colcon/build \
  --install-base /public/zjst/robot/planning_humble_env/colcon/install \
  --packages-select planning \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_EXE_LINKER_FLAGS="-L${ENV}/lib" \
  -DProtobuf_PROTOC_EXECUTABLE="$ENV/bin/protoc"
```

不要加 `-DROS2_JAZZY`。Ceres 2.2 走 `Manifold`，humble 的 rosbag 仍用 `time_stamp`，这两处已经按版本分开。

启动（脚本会激活前缀并检查二进制，缺了会直接退出）：

```bash
# run_planning_debug.sh：只启动 planning。前面的感知和定位要自己先起来。
bash Nav_D20/planning/script/run_planning_debug.sh
```

`lidar_3d_detection` 用的 2026-09-21 bag 里只有三路，和这条链路用的是同一份：

| bag 话题 | 谁用 |
|----------|------|
| `/robot/topic/rdap/top_point_cloud2` | `lidar_seg`、定位 |
| `/rte/rdap/rtk/imus/data_raw` | 定位 |
| `/rte/rdap/rtk/inspvax` | 定位 |

规划不订 `lidar_3d_detection` 的检测框。D20 局部障碍来自 `lidar_seg` 的 `/patchworkpp/ground_seg_intensity`，再由 `grid_map_memory_node` 发成 `/map/occupancy_grid`。定位发 `/lightning/odom`。默认 bag 是 `bags0921/rosbag2_2026_09_21-16_41_27`，大约 173 秒。

定位配置里的 `/opt/seres` 地图这台机器没有，会回退到仓库里的 `0c3f8a2d-...`。**这张内置地图不能用来定位**（2026-11-09，T42 复核）：它是用 0921 这包的 15 s 建的，对齐到 INS 后整张图倾斜约 51°（`slam/lightning-lm-master/scripts/check_map_tilt.py`），定位每帧只匹配上几十个点，每 10 s 重初始化一次、位置来回跳。本机回放用 `LOC_MAP_PATH=<现场地图目录>` 换成真的现场地图，或者 `LOC_INS_ONLY=1` 用 INS 直出（全程 RTK 固定解的包），见 `slam/lightning-lm-master/scripts/loc_debug_config.sh`。那不是这包录制时的那张图，激光定位可能对不上，里程计仍会按 LIO 往外发。规划的全局 A* 用的是磁盘上的 CJGC，不跟这张定位图走。

bag 的时间戳是 9 月，规划超时比较的是本机时钟。一键脚本会设 `PLANNING_TIMEOUT_WALL_CLOCK=1`，按收到消息的时刻判断断流，狗上的正式进程不要设这个变量。

```bash
# run_planning_stack_debug.sh：播这份 bag，并启动 lidar_seg、grid_map、定位和 planning。
bash Nav_D20/planning/script/run_planning_stack_debug.sh
```

只要前置节点、不启规划：`bash Nav_D20/planning/script/run_planning_stack_debug.sh --no-planning`。换 bag：`--bag <目录>`。只播一遍：`--once`（bag 放完就收，2026-11-12 起；以前会一直等）。日志在 `/tmp/nav_stack_debug/`。

回放一遍、录包、发目标、查包，一条命令（2026-11-12）：

```bash
LOC_MAP_PATH=<现场地图目录> bash Nav_D20/planning/script/replay_planning_check.sh <名字> <目标 x> <目标 y>
```

输出在 `work_dirs/planning_io_<名字>/`：录包、`logs/`（各进程日志和规划的 glog）、`report.txt`（查包结果，mission_state 各状态出现的次数，`CMD:` 的次数，起点、终点挪了没有，起点在不在障碍里）。目标点会等录包和规划两个订阅者都连上再发。

感知和定位用同一套前缀，各开一个终端。脚本在 `ROS_DOMAIN_ID` 没设置时用 `42`，三个进程要设成同一个值，并且不要跟正在跑的狗撞车。

```bash
# run_lidar_seg_debug.sh：启动地面分割，发布 /patchworkpp/ground_seg_intensity。
bash Nav_D20/perception_combo_ros/script/run_lidar_seg_debug.sh

# run_grid_map_debug.sh：用上面的点云做局部栅格，发布 /map/occupancy_grid。
bash Nav_D20/perception_combo_ros/script/run_grid_map_debug.sh

# run_loc_online_debug.sh：按 default_d20.yaml 启动定位，发布 /lightning/odom 和 /map。
bash Nav_D20/slam/lightning-lm-master/scripts/run_loc_online_debug.sh
```

定位配置里的地图路径是 `/opt/seres/nav/data/...`。这台机器没有这个目录时，节点会回退到仓库里的 `lightning-lm-master/map/0c3f8a2d-41eb-416d-8fca-aaad500e2be9`。没有顶激光和 IMU 时，定位进程能起来，但不会持续发布里程计。`lidar_seg` 同样在等点云。

`ROS_DOMAIN_ID` 要和感知、定位一致。没有 `/lightning/odom` 和 `/map/occupancy_grid` 时，Tracking 大约 1 秒后会被超时逻辑暂停。不要在连着狗的 domain 上随便起 planning，它会发 `/cmd_vel` 和 `/mc/motion_control`。

狗上正式进程是 systemd，不是这个脚本：

```bash
sudo systemctl start planning.service
journalctl -u planning.service -f
```

服务里的启动命令是：

```bash
source /opt/seres/nav/planning/setup.bash
ros2 run planning planning
```

给一个目标（规划已经起来、odom 和局部栅格在跳）：

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}"
```

回放时 bag 里至少要有 `/lightning/odom` 和 `/map/occupancy_grid`。只有全局图、没有这两路，Tracking 会在大约 1 秒后被超时逻辑暂停。
