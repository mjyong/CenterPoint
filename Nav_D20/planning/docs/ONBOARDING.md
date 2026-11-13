# D20 规划入门

2026-11-12 重写。对应 `Nav_D20` 里机器人团队狗上在跑的新版（04d3a34 同步）：A\* 在 `GlobalMapManager` 里，D20 的局部规划是 MPPI。我们打的补丁在 `Nav_D20/PATCHES.md`，定位见 `Nav_D20/slam/docs/ONBOARDING.md`，路口让行的设计见仓库根目录 `docs/PLANNING_INTEGRATION.md`。

先看什么：
1. 第 0 节和第 1 节，知道一个周期干了什么；
2. 在 Foxglove 里打开 `work_dirs/planning_io_t47_map/viz/planning_viz.mcap`，用布局 `Nav_D20/planning/script/planning_viz.json`（第 9 节）。右边"规划流程"面板就是第 1 节那张表，每个周期填着当时的数，边播边对着代码看；
3. 想改什么，查第 7 节的表。

## 0. 一页看懂

狗上导航是三个进程，只靠 ROS 2 话题和 Action 通信：

| 角色 | 代码 | 狗上 | 给规划什么 |
|---|---|---|---|
| 感知 | `Nav_D20/perception_combo_ros` | `/opt/seres/nav_perception`，`nav_perception.service`：`lidar_seg`、`grid_map_memory_node` | `/map/occupancy_grid`：狗周围 10 x 10 m 的障碍栅格 |
| 定位 | `Nav_D20/slam/lightning-lm-master` | `mode_switch_node`（建图 / 定位切换） | `/lightning/odom`：狗在 `map` 里的位姿和速度，100 Hz |
| 规划 | `Nav_D20/planning` | `/opt/seres/nav/planning`，`planning.service`：一个可执行文件 `planning` | 发 `/cmd_vel`、`/mc/motion_control` 给运控 |

```text
/goal_pose 或 Action /nav/navigation（终点）
        │
        ▼
┌──────────────── planning，每周期约 50 ms ────────────────┐
│ ① 输入   RosBridge：里程计、感知栅格、终点；断流就暂停      │
│ ② 全局   GlobalMapManager：CJGC 路网图；A* 全局路径（一次） │
│ ③ 局部图 CostMapProduction：狗周围 15 x 10 m 的代价地图    │
│ ④ 任务   Tracking 状态机：off → passive → standby → active  │
│ ⑤ 局部   MPPI：沿全局路径截 5 m，采样 3000 条轨迹挑最好的   │
│ ⑥ 输出   /cmd_vel（vx, vy, ω）→ /mc/motion_control         │
└──────────────────────────────────────────────────────────┘
```

车型由 `$PLANNING_ROOT_DIR/robot_factory/default_robot_vin` 决定，配置都在 `robot_factory/<车型>/planning_conf/`。**2026-11 同步来的新版里这个文件写的是 `d30/001`**（T48 回放就读成了 d30）；我们的狗是 `d20/001`。调试脚本用 `PLANNING_VIN`（默认 `d20/001`）在临时目录里搭一份软链接把它换掉（`script/planning_vin.sh`），不改安装目录；日志第一行 `Load robot config:` 看读的是哪个。

## 1. 一个周期怎么跑

入口 `src/runnable/entry/main.cc` 读三份配置，然后一直循环 `PlanningEntry::Proc()`（`src/runnable/entry/planning_entry.cc`）：

```cpp
communication_entry_.Proc();   // ① 收话题、查超时
preprocess_entry_.Proc();      // ② ③ 全局图、局部代价地图
mission_entry_.Proc();         // ④ ⑤ 当前任务的状态机，active 时跑 MPPI
communication_entry_.Publish();// ⑥ 发速度、调试话题
```

| 步 | 做什么 | 代码 | 在可视化里看哪 |
|---|---|---|---|
| ① 输入 | 订阅 `/lightning/odom`、`/map/occupancy_grid`、`/goal_pose`；任务在 active / standstill 时，里程计或感知栅格断流 1 s 就 `pause()`，恢复 3 s 后 `resume()` | `src/runnable/communication/ros_bridge/ros_bridge.cc`（`Proc`、`checkTimeout`、`OnGoalPose`） | 流程面板 ①；事件"输入超时" |
| ② 全局 | 启动时读 CJGC 路网图；进 standby 时 A\* 从狗到终点搜一条路并平滑；每周期给出从狗往前的一段（`ExtractLocalPath`） | `src/runnable/preprocess/global_map_manager/global_map_manager.cc`；A\* 在 `src/algorithm/grid_path_planner/` | 灰蓝路网、蓝色全局路径 |
| ③ 局部图 | 全部置空闲 → 全局图按狗的位姿叠进来 → 感知栅格的障碍叠进来 → 算到障碍的距离场 | `src/runnable/preprocess/cost_map_production/cost_map_production.cc`（`Proc`） | 半透明红（障碍）、橙（感知） |
| ④ 任务 | 当前任务（D20：跟踪 Tracking、回充 Recharging 等）的状态机走一步 | `src/runnable/mission/mission_entry.cc`、`tracking/tracking.cc`（`Proc` → `mission_fsm_ctx_.Proc()`） | 任务状态条；狗后面的文字 |
| ⑤ 局部 | active 时：截 5 m 参考路径 → 变到 `base_link` → MPPI 优化 → 局部路径和速度 | `Tracking::spin_s_active`；`src/algorithm/mppi_planner/` | 紫色参考路径、绿色局部路径和脚印 |
| ⑥ 输出 | 速度经 `VelocityCalibration` 发 `/cmd_vel`、`/mc/motion_control`（`MODE_WBMC` + `MOTION_GAIT_WALK`，`para = [vx, vy, ω]`）；`RosDebugBridge` 发调试话题 | `RosBridge::Publish`；`src/runnable/communication/ros_debug_bridge/` | 橙色指令箭头和转向弧 |

## 2. 任务和状态机

终点有两条入口，最后都是"把任务设成 Tracking 并 `start()`"：
- `/goal_pose`：当前必须是 Idle 或任务已 off。默认当 `map` 系；终点朝向要求到点（`constraint_theta`）。
- Action `/nav/navigation`：当前不是 Idle 会先 `stop()` 并拒绝这次目标；接受后用 `poses[0]`，`max_speed > 0.01` 会改目标速度。

Tracking 的状态（`src/runnable/mission/mission_state_machine/mission_fsm_base.h`；`check_e_*` 决定转不转，`on_entry_s_*` 进来时做一次，`spin_s_*` 每周期做）：

| 状态 | 进来时 / 每周期做什么 | 什么时候离开 |
|---|---|---|
| off | 清掉全局路径、局部参考、速度 | 收到终点（`need_start`）→ passive |
| passive | 什么都不做 | 下一周期 → standby |
| standby | **进来时做一次全局规划**：A\* 从狗到终点（`GlobalMapManager::PlanGlobalPath`） | A\* 成功（`global_plan_flag`）→ active；失败 → fault；要停 → off |
| active | **每周期**：截局部参考 → MPPI → 速度 | 到终点 0.5 m 内、朝向差 0.2 rad 内、速度 0.15 以下，保持 1 s → off；`need_pause` 或被挡（`nav_blocked`）→ standstill |
| standstill | 速度 0 | `need_resume` → active；要停 → off |
| fault | 全局规划失败 | 要停 → off |

调试字符串 `/planning/debug/plotjuggler/string` 里的 `mission_type`（0 空闲、1 跟踪、2 跟随、3 回充、4 技能）和 `mission_status`（0 off、1 passive、2 standby、3 active、4 standstill、5 fault）就是这两个枚举。

路口让行（有车就停、车过去再走）还没做，会用这里的 `pause()` / `resume()`，设计见 `docs/PLANNING_INTEGRATION.md`。

## 3. 全局地图和全局路径

- **CJGC 是一张路网图**：`robot_factory/data/astar_pgm/CJGC.pgm` + `CJGC.yaml`（0.1 m 一格，10293 x 11412，原点 `[-716, -321]`）。像素 255 是能走的路，是一条条约 1.7 m 宽的线，只占全图 0.26%；其余都是障碍。
- `GlobalMapManager::LoadPGMMap` 读 `<input_name>.pgm`（不管 yaml 里的 `image` 写的什么；`CJGC.yaml` 里写的是 `map.pgm`，那是同目录另一张图）。读进来以后 `obstacle` 层 **255 = 空闲、0 = 障碍**（`negate: 0` 时把像素反一下再按阈值判）。
- A\*（`src/algorithm/grid_path_planner/`）用距离场：离障碍至少 `safe_margin`（D20 0.1 m）再加 `inflation_margin`（0.05 m）的格子才能走。
  - 起点不合格（定位偏几分米就会落到 1.7 m 宽的路外面）：`allow_collision_start` 开着，在 1 m 内挪到最近的空地，日志 `Start adjusted`（我们加的，T47 验证）。终点同理 `allow_collision_goal`。
  - 搜完用平滑器平滑（`use_smoother`）。
- 切地图：`/map/switch`（`std_msgs/String`，地图目录）。`ReloadMap` 读那个目录 `global_map.yaml` 第一层的 `map_image_file`（ad68517c 是 `2d/<id>/map.png`）和 `map_yaml_file`，再读同目录的 `map.osm`，没有就读 `semantic.osm`（人行横道在这里面）。
- **新地图 ad68517c 比 CJGC 多一段路**：人行横道 6 往西那段（x 66–106、y 207–231）。T47 的狗 107 s 以后走在这段上，用 CJGC 时局部代价地图认为狗站在障碍里。
- 人行横道：`GlobalMapManager::crosswalks()`（`third_party/osm_map` 读，我们注册了 `crosswalk`），只存多边形，不改障碍层。

## 4. 局部代价地图

`CostMapProduction`，狗前 10 m、后 5 m、左右各 5 m，0.05 m 一格，跟着狗转（`base_link` 里），`enable_cuda: true`。每周期（`Proc`）：

1. `obstacle` 层全部置 255（空闲）；
2. `AddObstacleFromMap`：全局图的 `pgm_obstacle` 层按狗的位姿采样进来——路网外面全是障碍；
3. `AddObstacleFromPerception`：感知栅格里的障碍格写成 0（障碍）；
4. `UpdateObstacleDistance`：`cv::distanceTransform`（0 当障碍）得到到障碍的距离场；
5. 再算参考路径偏移、合成代价，给 MPPI 的 `obstacles_critic` 和碰撞检查用。

**调试话题 `/planning/debug/cost_map_production/occmap` 在 T47 复核之前是反的**：调试桥按 (0, 1) 转，空闲全成了 100、障碍成了 0，Foxglove 里路显示成障碍。已改成 (255, 0)，T48 重编后生效；老录包转可视化时会自动认出来并翻过来（第 9 节）。以前"狗脚下被占"的结论也是反着读的。

## 5. 局部规划：MPPI

`mission_conf.pb.txt` 里 `tracking_conf.planner_type: MPPI`（TEB 还在，`teb_planner_conf`）。`Tracking::spin_s_active` 每周期：

1. `ExtractLocalPath`：从全局路径上离狗最近处往前截 `local_reference_path_length`（5 m），变到 `base_link`——这是 MPPI 的**参考路径**（可视化里紫色）；
2. `MppiPlanner::solver(参考路径, 当前位姿 = 0, 当前速度)`：在上一周期的控制序列上加噪声（`sampling_std`：vx 0.4、vy 0.2、ω 0.2）采样 `batch_size` 3000 条控制序列（当前速度做了均值滤波后当起点），每条 `time_steps` 50 步 x `model_dt` 0.1 s = 5 s，用全向模型（`motion_model: Omni`，vx、vy、ω）推出轨迹，各代价项打分，按 `temperature` 0.3 加权平均出新的控制序列（`src/algorithm/mppi_planner/optimizer.cc`；`enable_cuda` 时在 GPU 上算）；
3. `getOptimizePath()`：优化后的轨迹，变回 `map` 存成 `local_path_2d`（绿色）；`getVelocityCmd(0.01)`：这一刻的速度。

速度限制（`dynamic_constraints`）：vx −1.0 ~ 3.0 m/s，vy ±0.5 m/s，ω ±1.5 rad/s，加速度 3 m/s²。

代价项（`critics/`，每项在配置里有 `enabled`、`weight`）：

| 代价 | 管什么 |
|---|---|
| `constraint_critic` | 超出速度、加速度限制 |
| `obstacles_critic` | 离障碍太近（排斥）、撞上（`collision_cost` 1e6）；用第 4 节的距离场 |
| `path_follow_critic` | 沿参考路径往前走 |
| `path_align_critic` | 轨迹贴着参考路径 |
| `path_angle_critic` | 朝向对着路径 |
| `goal_critic`、`goal_angle_critic` | 离终点 3 m 以内时，到点、到点时的朝向 |
| `prefer_forward_critic` | 少倒着走 |
| `twirling_critic` | 少原地打转 |
| `velocity_deadband_critic` | 避开太小的速度（运控做不出来） |
| `reference_velocity_critic` | 跟上目标速度 |

## 6. 话题和超时

`robot_factory/d20/001/planning_conf/communication_conf.pb.txt`，没写进配置的管道不会创建。

| 方向 | 话题 / Action | 类型 | 谁产生 | 超时 |
|---|---|---|---|---|
| 订 | `/lightning/odom` | `nav_msgs/Odometry` | 定位 | 0.2 s |
| 订 | `/map/occupancy_grid` | `nav_msgs/OccupancyGrid` | 感知 `grid_map_memory_node` | 0.5 s |
| 订 | `/map` | `nav_msgs/OccupancyGrid` | 定位（reliable + transient_local） | |
| 订 | `/map/switch` | `std_msgs/String` | 切图指令，定位和规划都听 | |
| 订 | `/goal_pose` | `geometry_msgs/PoseStamped` | RViz / 调试 | |
| 订 | `/initialpose` | `PoseWithCovarianceStamped` | RViz | |
| 订 | `/reflector_position`、`/robot/topic/rdap/battery_states`、`/mc/state` | | 回充、整机、运控 | 0.5–1 s |
| 发 | `/cmd_vel` | `geometry_msgs/Twist` | | |
| 发 | `/mc/motion_control` | `mc_interface/Motion` | | |
| Action | `/nav/navigation`、`/nav/back_charge`、`/nav/ptz_camera` | `nav2_interface` | 任务侧 | |

断流：任务在 active / standstill 时，里程计或感知栅格超时持续 1 s（`timeout_pause_duration`）就暂停，恢复连续 3 s（`timeout_resume_duration`）后继续。超时按消息头的时间戳和本机时钟比；回放旧 bag 时两者差几十天，永远算超时，暂停了就再也恢复不了（T48）。所以 `ros_bridge.cc` 加了 `TimeoutReference()`：脚本设 `PLANNING_TIMEOUT_WALL_CLOCK=1` 时改用收到消息的时刻；狗上不设，行为不变。这个补丁在 2026-11 同步时丢过一次（`PATCHES.md`），同步以后先 `grep -n TimeoutReference` 看还在不在。

调试话题要 `pub_flag: true`（D20 默认 false；本机运行中可以 `ros2 service call /planning/debug/set_pub_flag std_srvs/srv/SetBool "{data: true}"`）：

| 话题 | 内容 |
|---|---|
| `/planning/debug/path_tracking` | 全局路径、局部参考（`local_reference_2d`，T49 重编后才有）、局部路径和脚印、跟踪目标点、狗、碰撞圆 |
| `/planning/debug/goal_pose` | 终点 |
| `/planning/debug/cost_map_production/occmap` | 局部代价地图（T47 复核之前是反的） |
| `/planning/debug/global_map_manager/occmap` | 全局图（有订阅者时才发一次） |
| `/planning/debug/plotjuggler/string` | 每周期一条"key: value"：`mission_type`、`mission_status`、`cmd_vx/vy/omega`、`odom_*`、`odom_ts`、`occ_ts`、超时标志 |
| `/planning/debug/diagnostic`、`collision_checker`、`teb_planner`、`following` | 诊断、碰撞点、TEB、跟随 |
| `/tf` | `map -> base_link`、`base_link -> ptz_*`（一棵树，我们改的） |

## 7. 想改什么看哪

| 想改 | 看 |
|---|---|
| 话题名、超时、调试开关 | `planning_conf/communication_conf.pb.txt` |
| 车体尺寸 | `robot_factory/d20/001/robot_param.pb.txt`（1.0 x 0.4 m） |
| 全局地图、A\*（安全距离、起点挪动） | `planning_conf/preprocess_conf.pb.txt` 的 `global_map_manager_conf`；代码 `global_map_manager.cc`、`grid_path_planner.cc` |
| 局部代价地图范围、分辨率 | `preprocess_conf.pb.txt` 的 `cost_map_production_conf` |
| 到点判定、局部参考长度、局部规划用哪个 | `planning_conf/mission_conf.pb.txt` 的 `tracking_conf` |
| MPPI 速度限制、采样、代价权重 | `mission_conf.pb.txt` 的 `mppi_planner_conf`；代码 `src/algorithm/mppi_planner/` |
| 终点怎么变成任务 | `ros_bridge.cc` 的 `OnGoalPose`、Action 回调 |
| 状态机 | `src/runnable/mission/tracking/tracking.cc` 的 `check_e_*`、`on_entry_s_*`、`spin_s_*` |
| 发给运控的速度 | `RosBridge::Publish`、`VelocityCalibration` |
| 调试话题 | `ros_debug_bridge.cc`、`visualization_util.h` |
| 路口让行 | 还没做；`docs/PLANNING_INTEGRATION.md`，Python 参考实现 `dog_perception/yield_gate.py` |

## 8. 在这台机器上跑

命令都在 CenterPoint 仓库根目录执行。

**编译环境**：规划是 ROS 2 Humble 的 C++ 节点（acados、Ceres、g2o、OSQP 0.6、grid_map、OctoMap、PCL、CUDA）。狗上的 aarch64 包走官方镜像和 `bash build.sh -arm`；这台机器连不上那个镜像仓库，本机调试用 `/public/zjst/robot/planning_humble_env`：
- `env/`：micromamba 的 robostack-humble（加 grid_map、octomap、PCL、OpenCV 5、Ceres、g2o、glog、Eigen、Protobuf）；
- 源码装进同一前缀的 acados 0.5.1、OSQP 0.6.3（规划用 0.6 的接口，不能换 1.0）、OsqpEigen 0.10.0；
- `colcon/install/`：接口包、x86 的 `planning`、感知的 `lidar_seg`、`grid_map_memory_node`、定位的 `lightning`；
- gcc 10、`/usr/local/cuda` 的 nvcc；运行前 `micromamba activate`，让 conda 的 `libstdc++` 盖过系统的。

增量重编规划（新版多了 `third_party/nav_interface` 消息包，用 `--packages-up-to planning`）：

```bash
ENV=/public/zjst/robot/planning_humble_env/env
export MAMBA_ROOT_PREFIX=/public/zjst/robot/planning_humble_env/root CONDA_BUILD=0
set +u; eval "$(/public/zjst/robot/planning_humble_env/micromamba shell hook -s bash)"; micromamba activate "$ENV"; set -u
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda LIBRARY_PATH="$ENV/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
cd Nav_D20/planning
set +u; source /public/zjst/robot/planning_humble_env/colcon/install/setup.bash; set -u
colcon build \
  --build-base /public/zjst/robot/planning_humble_env/colcon/build \
  --install-base /public/zjst/robot/planning_humble_env/colcon/install \
  --packages-up-to planning \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_EXE_LINKER_FLAGS="-L${ENV}/lib" -DProtobuf_PROTOC_EXECUTABLE="$ENV/bin/protoc"
```

不要加 `-DROS2_JAZZY`。编不过先看是不是我们的补丁（`Nav_D20/PATCHES.md` 里的文件）。

**回放一遍、录包、发终点、查包、转可视化，一条命令**：

```bash
M=$PWD/data/loc_maps/ad68517c-e845-42a4-b5dd-a6b7c6d2af13      # Nav_D20/map/*.zip 解开
LOC_MAP_PATH=$M bash Nav_D20/planning/script/replay_planning_check.sh <名字> 80.0 126.9
#   LOC_INS_ONLY=1：定位用 INS 直出；MAP_SWITCH=$M：发终点前先切到这张图；RECORD_CLOUD=1：连感知点云一起录
```

它播 `bags0921/rosbag2_2026_09_21-16_41_27`（约 173 s；只有顶激光、IMU、INSPVAX 三路），拉起地面分割、局部栅格、定位、规划（`run_planning_stack_debug.sh --once`），`ROS_DOMAIN_ID` 默认 42。**不要在狗的域上跑**：规划会发 `/cmd_vel` 和 `/mc/motion_control`。输出在 `work_dirs/planning_io_<名字>/`：录包、`logs/`（各进程日志、规划的 glog）、`report.txt`（查包结果、mission_state 次数、`CMD:` 次数、起点挪没挪）、`viz/`（第 9 节）。

单独起某一个：`script/run_planning_debug.sh`（只起规划）、`script/run_planning_stack_debug.sh [--no-planning] [--bag 目录] [--once]`，日志在 `/tmp/nav_stack_debug/`。

狗上是 systemd：`sudo systemctl start planning.service`，`journalctl -u planning.service -f`；服务里是 `source /opt/seres/nav/planning/setup.bash; ros2 run planning planning`。

## 9. 看录包

原始录包直接放进 Foxglove 会一闪一闪：
- 规划的代价地图和感知栅格都挂在 `base_link` 上，狗走路时横滚、俯仰 1–4°，两个倾斜的平面互相穿插，空闲格子默认不透明，哪张在上面每帧都在变；
- 规划的标记时间戳是 0；
- 录包用服务器时钟（比 bag 的时间戳晚 52 天）。

所以转一份给人看的（回放脚本最后会自动转）：

```bash
python3 Nav_D20/planning/tools/planning_mcap_viz.py work_dirs/planning_io_<名字> \
    [--map <切过去的地图目录或 zip>] [--osm <有 semantic.osm 的目录或 zip>] [--loc-map <定位地图目录或 zip>]
# -> viz/planning_viz.mcap（十几 MB，原包几百 MB）和 viz/planning_viz.png（总览图）
```

在 Foxglove 里打开它，加载布局 `Nav_D20/planning/script/planning_viz.json`（看规划）或 `loc_viz.json`（看定位）。

- 一切换到 bag 的时间，摊平到地图平面，每一层一个高度；代价地图方向自动认、反的翻过来；空闲透明。
- 左边 3D 跟着狗俯视。右边：速度（实际 / 指令）、距离（到终点、偏离全局路径、最近感知障碍）、任务状态条，下面一组标签页：
  - **规划流程**：这一周期按第 1 节的六步，每步写着当时的数和代码位置——边播边对着代码看；
  - **事件**：任务状态变化、输入超时、开始走和停下、规划或栅格断了多久、位姿跳变、狗站到路外、定位故障、INS，有 glog 时加上规划和定位自己的警告；
  - **数**：这一周期 `/viz/state` 的全部数（含调试字符串的原始值）；
  - **图例**：每个颜色是什么。

| 图层 | 是什么 |
|---|---|
| 灰蓝 `/viz/route_map` | 规划地图的路网 |
| 黄 `/viz/crosswalks` | 人行横道（编号、横穿长度） |
| 半透明红 `/viz/cost_map` | 局部代价地图里的障碍 |
| 橙 `/viz/perception_grid` | 感知栅格里的障碍 |
| 彩色线 `/viz/trail` | 走过的路按速度着色；白圈是停下的地方 |
| 蓝 / 紫 / 绿 `/viz/plan` | 全局路径 / 局部参考（MPPI 的输入）/ 局部路径和脚印；还有跟踪目标点、碰撞圆、终点 |
| `/viz/robot` | 狗（青色）、实际速度（浅蓝箭头）、指令（橙箭头、转向弧）、状态文字 |
| `/viz/loc/*` | 定位：地图点云、当前帧（按贴不贴地图着色）、INS（见定位文档） |

T47 的两份已经转好：`work_dirs/planning_io_t47_{map,ins}/viz/`。

## 10. 查问题

先查包：

```bash
python3 Nav_D20/planning/tools/check_planning_mcap.py <录包目录或 .mcap>
```

它查：`/tf` 是不是一棵树；时间戳对不对得上 TF；标记寿命；`/lightning/odom` 像不像一只在走的狗（速度、高度、跳变）；代价地图话题是不是反的；狗是不是站在障碍上（脚印里至少 10% 的格子）。两张栅格挂在同一个会动的坐标系上时出一条 NOTE（不算问题，看可视化就行）。

规划日志（`/tmp/planning.*.log.INFO.*`，回放脚本会拷到 `logs/planning.INFO`）的关键字：

| 关键字 | 意思 |
|---|---|
| `Load robot config:` | 读的是不是 `d20/001`（回放报告的"车型"一行；读成 `d30/001` 就是 `PLANNING_VIN` 没生效） |
| `GlobalMapManager: PGM map loaded` | CJGC 读到了 |
| `Set mission to Tracking`、`Navigation started` | 终点进了状态机 |
| `mission_state:` | 状态 |
| `Start adjusted`、`Goal adjusted` | A\* 起点 / 终点挪到了路上 |
| `Start point is in collision`、`No free point found near start` | 起点在路外、1 m 内也没路 |
| `local planner solver failed` | MPPI / TEB 失败，速度清零 |
| `Data timeout exceeded, pausing mission` | 里程计或感知栅格断了；回放时每次都有、又没有 `Data recovered, resuming`，就是按时间戳判超时了（`TimeoutReference` 丢了） |
| `Map switch`、`crosswalks in`、`OSM file not exist` | 切图、读人行横道 |
| `Failed to load PGM` | 路径或 `PLANNING_ROOT_DIR` 错了 |

以前遇到过的：

| 现象 | 原因 | 怎么发现 / 修 |
|---|---|---|
| Foxglove 里坐标系来回跳 | `TfBuffer` 每条变换连逆变换一起发、云台多一条边 | 查包工具；已改成一棵树（T41） |
| `/lightning/odom` 一路往下掉（z −46 m） | 定位高频预测把 kygi665 的加速度（单位 g）当 m/s² | 已按重力归一化（T42） |
| `Start point is in collision` | CJGC 只有 1.7 m 宽的路，定位偏几分米起点就在路外 | 起点挪到路上（T47） |
| 录包一闪一闪、路显示成障碍 | 两张栅格挂 `base_link`；代价地图话题反了 | 可视化工具；调试桥改 (255, 0)（T47 复核） |
| 107 s 起狗站在障碍里 | CJGC 没有人行横道 6 那段路 | 切到新地图（T48） |

## 11. 我们的改动

都列在 `Nav_D20/PATCHES.md`（改在哪、为什么、怎么验、哪个提交）。从机器人团队同步代码前先看那一页，同步完重新打上。
