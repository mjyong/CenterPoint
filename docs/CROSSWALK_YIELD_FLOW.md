# 过人行横道给车让行：从感知到规划的整个流程

狗按全局路径走到地图上标的人行横道前，减速、在入口前停下、转到面朝对面，然后问门控"现在能不能过"：
只要看到在动、而且可能经过这条人行横道的车就等，车走了、过一会儿再过；上了人行横道就不再为车停。
这份把这件事从激光点云一直串到发给运控的速度：每一步在哪个文件、输入什么、怎么算、输出什么。
为什么这样定（"宁愿多等"、不做 TTC 急停等）在 [PLANNING_INTEGRATION.md](PLANNING_INTEGRATION.md)；
现场怎么跑、怎么看在 [DOG_FIELD_GUIDE.md](DOG_FIELD_GUIDE.md)。

---

## 0. 总图

```text
 激光点云 + INS + IMU
        │
 ① 感知 lidar_3d_detection（Nav_D20/perception_combo_ros/node_src/lidar_3d_detection，cw-nav-perception）
        │  检测（PointPillars，TensorRT）→ 跟踪（IMM）→ 运动状态 → 预测（3 s）
        ▼
 /perception/lidar_3d/objects   perception_interface/LidarObjectArray，10 Hz，lidar_det 系
        │
 ② 规划收消息 RosBridge::lidar_objects_callback（ros_bridge.cc）→ yield::Object 列表 + 时间戳 + 序号
        │
 ③ 每个规划周期（20 Hz，任务 active 时）Tracking::StepCrosswalk（tracking.cc）
        │  里程计历史、新的目标、地图里的人行横道多边形、全局路径
        ▼
 ④ CrosswalkYield（crosswalk_yield.cc）
        │  目标 lidar_det → base_link → map（用目标时间戳时的位姿）
        │  找路径上下一个人行横道 → 状态机 approach / align / wait / go / cross / none
        │        ▲
        │        └─ ⑤ 门控 yield::Assess + YieldGate::Update（src/algorithm/yield_gate/）：go 还是 wait、为什么
        ▼
 CrosswalkYieldOutput：hold（换成 0, 0, ω）或 speed_scale（缩放）
        │
 ⑥ Tracking::ApplyCrosswalk：改 MPPI 的速度指令 → RosBridge::Publish → /mc/motion_control、/cmd_vel → 运控
        │
 ⑦ 调试：日志一行、DebugDatas crosswalk_*、/planning/debug/crosswalk（MarkerArray）
```

---

## 1. 感知：`lidar_3d_detection` 输出什么

输入（`config/lidar_3d_detection.yaml` 的 `node`）：

| 话题 | 用途 |
|---|---|
| `/robot/topic/rdap/top_point_cloud2` | 激光点云 |
| `/rte/rdap/rtk/inspvax` | INS 位姿（`localization: inspvax`），把 5 帧点云去畸变、叠起来，跟踪在世界系 `camera_init` 里做 |
| `/rte/rdap/rtk/imus/data_raw` | 角速度，去畸变 |

处理（细节在 [DOG_PIPELINE.md](DOG_PIPELINE.md)、[TRACKING_PREDICTION.md](TRACKING_PREDICTION.md)）：

1. 叠 5 帧（0.55 s 内）、去自身、去近点；转到检测系 `lidar_det`：重力对齐、航向跟着狗、原点在激光雷达正下方的地面上，**x 向右、y 向前**；
2. 检测：PointPillars（TensorRT fp16），三类 vehicle / pedestrian / cyclist，分数门限 0.4 / 0.3 / 0.4，再过一遍检测过滤（点数、离地、跨类 NMS）；
3. 跟踪：每类一个 IMM（车：匀速 + 匀转弯两个模型），车 `min_hits` 3、`max_age` 1.5 s；在世界系里跟，所以速度是**对地速度**；
4. 运动状态 `motion`：static（没见动过）/ moving / stopped（动过、现在停着），连续 3 帧动才算 moving，停 1 s 才算停；
5. 预测：每条航迹几种模式（概率从高到低），0.5 s 一个点，到 3 s，每个点带 2x2 协方差；
6. 每帧把航迹和预测转回这一帧的 `lidar_det` 系发出去。

输出 `/perception/lidar_3d/objects`（`perception_interface/msg/LidarObjectArray`，消息定义在
`Nav_D20/perception_combo_ros/common_interface/seres_interface/ros/perception_interface/msg/`，规划那份是同样的三个文件）：

| 字段 | 内容 | 规划用不用 |
|---|---|---|
| `header.stamp` | 这帧扫描结束的时刻（定位时钟） | 用：在里程计历史里找这一刻的位姿 |
| `header.frame_id` | `lidar_det` | 不读，规划按外参认定 |
| `world_frame`、`world_pose` | `camera_init`、这一帧检测系在世界系的位姿 | 不用（规划用自己的定位） |
| `objects[].id` | 航迹 id，存在期间不变 | 用：门控按 id 记"等过谁"、计时 |
| `label`、`label_name` | 0 vehicle、1 pedestrian、2 cyclist | 用：只让 vehicle、cyclist |
| `motion` | 0 static、1 moving、2 stopped | 用：是不是在动 |
| `score`、`coasting` | 置信度、本帧没检测到在外推 | 不用 |
| `age` | 航迹存在多久 [s] | 用：新航迹（< 1 s）规则；静止的车站了多久 |
| `position` | 框中心 [m]（z 是离地高度） | 用 x、y |
| `yaw`、`size` | 朝向、长宽高 | 用：车身框 |
| `velocity` | 对地速度，在 `lidar_det` 轴上 | 用 |
| `covariance[16]` | [x, y, vx, vy] 协方差按行展开 | 用 [x, y] 那块：`[0] [1] [4] [5]` |
| `prediction_times`、`predictions[]` | 预测时刻（相对 stamp）；每种模式的概率、位置、每点 4 个协方差 | 用：门控的车会在哪 |

还有调试用的 `/perception/lidar_3d/tracks`、`/perception/lidar_3d/predictions`（Marker，`camera_init` 系）和
`/perception/lidar_3d/cloud`（`publish_cloud`）。

---

## 2. 规划收消息：`RosBridge::lidar_objects_callback`

`Nav_D20/planning/src/runnable/communication/ros_bridge/ros_bridge.cc`。每来一条消息，转成门控自己的
`algorithm::yield::Object`（`yield_gate.h`，不依赖 ROS），连同时间戳存起来，序号加一（加锁）：

| `yield::Object` | 来自 |
|---|---|
| `id` | `id` |
| `label` | `label_name`；空的话按 `label` 数字 |
| `motion` | `MOTION_MOVING` → kMoving，`MOTION_STOPPED` → kStopped，其他 → kStatic |
| `position`、`velocity`、`yaw` | x、y |
| `length`、`width` | `size.x`、`size.y` |
| `age` | `age`（`has_age = true`） |
| `covariance` | `covariance[0, 1, 4, 5]` |
| `pred_times`、`modes` | `prediction_times`；每种模式的概率、位置、协方差。点数和 `prediction_times` 对不上的模式丢掉；一种都没有就清空 `pred_times`，门控按匀速外推 |

规划主循环里 `RosBridge::lidar_objects()` 取最新的一份（时间戳、序号、目标列表）。

---

## 3. 每个周期：`Tracking::StepCrosswalk`

`Nav_D20/planning/src/runnable/mission/tracking/tracking.cc`。跟踪任务 active 时每个周期（20 Hz）在 MPPI 之前调一次：

1. `AddOdom(stamp, xy, yaw)`：把 `/lightning/odom` 的位姿（map 系）记进最近 5 s 的历史；时间倒退 1 s 以上（换了包）就清空；
2. 有新的感知消息（序号变了）→ `SetObjects(stamp, objects)`；
3. `SetCrosswalks(gmm->crosswalks())`：地图里的人行横道多边形。来源是站点地图的 `semantic.osm` 里类型为 Crosswalk、至少 3 个点的语义；
   规划一启动按 `preprocess_conf` 的 `startup_map_dir` 读一次（日志 `startup map … 5 crosswalks`），收到 `/map/switch` 再按新图读
   （`GlobalMapManager: N crosswalks in …`）；
4. `Step(now = 里程计时间戳, 狗的位置, 朝向, 速度, 全局路径)` → `CrosswalkYieldOutput`；
5. 状态、理由或 "没放行就在人行横道上" 变了就打一行日志，每周期写 `crosswalk_*` 调试数（第 7 节）。

新的全局路径（任务从 standby 出来，`on_exit_s_standby`）时 `Reset()`：忘掉过过的人行横道和门控的计时。

---

## 4. 坐标：目标放到地图上（`CrosswalkYield::ObjectsInMap`）

`crosswalk_yield.cc`。门控在 map 系里算，所以每个周期把最新一条感知消息整体搬到 map：

```text
lidar_det → base_link：转 det_to_base_yaw_deg = −90°（det 的 x 向右 = base_link 的 −y），平移 (0.319, 0.043) m（D20 A2 雷达位置，T46）
base_link → map：     狗在消息时间戳那一刻的位姿（里程计历史里插值）
p_map = R(yaw_odom) · ( R(−90°) · p_det + (0.319, 0.043) ) + xy_odom
```

位置、速度、朝向、协方差、每个预测点和它的协方差都这样转。时间戳在历史外超过 `max_pose_gap`（0.3 s）就找不到位姿，
这一帧算没有感知（门控判 stale，等）。外参对不对用 `Nav_D20/planning/tools/objects_in_map.py` 在录包上验：
停着的车在地图上应该不动，换成另外三个 90° 的朝向都更晃。

---

## 5. 状态机：`CrosswalkYield::Step`

每个周期按顺序：

1. 没开（`enable: false`）或地图里没有人行横道 → none；
2. `PathAhead`：从狗的位置接上全局路径上离它最近的那一段往后，最多 `lookahead`（40 m）；
3. 狗中心在某个人行横道里？
   - 是这个人行横道、而且已经放行（go / cross）→ **cross**：不停；门控照算（`on_crosswalk`，永远 go），给调试看；
   - 没放行就在里面（等在路边时定位跳了一下、或者狗是从路上起步的）：沿着过街方向往回量离入口线多深；
     不超过 `max_unreleased_depth`（1.5 m）就当成还在路边，照常 align / wait / go；更深就 cross（已经在路中间，停下更糟）；
4. 找路径上下一个还没过的人行横道：`CrosswalkSpan` 给出路径进入（`s_in`）和离开（`s_out`）它的位置。没有 → none；
5. 换了一个新的人行横道 → approach。停止点 `s_stop = s_in − (0.5 × 狗长 1.0 + stop_margin 0.3)`：狗中心停在入口线前 0.8 m；
6. 门控查的路径段：从狗到出口再多 `check_beyond_exit`（1 m）。

| 状态 | 什么时候 | 输出 | 门控 |
|---|---|---|---|
| none | 前面 40 m 没有人行横道 | 不改指令 | 不调 |
| approach | 离停止点 > `hold_distance`（0.1 m） | `speed_scale = clamp(s_stop / slow_distance 3 m, 0.2, 1)`：最后 3 m 线性减速到两成 | 不调 |
| align | 到了停止点，朝向和过街方向差 > `align_tolerance_deg`（10°） | hold，ω = clamp(1.0 × 朝向差, ±0.5 rad/s)：原地转 | 一到停止点就 `Reset` 重新开始等；转的时候计时照走 |
| wait | 朝向对了、门控说 wait | hold，ω = 0 | 每周期问 |
| go | 门控说 go | 不改指令（MPPI 照常走） | 每周期问，狗在动时传 `dog_moving`；新来车还会变回 wait，除非已经在车的扫掠范围里（committed） |
| cross | 狗在已放行的人行横道里 | 不改指令；进去时记成"过过了" | `on_crosswalk` |

```text
            新的人行横道在前面
 none ───────────────────────────▶ approach（减速）
  ▲                                   │ 到停止点
  │                                   ▼
  │                               align（原地转）──┐
  │                                   │ 朝向对了    │
  │                                   ▼            │
  │                ┌────────────── wait ◀──┐       │ 朝向对了且门控 go
  │                │ 门控 go               │ 门控 wait
  │                ▼                       │
  │               go ──────────────────────┘
  │                │ 狗中心进了人行横道
  │                ▼
  └──────────── cross（不停）
     出了人行横道、前面没有别的
```

---

## 6. 门控：`Assess` + `YieldGate::Update`

`Nav_D20/planning/src/algorithm/yield_gate/yield_gate.{h,cc}`，是 `dog_perception/yield_gate.py` 的 C++ 版，两边逐帧对过
（`Nav_D20/planning/tools/yield_gate_parity/`）。现在用的规则是 **wait**（2026-11-12 定）。

### 6.1 `Assess`（无状态：现在路上有什么）

- 感知太旧：没有时间戳，或 `now − stamp > stale_after`（0.3 s）→ stale；
- 路径段按 0.1 m 取点；狗走廊半宽 = 0.5 × 0.4 + 0.5 = 0.7 m；
- 只看 vehicle、cyclist；
- **在动的车**（`motion` 是 moving，或速度 ≥ 0.5 m/s）：
  - 往后看多久：`clamp(60 m / 车速, 8 s, 30 s)`；
  - 车会在哪：感知的每种预测模式（概率 ≥ 0.1 的，最可能的那种一定要），0.1 s 一步插值；3 s 以后按最后一段的速度直线外推，
    位置标准差每秒长 0.3 m、最多 1 m；再加左转、右转两种（半径 10 m，最多转 90°）；
  - 车身框每边放大：0.5 m（车）+ 1 × 标准差 + 0.7 m（狗走廊）+ 1.5 m（wait 规则多给的）；
  - 放大的框在这段时间里**碰到路径段上任何一点** → 冲突（kind 是 predicted 或 turn）。wait 规则不算"来不来得及"，碰到就算；
  - 狗现在站的地方在车（不放大的）扫掠范围里 → `in_lane`（下面 committed 用）；
- **站着的车**：在走廊旁边（横向 ≤ 半车宽 + 0.5 + 0.7 m，纵向 ≤ 半车长 + 10 m，车头或车尾对着路径）→ standing；车身压在走廊上 → blocking；
- **新航迹**：`age` < 1 s、离路径 30 m 内、还没算冲突 → 冲突（kind young）：速度和预测还不可信。

### 6.2 `YieldGate::Update`（有状态：计时、滞回）

按顺序，第一条成立的就是结果：

| 条件 | 结果（理由） |
|---|---|
| 狗在人行横道上 | go（`on_crosswalk`），清空等过的车 |
| 感知 stale | wait（`stale`） |
| 狗在动、没在等、狗已经在某辆在动的车的扫掠范围里 | go（`committed`）：不在车前面停下 |
| 有冲突 | wait（全是新航迹 `young`，否则 `vehicle`）；这些车记进"等过的" |
| 有站着的车：近路径的站够 3 s 以前、压着路径的站够 15 s 以前（static 的车站了多久从感知的 `age` 起算，stopped 的从门控第一次看到它站着起算） | wait（`standing` / `blocked`） |
| 等过的车里还看得到、离它最后一次冲突不到 3 s（`wait_clear`） | wait（`clearing`） |
| 等过的车看不到了、离最后一次看到不到 5 s（`lost_hold`） | wait（`lost`）：被挡住或开出范围的车，也得等一会儿 |
| 都不是 | go（`clear`），清空等过的车 |

结果 `Decision`：`go`、理由、相关的航迹 id（`waiting for …`）、最小间隔。

---

## 7. 输出

### 7.1 改速度：`Tracking::ApplyCrosswalk`

MPPI 算出 (vx, vy, ω) 以后：

- `hold`（align、wait）：换成 (0, 0, `hold_omega`)，并且 `ResetNavBlocked()`、本周期不判"被挡"（是故意停的，不能变成 standstill、重新规划）；
- 否则：vx、vy 乘 `speed_scale`（approach 减速；go、cross 是 1），ω 不变；
- MPPI 求解失败时也调一次（指令本来是 0），所以路边照样能转向。

### 7.2 发出去：`RosBridge::Publish`

改过的速度经 `VelocityCalibration`（积分项现在关着）发 `/cmd_vel`；任务不在 standstill 时还发 `/mc/motion_control`
（`MODE_WBMC`、`MOTION_GAIT_WALK`、`para = [vx, vy, ω]`，运控执行的是这个）。等车时任务还是 active，所以发的是 (0, 0, 0)。

**最终输出就是发给运控的这一个速度**：路边减速、停下、原地转向、等、放行后正常走。人行横道让行没有别的输出通道，
不改路径、不改任务状态。

### 7.3 调试输出

- 日志（`cw-planning`），状态、理由或 unreleased 变了才打：

  ```text
  Crosswalk yield: <状态> (<理由>) crosswalk <序号>, stop point in <m> m, entry <m> m, exit <m> m, heading error <deg> deg,
                   vehicles <n>[, perception stale][, waiting for <id> ...][, on the crosswalk without a go (<m> m past the entry)]
  ```

- `DebugDatas`（`/planning/debug/plotjuggler/string`，Foxglove 侧栏"规划的数"）：`crosswalk_mode`（0–5 依次 none、approach、align、
  wait、go、cross）、`crosswalk_index`、`crosswalk_s_stop`、`crosswalk_hold`、`crosswalk_go`、`crosswalk_vehicles`（最近一帧里
  门控管的车有几辆）、`crosswalk_waiting_for`、`crosswalk_stale`、`crosswalk_unreleased`、`crosswalk_heading_error`；
- `/planning/debug/crosswalk`（`visualization_msgs/MarkerArray`，map 系，`pub_flag` 开着才发，规划自己的 bag 也录）：
  80 m 内的人行横道（白框；正在处理的加粗、按状态变色：黄 approach、红 align / wait、绿 go / cross）；
  门控看到的车（转到 map 以后的位置，蓝框，门控在等的红框，箭头是速度）；狗头上一行"状态（理由）, waiting for N"。

---

## 8. 参数

**能在配置里改的**：`mission_conf.pb.txt` 的 `tracking_conf.crosswalk_yield_conf`（`Nav_D20/planning/src/proto/mission_conf.proto`）。
仓库的 d20/001 只写了 `enable: true`、`yield_rule: "wait"`，其余用默认值：

| # | 参数 | 默认 | 意思 |
|---|---|---|---|
| 1 | `enable` | false（d20/001 设 true） | 开关 |
| 2 | `stop_margin` | 0.3 m | 狗中心停在入口线前半个狗长 + 这么多 |
| 3 | `slow_distance` | 3.0 m | 停止点前这么长开始减速 |
| 4 | `min_speed_scale` | 0.2 | 减到 MPPI 指令的这么多 |
| 5 | `hold_distance` | 0.1 m | 离停止点这么近（或过了）就停下转向 |
| 6 | `align_tolerance_deg` | 10° | 朝向差在这以内才问门控 |
| 7 | `align_gain` | 1.0 /s | 原地转的增益 |
| 8 | `max_align_omega` | 0.5 rad/s | 原地转最快 |
| 9 | `check_beyond_exit` | 1.0 m | 门控查到出口再多这么长 |
| 10 | `lookahead` | 40 m | 路径上往前找人行横道 |
| 11 | `moving_speed` | 0.1 m/s | 狗超过这个速度算在动 |
| 12 | `max_pose_gap` | 0.3 s | 感知时间戳离里程计历史超过这么多：没有位姿 |
| 13–15 | `det_to_base_x / y / yaw_deg` | 0.319、0.043、−90° | 感知系在 base_link 里（D20 A2，T46） |
| 16 | `yield_rule` | "wait" | wait / gap / stop |
| 17 | `stale_after` | 0.3 s | 感知比这旧就等 |
| 18 | `max_unreleased_depth` | 1.5 m | 没放行就在人行横道上：这么深以内当路边 |

**只在代码里的**（`yield_gate.h` 的 `YieldConfig`，改要重编；Python 参考 `dog_perception/yield_gate.py` 要一起改，parity 才过）：
`classes`（vehicle、cyclist）、`dog_length` 1.0、`dog_width` 0.4、`lateral_margin` 0.5、`vehicle_margin` 0.5、`sigma_k` 1、
`min_mode_prob` 0.1、`horizon` 8 s、`sigma_growth` 0.3、`sigma_max` 1.0、`stopped_speed` 0.5、`stopped_wait` 3 s、`blocked_wait` 15 s、
`start_zone` 10 m、`young_age` 1 s、`young_range` 30 m、`turn_radius` 10 m、`turn_max_deg` 90°、`wait_reach` 60 m、
`wait_max_horizon` 30 s、`wait_margin` 1.5 m、`wait_clear` 3 s、`lost_hold` 5 s、`min_watch` 0。

---

## 9. 怎么验

| 层 | 工具 | 看什么 |
|---|---|---|
| 门控的规则 | `dog_perception/yield_gate.py`（参考实现）、`tools/dog/yield_eval.py`（1104 的标注当真值，T44） | 该等的没放、放的时候安全；不安全的按原因分类 |
| C++ 和 Python 一样 | `Nav_D20/planning/tools/yield_gate_parity/run.sh`（`parity.cc` 逐帧对；`crosswalk_sim.cc` 闭环走一遍 `CrosswalkYield`） | 每帧 go / 理由一样；闭环状态序列对 |
| 外参 | `Nav_D20/planning/tools/objects_in_map.py <包>` | extrinsic OK |
| 回放 | `Nav_D20/planning/script/replay_planning_check.sh`（服务器 nav-dev 镜像，T58；狗上域 42，T56） | 日志里的状态序列；`check_planning_mcap.py`、`planning_mcap_viz.py` |
| 真狗 | T57（DOG_FIELD_GUIDE.md 第 8 节） | 空人行横道 → 同事开车 → 正常车流 |
| 单元测试 | `tests/dog/test_yield_gate.py`、`test_yield_gate_cpp.py`、`test_objects_in_map.py` | |

---

## 10. 已知的限制

- 感知只看 40 m 左右的点云，算不了"来得及过去"，所以用 wait 规则：车多的路口会等很久；
- 不给行人让（只 vehicle、cyclist）；近处的人和障碍由局部栅格和 MPPI 管；
- 人行横道的位置全靠地图和定位：地图选错（App 的 `/map/switch`）、定位偏了，停的地方就错；
- 外参、INS 朝向错了，停着的车会"动起来"，狗会一直等（1112 的教训：天线接反时的目标）；
- 感知和里程计的时间对不上（> 0.3 s）就一直 stale、一直等；
- 上了人行横道就不再停：这时候来车只能靠车让狗（人定：车撞狗车全责；停在车道里更危险，PLANNING_INTEGRATION.md 第 4 节）。
