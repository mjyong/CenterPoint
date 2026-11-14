# 8 号狗上板指南：跑起来、调、看、安全地试

这份是现场用的：我们的感知、定位、规划（`/opt/seres/nav_cw`）怎么在 8 号狗上起来、怎么换回机器人团队的、
现场怎么和狗交互、各模块的参数在哪调、Foxglove 里看什么、调试开关、规划和运控怎么安全地试。
装（编译、拷地图、写服务）是 T56 做的，步骤在 `docs/SERVER_TASKS.md` 的 T56；第一次现场过人行横道是 T57。
人行横道让行从感知到规划的整个流程在 [CROSSWALK_YIELD_FLOW.md](CROSSWALK_YIELD_FLOW.md)。

规矩（一直有效）：

- sudo 一律 `sudo -n`；密码不写进仓库、文档、`~/.ssh/config`；
- 不用 `l4tbr0` 上的 192.168.55.1（down 的）；
- 狗的 ROS 域 0 上只跑狗自己的服务和我们换上去的那三个；回放、演练一律 `ROS_DOMAIN_ID=42`；
- 不 `apt install`，不改机器人团队的服务和它们的文件；我们的东西只在 `/opt/seres/nav_cw/`、`/etc/systemd/system/cw-*.service`；
- 从狗上拷东西回来只读；
- 狗在走的时候，旁边一定有人拿急停。

---

## 0. 一页纸

```text
# 服务器上（经隧道）
ssh dog 'sudo -n bash /opt/seres/nav_cw/nav_cw.sh status'          # running: theirs
ssh dog 'sudo -n bash /opt/seres/nav_cw/nav_cw.sh switches'        # 开关：影子、App 地图、限速、感知调试
ssh dog 'sudo -n bash /opt/seres/nav_cw/nav_cw.sh shadow on'       # 第一次：先影子模式（狗不按规划走）
#   App 地图：appmap ignore（默认，2026-11-14）：我们不听 App 的切图，一直用站点地图 ad68517c（App 的数据库只有 c1adbe7b）
ssh dog 'sudo -n bash /opt/seres/nav_cw/nav_cw.sh speed 0.5'       # 限速
ssh dog 'sudo -n bash /opt/seres/nav_cw/nav_cw.sh use-new'         # 换成我们的（15 s 内没起稳、或 ROS 图和开关对不上，自己换回）
ssh dog 'sudo -n bash /opt/seres/nav_cw/nav_cw.sh check'           # 谁在发狗的指令、谁在听 /map/switch，和开关对得上

ssh dog 'sudo -n bash /opt/seres/nav_cw/nav_cw_foxglove.sh'        # 另一个终端：Foxglove 的桥，前台留着
#   笔记本 Foxglove：ws://<狗>:8765，Layout → Import Nav_D20/planning/script/planning_live.json
#   3D 面板右上 "Publish pose" 点目标（只有 /goal_pose 能从 Foxglove 发）
ssh dog 'journalctl -u cw-planning -f | grep -E "Crosswalk yield|Map switch|startup map|crosswalks in"'
ssh dog 'sudo -n bash /opt/seres/nav_cw/nav_cw.sh use-original'    # 换回（重启狗也是换回）
```

---

## 1. 现场有什么

| 谁 / 什么 | 在哪 | 干什么 |
|---|---|---|
| 服务器 | `10.36.136.7` | `ssh dog` 经反向隧道到狗（`DEPLOY_ORIN.md` 第 1 节：端口 5038，用户 `nvidia`）；拷回录包，跑分析 |
| 狗边上的本机 | 连着狗 | 维持隧道；跑 Foxglove 桌面版（连狗的 8765，不在一个网段就 `ssh -N -L 8765:127.0.0.1:8765 nvidia@<狗>`） |
| 8 号狗 | Orin，L4T R36.4.0，`tegra-ubuntu` | 跑导航；CycloneDDS 只绑 eth1，域 0，`/opt/ros/cyclonedds.xml` |
| App | 手机 | 看狗的状态、遥控。App 认的是它的数据库（`/userdata/config/robot_service/db/robot_service.db`），库里只有 c1adbe7b，它发的 `/map/switch`（latched）一直是 c1adbe7b；数据库我们不改。所以 `appmap ignore`（默认，2026-11-14）：我们的规划和定位不听 App 的 `/map/switch`、`/map/file_update`，一直用站点地图 ad68517c；App 目录里什么都不放。**App 上的狗画在 c1adbe7b 上，位置不准**，看 Foxglove。换回机器人团队的服务后一切照旧 |
| 遥控器 | 人手里 | 人接管。和规划的指令谁优先由机器人团队的运控决定，第一次上板先试（第 8 节第 1 步） |
| 急停 | 专人拿着 | 狗走的时候一直拿着，不干别的 |
| Foxglove | 本机 | 看（第 6 节）、发目标；不能发别的 |

狗上看话题（读，不发）：

```bash
ssh dog
source /opt/ros/humble/setup.bash && source /opt/seres/nav_cw/install/planning/setup.bash   # 我们的消息定义
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///opt/ros/cyclonedds.xml ROS_DOMAIN_ID=0
ros2 topic hz /perception/lidar_3d/objects       # 10 Hz 左右
ros2 topic echo --once /lightning/odom
```

收不到大话题（点云）时用 root 跑（机器人的共享内存是 root 的，T40）。域 0 上**不要** `ros2 topic pub`。

---

## 2. 两套导航

| | 机器人团队的（默认，开机起） | 我们的（`/opt/seres/nav_cw`，不开机起） |
|---|---|---|
| 感知 | `nav_perception.service`（`lidar_seg`、`grid_map_memory_node`、云台检测、回充）+ `lidar-3d-detection.service`（我们以前的 deb） | `cw-nav-perception.service`：仓库的 `lidar_seg`、`grid_map_memory_node`、`lidar_3d_detection`（2026-11-14 并进感知），云台检测和回充照旧用 `/opt/seres/nav_perception` 的程序 |
| 定位 | `slam.service` | `cw-slam.service`：仓库的 lightning（`mode_switch_node`），地图是仓库的 |
| 规划 | `planning.service` | `cw-planning.service`：仓库的规划，人行横道让行打开，开机就读站点地图和 `semantic.osm` 里的人行横道（`startup_map_dir`） |

```text
/opt/seres/nav_cw/
  install/{nav_perception,slam,planning}/   各自的前缀（各用各的接口包，build_stack.sh）
  deps/                                     编的依赖（g2o、acados、OsqpEigen），不进系统
  maps/<id>/                                仓库的站点地图（Nav_D20/map/*.zip 解开）
  start/                                    cw-nav-perception 的启动脚本
  nav_cw.sh  nav_cw_foxglove.sh             换 / 换回 / 开关；Foxglove 的桥
  nav_cw.env                                开关写在这里（三个 cw unit 用 EnvironmentFile= 读）
```

数据怎么走：

```text
激光 /robot/topic/rdap/top_point_cloud2 ─┬─ lidar_seg → grid_map_memory_node → /map/occupancy_grid（狗周围 10 x 10 m 障碍栅格）
INS /rte/rdap/rtk/inspvax、IMU           │                                    /fused_point_cloud
                                        ├─ lidar_3d_detection → /perception/lidar_3d/objects（车、人、骑车的：框、速度、预测）
                                        └─ lightning（定位）→ /lightning/odom、TF map → base_link、/map（全局栅格）
/goal_pose（Foxglove）→ 规划 → /mc/motion_control（MODE_WBMC、GAIT_WALK，para = [vx, vy, ω]：狗走的是这个）
                              /cmd_vel（同一个速度）
```

换的时候 `nav_cw.sh` 先停原来的四个（含 `lidar-3d-detection.service`，不然两个节点发同一个话题），再起我们的三个，
15 s 内三个都要 active 且没重启过，否则把原来的起回来、退出 1、打出我们三个服务最后 15 行日志。
两套 unit 互相 `Conflicts=`，systemd 不会同时跑。什么都没 enable：**狗重启就回到机器人团队的**。

---

## 3. 跑起来

### 3.1 每次上板

1. 狗站着、在 ad68517c 那片；急停在人手里；`switches` 里 appmap 是 ignore。
2. `nav_cw.sh status` → `running: theirs`；`nav_cw.sh switches` 看开关（第 5 节），第一次上板要影子模式、限速。
3. `nav_cw.sh use-new`。成功后看日志（`journalctl -u <服务> -n 50`）：

   | 服务 | 要看到 | 不对 |
   |---|---|---|
   | `cw-slam` | 收敛，没有 fault，不一直重定位；读的是 `/opt/seres/nav_cw/maps/...` | 换回 |
   | `cw-planning` | `Tracking: crosswalk yield on, rule wait`；`startup map … 5 crosswalks`；之后没有 `Map switch`、`Map update`（appmap ignore） | 有切换：appmap 没生效（`check` 看图），换回 |
   | `cw-nav-perception` | `lidar_3d_detection` 每秒一行、10 Hz 左右；`lidar_seg`、`grid_map_memory_node` 起来了 | 换回 |

4. Foxglove（第 6 节）：狗在地图上的位置对、障碍栅格跟着狗、人行横道框在路上。
5. 发目标（第 4 节），按第 8 节的顺序试。
6. 完了 `nav_cw.sh use-original` → `running: theirs`，App 里狗正常。

### 3.2 回放和演练（域 42，不碰狗的域 0）

狗上演练用的就是现场那三个 `cw-*` 服务、机器人原来的 CycloneDDS 配置（开着共享内存），只是在域 42 上、喂的是包（T59 第 5 步是完整的做法）：

```bash
sudo -n bash /opt/seres/nav_cw/nav_cw.sh shadow on           # 演练必须开着：规划不在 /mc/motion_control 上发
sudo -n bash /opt/seres/nav_cw/nav_cw.sh rehearsal on        # nav_cw.env 写 ROS_DOMAIN_ID=42、PLANNING_TIMEOUT_WALL_CLOCK=1；
                                                             # 停机器人团队的四个导航服务，起我们的三个，在域 42 上查图
E='source /opt/ros/humble/setup.bash; source /opt/seres/nav_cw/install/planning/setup.bash; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///opt/ros/cyclonedds.xml ROS_DOMAIN_ID=42'
sudo -n bash -c "$E; ros2 bag play <包> --clock --remap /perception/lidar_3d/objects:=/bag/perception/lidar_3d/objects /map/switch:=/bag/map/switch /cmd_vel:=/bag/cmd_vel /mc/motion_control:=/bag/mc/motion_control"
sudo -n bash -c "$E; ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped '{header: {frame_id: map}, pose: {position: {x: X, y: Y}, orientation: {w: 1.0}}}'"
sudo -n bash /opt/seres/nav_cw/nav_cw.sh rehearsal off       # 我们的停、他们的起回来，nav_cw.env 里域 42 那两项去掉
```

- 包里录的目标、地图切换、指令都改名避开：目标用我们现场重算的；切换和指令是录的时候的，不能当成现在的；
- `PLANNING_TIMEOUT_WALL_CLOCK=1`：回放旧包时规划的超时按收到消息的时刻算（ONBOARDING.md），只在演练时设；
- 用 root 跑 ros2 命令：机器人的共享内存是 root 的 `iox-roudi`，`nvidia` 的进程拿到的数据不全（T40）；
- 第一次用共享内存演练前先验它不串域（T59 第 5 步 b：域 42 发、域 0 收不到）；
- `rehearsal on` 开着时 `use-new` 不肯执行，`switches` 显示 REHEARSAL ON；我们的服务在域 0 上跑着时 `rehearsal on` 也不肯执行。

Foxglove：`rehearsal on` 时 `sudo -n bash /opt/seres/nav_cw/nav_cw_foxglove.sh` 自己从 `nav_cw.env` 读到域 42；别的时候要看域 42 就
`sudo -n env ROS_DOMAIN_ID=42 bash …`（变量写在 sudo 后面：sudo 会清掉前面设的环境变量）。服务器上回放：`Nav_D20/planning/script/replay_planning_check.sh`（nav-dev 镜像，T58）。

---

## 4. 和狗交互

- **发目标**：Foxglove 3D 面板右上 "Publish pose"，在地图上点一下、拖出朝向，发的是 `/goal_pose`（`map` 系；别的系规划用 TF 转到 map）。
  人行横道的测试点在对面路边。拖出的朝向就是到了以后狗面朝的方向：到终点 0.5 m 内、朝向差 0.2 rad 内才算到，
  朝向拖反了狗会在终点原地转过来。App 的 `/nav/navigation` 我们不用（人定 2026-11-14）。
- **停**：
  - 硬的：急停；
  - 遥控接管（第一次先确认接得过来）；
  - 软的：在狗脚下发一个目标（到了就停）；`sudo -n systemctl stop cw-planning` 不再发指令——运控收不到指令时是停下还是保持最后的速度，
    第一次上板狗站着、限速时先试清楚（第 8 节第 1 步），没试清楚前不当成停车手段；
  - 收工、出错：`nav_cw.sh use-original`。
- **看状态**：`journalctl -u cw-planning -f`。`mission_state: active / standstill / off` 每个周期打一行（要 grep）；
  人行横道的状态每变一次打一行：

  ```text
  Crosswalk yield: wait (vehicle) crosswalk 6, stop point in -0.02 m, entry 0.78 m, exit 7.9 m, heading error 3 deg, vehicles 2, waiting for 17
  ```

- **App**：电量、遥控照旧，位置不准（画在 c1adbe7b 上）。`appmap follow` 时才跟 App：在 App 里切图，规划和定位都会跟着切
  （`/map/switch`）；`appmap ignore`（默认）时 App 怎么切都不影响我们。

---

## 5. 调试开关

都在狗上用 `sudo -n bash /opt/seres/nav_cw/nav_cw.sh …`；开关存在 `/opt/seres/nav_cw/nav_cw.env`（三个 cw unit 读），程序从配置里读的再写进装好的那份配置（`communication_conf`、`mission_conf`），只重启相关的那个服务（没在跑就下次起来生效）；
`switches` 看当前设了什么。不影响机器人团队的服务。

| 开关 | 做什么 | 代价 |
|---|---|---|
| `shadow on / off` | 影子模式：规划照常算，`/cmd_vel` 和 `/mc/motion_control` 改发到 `/nav_cw/shadow/cmd_vel`、`/nav_cw/shadow/mc/motion_control`，**狗不按规划走**；人用遥控遛狗，Foxglove 里比。改的是装好的 `communication_conf` 里 `pub_cmd_vel`、`pub_motion` 的名字（规划不读命令行参数：`rclcpp::init(0, nullptr)`，`--ros-args -r` 没用）；重启后到 ROS 图里查，规划还在发 `/mc/motion_control` 就把 `cw-planning` 停掉、退出 3 | 重启 `cw-planning`（任务没了，重发目标） |
| `appmap ignore / follow` | ignore（默认）：规划（`communication_conf` 的 `sub_map_switch`、`sub_map_update`）和定位（launch 参数 `map_switch_topic`）改听 `/nav_cw/map/switch`、`/nav_cw/map/file_update`，App 的切图不再让它们换图，一直在 `configure` 设的站点地图上。follow：跟 App（只在 App 的数据库里有站点地图时用；8 号狗的没有） | 重启 `cw-slam`（重新收敛）、`cw-planning` |
| `apply` | 把 `nav_cw.env` 里的开关（`NAV_CW_SHADOW`、`NAV_CW_APPMAP`、`NAV_CW_SPEED`）再写进装好的配置：重装会把仓库的配置放回来，开关本身不会丢；`use-new` 换之前自己也写一遍；`switches` 里出现 NOT APPLIED 就是没写进去 | 重启我们在跑的 `cw-slam`、`cw-planning` |
| `rehearsal on / off` | 演练：on 在 `nav_cw.env` 写 `ROS_DOMAIN_ID=42`、`PLANNING_TIMEOUT_WALL_CLOCK=1`，停机器人团队的四个导航服务，起我们的三个、在域 42 上查图（要影子模式开着；我们的在域 0 上跑着时不肯）；off 换回去、去掉那两项（第 3.2 节） | 演练期间狗没有导航 |
| `check` | 查 ROS 图：谁在发 `/mc/motion_control`、`/cmd_vel`、影子话题，谁在听 `/map/switch`，和上面两个开关对不对；`use-new` 换完自己也查，对不上就换回 | 无 |
| `speed <m/s> / off / show` | 限速：改装好的 `mission_conf` 里 MPPI 的 `max_vx`（仓库 3.0）和巡航 `reference_velocity`（仓库 1.5）；第一次限速时原文件存成 `mission_conf.pb.txt.repo`，`off` 放回 | 重启 `cw-planning` |
| `debug planning on / off` | 规划的调试话题（`/planning/debug/*`，`pub_flag`），马上生效（服务 `/planning/debug/set_pub_flag`）。`configure` 已经默认打开（`NAV_CW_DEBUG=0` 装的时候不开） | 规划多发几十 KB/s |
| `debug perception on / off` | `lidar_3d_detection` 发调试点云 `/perception/lidar_3d/cloud`、TF `camera_init → lidar_det`（`LIDAR3D_ARGS`） | 重启 `cw-nav-perception`：栅格断几秒，规划暂停 1 s 后 3 s 恢复 |

Foxglove 的桥（`nav_cw_foxglove.sh`，以 root 跑，带白名单：不限的话 iceoryx 的发布者表会满，桥退出，T40）：

| 变量 | 做什么 |
|---|---|
| 默认 | 布局里显示的：规划的调试话题、两张栅格、`/fused_point_cloud`、`/lightning/odom`、`/cmd_vel`、`/mc/motion_control`、影子的两个、目标、TF |
| `NAV_CW_FOXGLOVE_LEVEL=full` | 再加重的：定位的 `current_scan_world`、`map_cloud`、`ins_path`、`lio_path`、`ins_pose`、`/map`；感知的 `tracks`、`predictions`、`cloud`；规划的 `teb_planner`、`following`、`gradient_vector`、`cost_map_production/grid_map`、`plotjuggler/data_points`。布局里这些层默认藏着，3D 面板里勾上。几分钟就关 |
| `NAV_CW_FOXGLOVE_EXTRA="^/a$ ^/b$"` | 临时再放几个话题 |
| `NAV_CW_FOXGLOVE_PORT` | 端口，默认 8765 |

能从 Foxglove 发的只有 `/goal_pose`（`client_topic_whitelist`），所以 Foxglove 永远发不了速度。

---

## 6. 可视化看什么（`Nav_D20/planning/script/planning_live.json`）

布局：左边 3D（跟着 `base_link`），右边速度、角速度两张曲线，侧栏标签页里是规划的调试数、诊断、感知的原始目标。

| 层 / 面板 | 看什么 | 正常 | 不对说明 |
|---|---|---|---|
| `/planning/debug/global_map_manager/occmap` | 全局栅格（站点地图） | 和路网对得上 | 地图不对（App 选错图） |
| `/map/occupancy_grid` | 感知的局部障碍栅格 | 10 x 10 m 跟着狗，车、墙在对的地方 | 感知或外参问题；断流时规划会暂停 |
| `/planning/debug/cost_map_production/occmap` | 规划的代价地图 | 障碍外有膨胀 | |
| `/planning/debug/path_tracking` | 全局路径、局部参考、MPPI 的优化路径 | 沿路，经过人行横道 | 路径穿墙：地图或定位 |
| `/planning/debug/goal_pose` | 目标 | 在你点的地方 | |
| `/planning/debug/collision_checker` | 碰撞检查的足迹 | | 一直红：被挡（nav_blocked） |
| `/planning/debug/crosswalk` | 人行横道（80 m 内）：白框；正在处理的那个按状态变色：黄 approach、红 align / wait、绿 go / cross；车框：蓝 = 看到的车、红 = 门控在等的车，箭头是速度；狗头上一行状态字 | 车框贴在真车上，停着的车在地图上不动 | 车框跟着狗转：外参不对（`objects_in_map.py`）；整体偏：定位或时间戳 |
| `/lightning/odom` | 狗的位置和朝向 | 平稳，不跳 | 跳：定位，停下来查 |
| 速度曲线 | 实际 vx、`/cmd_vel` 的 vx / vy、发给运控的 vx（`/mc/motion_control`）、影子模式的两条 | 实际跟着指令；限速时指令不超过限速 | 指令有、实际没有：运控没接（遥控接管了？） |
| 角速度曲线 | 实际 ω、指令 ω | align 时指令 ω ≤ 0.5 rad/s | |
| 侧栏 "规划的数"（`/planning/debug/plotjuggler/string`） | `crosswalk_mode`（0 none 1 approach 2 align 3 wait 4 go 5 cross）、`crosswalk_index`、`crosswalk_s_stop`、`crosswalk_hold`、`crosswalk_go`、`crosswalk_vehicles`、`crosswalk_waiting_for`、`crosswalk_stale`、`crosswalk_unreleased`、`crosswalk_heading_error`；`cmd_vx` 等 | | `crosswalk_stale` 一直 1：感知断了或时间对不上 |
| 诊断 | 规划的 fault | 空 | |
| 感知的原始目标（`/perception/lidar_3d/objects`） | 每个目标的 id、类别、motion、age、速度 | | |

full 级别多看的：

- 定位：`ins_path`（INS）和 `lio_path`（激光里程计）应该叠在一起；`current_scan_world` 应该贴在 `map_cloud` 上；
- 感知：`tracks` 和 `predictions` 是感知自己的框和预测（`camera_init` 系，不在狗的 TF 树里：3D 面板的显示坐标系选 `camera_init` 看；
  开了 `debug perception` 也只有 `camera_init → lidar_det`）。规划实际用的车位置看 `/planning/debug/crosswalk`（已经转到 map）。

---

## 7. 调参：在哪改、怎么生效

**仓库是源头**。狗上改装好的那份只为现场快速试；试好的值抄下来（文件、行、旧值、新值、为什么）回来改仓库、提交，
下次装（T56 的第 2、3、5 步）就带上了。狗站着的时候改，改完重启对应的服务。
改了代码要重编：狗上编（T56 第 3 步，一个多小时，占狗的 CPU），或者在服务器上狗的系统镜像里编（T60：`Nav_D20/docker/dog_rootfs.sh`、`run_arm64.sh`，QEMU，不占狗）。重装会盖掉装好的配置：之后要重跑 `nav_cw.sh configure`、再 `nav_cw.sh apply` 把开关写回去（T60 第 7 步）。

| 模块 | 文件（装好的；`find /opt/seres/nav_cw/install/<模块> -name <文件> -not -path '*/src/*'`） | 常调的 | 生效 |
|---|---|---|---|
| 规划 | `robot_factory/d20/001/planning_conf/mission_conf.pb.txt` | `crosswalk_yield_conf`（`enable`、`yield_rule`、`stop_margin`、`slow_distance`、`align_tolerance_deg`、`max_align_omega`、`stale_after`、外参 `det_to_base_*` 等，CROSSWALK_YIELD_FLOW.md 第 8 节）；`mppi_planner_conf` 的速度（用 `nav_cw.sh speed`） | `systemctl restart cw-planning`，重发目标 |
| 规划 | `…/preprocess_conf.pb.txt` | `global_map_manager_conf.startup_map_dir`（`nav_cw.sh configure` 写的）、代价地图 | 同上 |
| 规划 | `…/communication_conf.pb.txt` | `pub_flag`、`enable_global_map_pub`、话题名 | 同上（`pub_flag` 用 `debug planning` 不用重启） |
| 定位 | `lightning/config/default_d20.yaml` | `map_path`（`configure` 写的） | `restart cw-slam`；狗要站着，等重新收敛；这期间规划暂停 |
| 感知（我们的） | `lidar_3d_detection/config/lidar_3d_detection.yaml` | `score_thresholds`、`tracker`、`predictor`、`node.ins_azimuth_offset_deg`（INS 朝向偏置，现在 0） | `restart cw-nav-perception` |
| 感知（机器人团队的） | `lidar_seg` 的 `lidar_seg.yaml`、`grid_map_memory_node.yaml` | 分割、栅格 | 同上 |

注意：

- `PLANNING_TIMEOUT_WALL_CLOCK` 狗上**不设**（只给回放旧包用）；
- 改了外参（`det_to_base_*` 或感知的 `T_body_lidar` / `extrinsic_rpy`）一定录一段包跑 `objects_in_map.py`，要 "extrinsic OK"；
- 门控的大多数参数（`YieldConfig`：`wait_clear`、`lost_hold`、`young_age` 等）不在配置文件里，是代码里的默认值，改要改代码重编（CROSSWALK_YIELD_FLOW.md 第 8 节）。

---

## 8. 规划和运控怎么安全地试

规划能动狗的只有两个话题：`/mc/motion_control`（运控实际执行的）和 `/cmd_vel`。有任务时每个周期都发 `/cmd_vel`，
不在 standstill 时还发 `/mc/motion_control`；人行横道等车时发的是 (0, 0, 0) 或对朝向的原地转（任务没停，所以照样发）。
Foxglove 只能发 `/goal_pose`。一层一层来，前一层过了再下一层：

1. **狗站着，先摸清运控**（第一次上板，限速 `speed 0.3`，影子关，急停在手）：在 2 m 外发一个目标，狗开始走以后依次试：
   - 遥控能不能接管、松手以后谁接着发；
   - `systemctl stop cw-planning`：狗是停下还是继续按最后的速度走（决定它能不能当软停用）；
   - 急停按下、松开以后狗的状态。
   结果写进 T57 的报告。
2. **域 42 回放**（狗上或服务器上，第 3.2 节）：新的代码、新的配置先过一遍包，`check_planning_mcap.py`、`planning_mcap_viz.py` 看没问题。
3. **影子模式**（`shadow on`，输出里图那几行都是 ok 才往下；`use-new` 以后再 `check` 一次）：规划在真狗、真传感器上跑，指令不执行。人用遥控按规划的意图遛狗：
   - Foxglove 速度曲线里绿线是规划想发的，和人走的比；
   - 遛到人行横道：状态应该 approach（快到停止点）→ 人在入口前停下 → align（人把狗转到面朝对面）→ wait / go；
     有车经过时看车框变红、`waiting for` 有它，车走 3 s 后放行；这一步不冒任何险就能验感知 + 门控；
   - 狗没按指令动，规划可能判成被挡（`nav_blocked` → standstill），重发目标即可。
4. **限速实走**（`shadow off`，`speed 0.5`，再 `0.8`）：空场地、空的人行横道，急停在手：approach → align → go → cross → none，
   入口前停下、转向、走过去、中间不停。门控用的是"等"规则，只要看得到在动、可能经过的车就等，不靠狗的速度算来不来得及，
   所以限速不影响等不等车，只是过得慢。
5. **同事开车**（≤ 10 km/h，从左、从右）：车在可能经过的范围里时 wait（理由 vehicle），车过去 3 s 以后 go。
6. **正常车流**：人判断有没有不该放的时候放了。

一直有效：

- 任何时候人觉得不对就急停，记下时间（回来对录包）；
- 一次测试一个录包（第 9 节）；
- 出错、收工都 `use-original`；狗重启也会回到原来的；
- 不在狗的域 0 上发任何话题，回放只用域 42；
- 改了代码或配置，先回到第 2 层。

---

## 9. 录包和回来分析

狗上（域 0，录，不发）：

```bash
ros2 bag record -s mcap -o /home/nvidia/bags/<日期>_cw_<n> \
  /robot/topic/rdap/top_point_cloud2 /rte/rdap/rtk/inspvax /rte/rdap/rtk/imus/data_raw \
  /lightning/odom /map/occupancy_grid /perception/lidar_3d/objects /goal_pose \
  /cmd_vel /mc/motion_control /nav_cw/shadow/cmd_vel /nav_cw/shadow/mc/motion_control \
  /planning/debug/crosswalk /planning/debug/path_tracking /planning/debug/plotjuggler/string /tf /tf_static
```

（T57 第 3 步是同一张表；激光和 INS 要录上，回来才能重算感知。）拷回服务器 `/public/zjst/robot/cp_data/`，然后：

```bash
python Nav_D20/planning/tools/check_planning_mcap.py <包>          # 规划的输入输出、状态
python Nav_D20/planning/tools/planning_mcap_viz.py <包> --map Nav_D20/map/ad68517c-e845-42a4-b5dd-a6b7c6d2af13.zip
python Nav_D20/planning/tools/objects_in_map.py <包>               # 外参：要 extrinsic OK
journalctl -u cw-planning --since "<开始>" --until "<结束>" > planning.log   # 狗上，状态变化的那些行
```

---

## 10. 出问题时

| 现象 | 先看 | 多半是 |
|---|---|---|
| `use-new` 退出 1 | 它打出的三个服务最后 15 行 | 库没找到（`LD_LIBRARY_PATH`）、配置读不到；它已经换回原来的了 |
| `startup map … 0 crosswalks` 或 `Map switch` 到别的图 | `switches` 的 appmap；`check` | appmap 是 follow（App 发的是 c1adbe7b），或者 ignore 没写进装好的配置（NOT APPLIED：`apply`）；换回 |
| `shadow on` 退出 3，`SHADOW NOT IN EFFECT` | 它打出的图 | 装的规划没按配置改话题名（版本不对？）；它已经把 `cw-planning` 停了，别再起，查清楚 |
| `use-new` 退出 1，图那几行有 WRONG | 哪一行 | 开关和实际不符（launch 是旧的、没读 `nav_cw.env`）；它已经换回了 |
| 狗不走、`mission_state: standstill` | `journalctl` 里的超时、`nav_blocked` | 里程计或栅格断流（暂停 1 s、恢复 3 s）；被挡；影子模式没关；限速设成了 0 |
| 一直 `wait (stale)` | `ros2 topic hz /perception/lidar_3d/objects`；`crosswalk_stale` | 感知没起来；感知和里程计的时间对不上（`max_pose_gap` 0.3 s） |
| 一直 `wait (vehicle)`，路上没车 | Foxglove 里红框在哪 | 停着的车被当成在动（速度噪声、INS 朝向不对）；路边的车被预测"可能拐过来" |
| 车框跟着狗转 | `objects_in_map.py` | 外参不对 |
| Foxglove 连不上、桥退出 | 桥的输出 | 不是 root；白名单放太多（iceoryx）；端口 |
| 看不到规划的调试话题 | `debug planning on` | `pub_flag` 关着 |
