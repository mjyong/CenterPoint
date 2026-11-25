# 规划调试用的小包（`20261115_x6_clip`，37.7 s，5.3 MB）

从上一级的切片（`20261115_x6_clip_0.mcap`，325 MB，要 Git LFS）用 `Nav_D20/planning/tools/planning_debug_bag.py` 做的。只留规划要的东西，
普通 git 文件，`git pull` 就有，不用 LFS。

```bash
python3 Nav_D20/planning/tools/planning_debug_bag.py data/examples/20261115_x6_clip \
  -o data/examples/20261115_x6_clip/planning_debug --name 20261115_x6_clip_planning_debug
```

## 里面有什么

| 话题 | 类型 | 条数 | 说明 |
|---|---|---|---|
| `/lightning/odom` | `nav_msgs/msg/Odometry` | 3770 | 规划的输入，原名 |
| `/map/occupancy_grid` | `nav_msgs/msg/OccupancyGrid` | 377 | 感知栅格，输入 |
| `/perception/lidar_3d/objects` | **`nav_interface/msg/LidarObjectArray`** | 377 | 感知的车。现场录的是 `perception_interface/msg/LidarObjectArray`，字段一样，字节原样，只换了类型名。新规划订的就是这个 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 4524 | 只有 `map -> base_link`。现场规划自己发的 `base_link -> ptz_*` 去掉了，回灌的规划会自己发 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | 1 | `base_link -> hesai_lidar`、`lidar_det`，transient local（和现场录的一样） |
| `/goal_pose` | `geometry_msgs/msg/PoseStamped` | 1 | **map 里**的 (80.725, 216.437)，朝向 152.1°，在包开头 |
| `/field/planning/debug/*`（plotjuggler/string、crosswalk、path_tracking、goal_pose、diagnostic、代价地图、全局栅格） | 同原来 | 各 754 | 狗上的规划当时算出来的，改了名，回灌时不和新规划发的撞 |
| `/field/nav_cw/shadow/cmd_vel`、`/field/nav_cw/shadow/mc/motion_control` | 同原来 | 各 754 | 狗上的规划当时的影子指令 |

激光、四路鱼眼、IMU、INS 都不在这里：要看去上一级的切片，或者 Foxglove 复盘（`docs/DOG_FIELD_GUIDE.md` 9.1）。

**终点**：切片里的 `/goal_pose` 是 T66 那种在 base_link 里发的 (27.00, 4.03)，被切片重新打成了切片开头的时间。这里用切片开头
狗的位姿转进 map，得到 (80.725, 216.437)。和现场规划的全局路径终点 (80.75, 216.45) 差 3 cm，和 T66 日志里规划自己转的
(80.72, 216.44) 也对得上：发目标以后狗基本没动。现场规划的 `/field/planning/debug/goal_pose` 画在 (27.0, 4.0)，是还没修的
T66 标记问题（仓库里已修，狗上的规划还是旧的），不用管。

## 这一段发生了什么（现场的规划）

狗一直站在 6 号人行横道（规划 crosswalk 0）的入口上（x 106.8–107.1，y 209.0–209.2，最快 0.38 m/s）。人在遥控，影子模式下规划只算
不发：

| 包内时间 | 现场规划 | 感知 |
|---|---|---|
| 0–1.7 s | cross (on_crosswalk)，有指令 | 停着的重卡 1316、1323 |
| 1.7–11.25 s | cross_wait (ahead → clearing)，等 1 辆，指令 0 | 重卡 1327 从左边过来，在狗前方约 4 m 横穿 |
| 11.25–18.9 s | cross，有指令 | 路口空 |
| 18.9–37.35 s | cross_wait，等 1–3 辆，指令 0 | 1333、1338、1340、1342 一波车 |
| 37.35 s | cross | |

- 指令为 0 的 560 个周期全在 cross_wait 里；cross 的 194 个周期都有指令。cross 的时候狗前方 30 m 内没有在动的车，没有可疑的放行。
- cross 和 cross_wait 没有来回翻（T66 在 13 号是来回翻的）。
- 感知：15 条车的航迹，没有一辆是 12 m 以内才出现的。只有两个闪框：#1328（46 m，1 帧）、#1341（38 m，3 帧），都远。
- 规划的局部代价地图在狗脚下一直有 32–36 格障碍，都在狗后半身（中心往后 0.27–0.47 m）。这里是人行横道入口，路网的边。感知栅格
  脚下是空的。局部规划照样出指令，这一段不卡；回灌时如果规划起步就卡住，先看这里。

## 回灌给规划

一台编好规划的机器上（nav-dev 镜像，或者 micromamba 环境），用域 42，不碰狗：

```bash
git lfs pull --include="Nav_D20/map/ad68517c-e845-42a4-b5dd-a6b7c6d2af13.zip"     # 站点地图（206 MB），已经有就跳过
mkdir -p data/loc_maps && unzip -qn Nav_D20/map/ad68517c-e845-42a4-b5dd-a6b7c6d2af13.zip -d data/loc_maps
PLANNING_STARTUP_MAP=data/loc_maps/ad68517c-e845-42a4-b5dd-a6b7c6d2af13 \
  bash Nav_D20/planning/script/replay_planning_check.sh x6_clip 80.725 216.437 \
    --bag data/examples/20261115_x6_clip/planning_debug --planning-only
```

- `--planning-only` 只起规划：不起地面分割、栅格、定位，包里的里程计、栅格、目标、TF 直接给规划。
- 规划先起，等它打出 `startup map … 5 crosswalks`（读站点地图要二十多秒，最多等 `NAV_PLANNER_WAIT`，默认 90 s）再放包，37.7 s 的
  开头不丢。
- 包里带朝向的 `/goal_pose` 就是这次的目标，脚本不再另发。
- 规划用的是你本机编的版本：订阅 `nav_interface/msg/LidarObjectArray`（2026-11-25 以后的规划），接口包要先编好，不然 `ros2 bag play`
  会跳过这个话题。
- 结果在 `work_dirs/planning_io_x6_clip/`：录包、`report.txt`（状态切换、指令）、`viz/planning_viz.mcap`。

**和现场会不一样的地方**：
- 开头狗只过了入口几厘米（T66 记的是 0.06 m、0.007 m）。回灌的规划从头起，没有放行就在人行横道上，这算"浅"的 unreleased：
  按路边处理，先走 align / wait / go，不是一上来就 cross。现场那一次在切片之前已经放行了。
- 回灌的规划带了 306cd0a 的入口线抖动修复，现场的没有。

## 在 Foxglove 里看

- **只看包**：Open local file 选 `20261115_x6_clip_planning_debug_0.mcap`，Layout → Import from file 选
  `foxglove_planning_debug.json`。能看到输入和狗上的规划（`/field/…`）。
- **回灌的时候**：Foxglove 连上回灌机器的 foxglove_bridge（域 42）。同一个布局里两套叠着显示：狗上的规划在 `/field/planning/debug/*`
  （包里放出来的），回灌的规划在 `/planning/debug/*`。在 3D 左边的话题列表里关掉一套，看另一套。
  - 速度图有三条线：实际、狗上的规划、回灌的规划；
  - 右边标签页把两边每周期的数（`crosswalk_mode` 等）并排放着。
