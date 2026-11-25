# Dog #8 六号斑马线在线测试切片（`20261115_x6_clip`，37.7 s 双轮过车让行片段）

本目录是从服务器完整在线实测包 `/public/zjst/robot/cp_data/20261116_dog/20261115_x6_all_1/`（`394.1 s`，`14.8 GB` SQLite3 `.db3`）中截取的 **`37.7 s`（原包 `t = 239.10 s .. 276.80 s`）完整过车与让行闭环切片**，经 Zstandard 无损压缩为 ROS 2 MCAP（`311 MiB` / `325 MB`），供远程克隆仓库后直接在 **Foxglove Studio** 中回放查看。

---

## 1. 片段内容与时间轴（相对于切片起点 `0.0 s`）

本片段完整保留全部 **24 个话题**（禾赛 XT32 原始 32 线点云、前/左/右/后四路 H.264 鱼眼相机、CenterPoint 3D 车辆检测与跟踪、路口让车规划可视化与状态机、全局路网栅格、里程计、RTK 组合导航与完整 TF 树），涵盖**两轮完整的“检测来车 → 停等让行 → 清空恢复通行”闭环**：

| 切片内时间 | 原包时间 | 过往车辆与 3D 感知跟踪 | `cw-planning` 路口让车状态切换 |
| :--- | :--- | :--- | :--- |
| `0.0 s .. 1.7 s` | `239.1 .. 240.8 s` | 路侧停放重卡 `ID 1316`、`ID 1323` 稳定跟踪，斑马线无来车 | `cross (on_crosswalk)`（正常通行） |
| `1.7 s .. 8.3 s` | `240.8 .. 247.4 s` | **重卡 `ID 1327`**（车长 `12.1 m`，`8.5 m/s`，连续 `112` 帧零丢帧）由左侧 `-38 m` 驶入，在 `t = +6.4 s`（原包 `245.5 s`）距狗正前方 **`4.3 m`** 横穿斑马线，左→前→右鱼眼相机与 3D 点云同步清晰可见 | `cross_wait (ahead), waiting for 1`（主动让车停等） |
| `8.3 s .. 11.2 s` | `247.4 .. 250.35 s` | `ID 1327` 驶离右侧 `+41 m` | `cross_wait (clearing), waiting for 1` → **`cross (on_crosswalk)`**（`+11.25 s` 恢复通行） |
| `11.2 s .. 18.9 s` | `250.35 .. 258.0 s` | 路口净空，`27.7 m` 全局路径保持激活 | `cross (on_crosswalk)` |
| `18.9 s .. 37.4 s` | `258.0 .. 276.45 s` | **多车连续横穿波次**（`ID 1333` 连续 `130` 帧、`ID 1338` 连续 `134` 帧，及 `ID 1340`、`ID 1342`）依次通过斑马线 | `cross_wait (ahead), waiting for 1 → 2` → `cross_wait (clearing), waiting for 3 → 2 → 1` → **`cross (on_crosswalk)`**（`+37.35 s` 恢复通行） |

> **切片对齐处理说明**：
> - 四路鱼眼 H.264 视频流（`*_sensor_camera`）均从首个 IDR 关键帧（`+0.05 s .. +0.10 s`）起写入，在 Foxglove 中从第 0 秒起即可直接解码画面，无花屏或黑屏。
> - 已在切片起点补齐原包前序锁存的全局栅格地图 `/planning/debug/global_map_manager/occmap`、目标点 `/goal_pose` 以及 `/tf_static`（`base_link -> hesai_lidar` 与 `base_link -> lidar_det`），点云、3D 车框与全局地图在 `map` 坐标系下零报错直接对齐。

---

## 2. 文件清单

| 文件 | 大小 | 存储方式 | 用途 |
| :--- | :--- | :--- | :--- |
| `20261115_x6_clip_0.mcap` | `311 MiB` | **Git LFS** | 24 话题全量原始传感器 + 四路相机 + 3D 点云 + 感知 + 规划切片 |
| `metadata.yaml` | `6 KB` | 普通 Git | ROS 2 bag 目录元数据（支持 `ros2 bag info/play` 与 `rosbags` 目录读取） |
| `foxglove_x6_clip.json` | `7 KB` | 普通 Git | 配套 Foxglove 布局（含 3D 点云/车框/路网 + 左/前/右三路鱼眼相机 + 速度/状态面板） |
| `viz/20261115_x6_clip_planning_viz.mcap` | `4.1 MiB` | 普通 Git（免 LFS） | `planning_mcap_viz.py` 生成的防闪烁规划 + 矢量斑马线可视化包（未装 Git LFS 也能直接打开） |
| `viz/20261115_x6_clip_planning_viz.png` | `130 KB` | 普通 Git | 轨迹与斑马线鸟瞰总览图 |

---

## 3. 远程拉取与 Foxglove 打开方式

### 方式 A：看全量点云 + 四路实景鱼眼视频 + 3D 车框（推荐）

1. 拉取代码及本目录的 Git LFS 数据文件：
   ```bash
   git pull origin claude/me/v01
   git lfs pull --include="data/examples/20261115_x6_clip/**"
   ```
2. 打开 **Foxglove Desktop**：
   - 点击 **Open local file**，选择 `data/examples/20261115_x6_clip/20261115_x6_clip_0.mcap`
   - 右上角 **Layout → Import from file...**，导入同目录下的 `data/examples/20261115_x6_clip/foxglove_x6_clip.json`（或使用 `Nav_D20/planning/script/planning_live.json`）

### 方式 B：免 Git LFS 秒开轻量规划状态回放（仅 `4.1 MiB`）

无需拉取 LFS 大文件，`git pull` 后直接在 Foxglove Desktop 中：
- **Open local file** → 选择 `data/examples/20261115_x6_clip/viz/20261115_x6_clip_planning_viz.mcap`
- **Layout → Import from file...** → 选择 `Nav_D20/planning/script/planning_viz.json`

### 方式 C：只给规划调试的小包（`planning_debug/`，5.3 MB，免 Git LFS）

规划的输入（里程计、感知栅格、感知目标、TF、map 里的终点）加上狗上规划当时的输出（改名到 `/field/…`）。感知目标已经是
`nav_interface/msg/LidarObjectArray`。可以直接 `ros2 bag play` 给规划，也可以用 `replay_planning_check.sh … --planning-only`
回灌。说明和这一段的分析见 `planning_debug/README.md`。
