# 录制数据元信息

专门登记狗上采回来的 rosbag：在哪里、录了什么、和后面场景 / 训练数据怎么对应。
采集怎么做见 [SENSORS.md](SENSORS.md) 第 7 节；切场景和体检见 [BAG_SPLIT.md](BAG_SPLIT.md)；缺什么、下一批怎么采见 [DATA_COLLECTION_PLAN.md](DATA_COLLECTION_PLAN.md)。

**当前共两批可用数据**：首批 `0921`、本批 `1104`。`1016` 因点云话题录错整批作废，不算在内。

仓库里的软链（原始 bag 在公共盘，不进 git）：

| 软链 | 指向 |
|---|---|
| `bags0921`（本仓库根、以及 `d20car_predict/bags0921`） | `/public/zjst/robot/rosbag0921` |
| `bags1104`（本仓库根、以及 `d20car_predict/bags1104`） | `/public/zjst/robot/bags1104` |

`/zjst/robot/...` 与 `/public/zjst/robot/...` 是同一块 NFS。

---

## 总览

| 批次 | 日期 | 公共盘 | 录制份数 | 合计时长 | 体积 | 雷达帧 | 状态 |
|---|---|---|---|---|---|---|---|
| 首批 0921 | 2026-09-21 | `rosbag0921/` | 2 | 约 18.3 min | 35 GB | 10 971 | 已切成 35 个场景，训练 / 验证已用 |
| 本批 1104 | 2026-11-04 | `bags1104/` | 4（cp01–cp04） | 约 17.6 min | 35 GB | 10 565 | 已落盘，待体检 / 切场景 |
| （作废）1016 | 2026-10-16 | `/public/zjst/robot/cp_data/` | — | — | — | — | 点云话题录错，整批取消；仓库里不再挂软链 |

两批都是车顶 Hesai XT32 + INS/IMU。1104 起按 SENSORS 第 7 节把四路鱼眼（含后视）和雷达丢包计数一并录上。

---

## 1. 首批：0921

| 项 | 内容 |
|---|---|
| 日期 | 2026-09-21 |
| 机器人 | kygi665，车顶 XT32 |
| 原始路径 | `/public/zjst/robot/rosbag0921/`（软链 `bags0921`） |
| 切出的场景 | `data/seq/0921_*`，共 35 个；划分见 `data/survey/split/split.csv` |
| 训练 / 验证 | 训练 25（`0921_165659_s000–s002`、`s007–s028`），验证 10（`0921_164127_s000–s005`、`0921_165659_s003–s006`） |
| 已导出数据集 | `data/dog_veh` → `/public/zjst/robot/centerpoint_data/dog_veh_1026_1410`（v2） |
| 已知问题 | 录制时主、从 GNSS 天线接反（航向偏约 180°）；无相机话题。详见 [SENSORS.md](SENSORS.md) 3.2、[BAG_SPLIT.md](BAG_SPLIT.md) 第 5 节 |

### 1.1 原始 bag

| 目录 | 开始时间（本地） | 时长 | 体积 | 雷达帧 | 话题 |
|---|---|---|---|---|---|
| `rosbag2_2026_09_21-16_41_27` | 16:41:28 | 172.8 s（约 2.9 min） | 5.7 GB | 1 728 | 雷达、IMU、INSPVAX |
| `rosbag2_2026_09_21-16_56_59` | 16:56:59 | 924.4 s（约 15.4 min） | 29 GB | 9 243 | 同上 |

话题名与现在相同：

- `/robot/topic/rdap/top_point_cloud2`（约 10 Hz）
- `/rte/rdap/rtk/imus/data_raw`（约 100 Hz）
- `/rte/rdap/rtk/inspvax`（约 100 Hz）

场景名前缀：`0921_164127`、`0921_165659`。

---

## 2. 本批：1104

| 项 | 内容 |
|---|---|
| 日期 | 2026-11-04（晚上约 20:02–20:23） |
| 机器人 | kygi665，车顶 XT32；四路鱼眼都录（含后视） |
| 原始路径 | `/public/zjst/robot/bags1104/`（软链 `bags1104`） |
| 录制命名 | 板上 `/home/nvidia/bags/<NAME>`，本批为 `cp01`–`cp04` |
| 体积 | 合计约 35 GB |
| 状态 | 2026-11-05 已拷到公共盘；**尚未**跑 `bag_survey` / 切场景 |

### 2.1 各份 bag

| 目录 | 开始时间（本地） | 时长 | 体积 | 雷达帧 | 雷达 / IMU / 前视相机 |
|---|---|---|---|---|---|
| `cp01` | 20:02:57 | 122.6 s（约 2.0 min） | 4.1 GB | 1 227 | 10.0 / 100 / 25.0 Hz |
| `cp02` | 20:05:57 | 171.3 s（约 2.9 min） | 5.7 GB | 1 713 | 10.0 / 100 / 25.0 Hz |
| `cp03` | 20:09:21 | 517.8 s（约 8.6 min） | 17 GB | 5 178 | 10.0 / 100 / 25.0 Hz |
| `cp04` | 20:18:59 | 244.8 s（约 4.1 min） | 8.1 GB | 2 447 | 10.0 / 100 / 25.0 Hz |

每份一个 `*_0.db3` + `metadata.yaml`。

### 2.2 话题（四份相同）

| 话题 | 类型 | 约频率 | 用途 |
|---|---|---|---|
| `/robot/topic/rdap/top_point_cloud2` | `sensor_msgs/PointCloud2` | 10 Hz | 检测 / 训练 |
| `/rte/rdap/rtk/imus/data_raw` | `sensor_msgs/Imu` | 100 Hz | 去畸变 |
| `/rte/rdap/rtk/inspvax` | `kygi665_msgs/INSPVAX` | 100 Hz | 位姿 |
| `/robot/topic/rdap/front_sensor_camera` | `foxglove_msgs/CompressedVideo` | 25 Hz | 审核 |
| `/robot/topic/rdap/left_sensor_camera` | 同上 | 25 Hz | 审核 |
| `/robot/topic/rdap/right_sensor_camera` | 同上 | 25 Hz | 审核 |
| `/robot/topic/rdap/rear_sensor_camera` | 同上 | 25 Hz | 审核（0921 没有） |
| `/lidar_packets_loss` | `hesai_ros_driver/LossPacket` | 与雷达同频 | 录制质量 |

相对首批的变化：四路鱼眼（含后视）和丢包计数都录了；天线应按 SENSORS 现行接法（不再接反）。区域、天气、路线等现场记录若有，补进本节。

### 2.3 下一步

按 [BAG_SPLIT.md](BAG_SPLIT.md)：体检 → 切场景 → 预览 / 位姿融合；场景前缀建议 `1104_cp01` … `1104_cp04`（或按开始时刻命名，与 0921 一致）。和 0921 的合并、标注计划见 [DATA_COLLECTION_PLAN.md](DATA_COLLECTION_PLAN.md) 第 4 节。

---

## 3. 不作数的录制

| 批次 | 说明 |
|---|---|
| 1016 | 前置 Livox Mid-360 + 三路鱼眼；点云话题录错，整批取消。公共盘仍在 `/public/zjst/robot/cp_data/`，仓库不再挂软链。见 [BAG_SPLIT.md](BAG_SPLIT.md) 3.6、5.15，[RETROSPECTIVE.md](RETROSPECTIVE.md)。 |

---

## 4. 以后怎么追加

每来一批新 bag：

1. 放到 `/public/zjst/robot/<名字>/`，在本仓库根建同名软链（需要时在 `d20car_predict/` 下也建一份）；
2. 在本文「总览」加一行，并新开一节写日期、路径、各 bag 时长 / 体积 / 话题 / 已知问题；
3. 现场元信息（区域、天气、可控场景记录表）一并写进该节。
