# 定位优化：流程、数据链路和详细设计

kygi665 机器狗的定位从哪里来、哪里不对、怎么修，离线（做训练数据）和在线（车上节点）各怎么用。这一份是定位的总说明，细节分散在别的文档里的，这里都收拢了：

| 文档 | 里面有什么 |
|---|---|
| [BAG_SPLIT.md](BAG_SPLIT.md) 3.5、5.4–5.10 | 位姿融合每一轮的原始记录和数字 |
| [DATA_LINEAGE.md](DATA_LINEAGE.md) | 从 bag 到训练集的完整数据链路（定位只是其中一部分） |
| [SENSORS.md](SENSORS.md) 1、3、6 | 外参、组合导航、双天线、授时 |
| [DOG_DATAFLOW.md](DOG_DATAFLOW.md) 3 | 在线位姿通路的调用链和数组形状 |
| [DEPLOY_ORIN.md](DEPLOY_ORIN.md) 8 | C++ 节点的消息处理 |

代码位置：`tools/dog/refine_poses.py`（离线融合）、`dog_perception/inspvax.py`（INSPVAX）、`dog_perception/pose_buffer.py`（位姿缓存和 IMU 传播）、`dog_perception/preprocess/`（去畸变、检测系）、`dog_perception/online.py`（在线消息处理）；C++ 在 `Nav_D20/perception_combo_ros/node_src/lidar_3d_detection/`（下文 `$L`）的 `lidar3d/localization.hpp`、`pose_buffer.hpp`、`online.hpp`。

---

## 0. 一页总结

**为什么要优化**：kygi 的组合导航（INSPVAX）在 0921 的录制上有几个问题，多帧叠加和自动标注都会被它带偏：

| 问题 | 大小 | 后果 |
|---|---|---|
| 急停时位置超前 | 约 0.45 m，保持 1 s 以上，之后每 0.2 s 往回拉一点，拉不够 | 5 帧叠加出重影；静止物体的框和点云错开 |
| 报的位置不在雷达上（杠杆臂） | 雷达后方约 0.54 m 的一个点，原地掉头差 1.1 m | 一转弯位置就绕着机身甩 |
| 快速转弯后航向偏 | 4–10°，3–5 s 才回来 | 10 m 处每度约 0.17 m |
| 横滚、俯仰不可信 | 急停时 pitch 0.6 s 内从 +2.8° 摆到 −9.9°，陀螺积分不到 2° | 不能拿来找平 |
| 狗本身倾斜大 | 相对平常最大 15–24°，1 s 内变 8–14°（路沿、台阶、坡） | 只靠地面找平（最多 5°）跟不上 |

根子在硬件：**0921 录制时双天线主、从接反了**（机器人团队确认）。航向因此转了 180°，INS 把水平加速度投到了反方向，急停就当成加速；天线杠杆臂也按反了的轴去扣，输出点落到了 IMU 后面。1016 起天线已经接正。

**做法**：
- **离线**（训练数据）：`refine_poses.py` 把 KISS-ICP 激光里程计（陀螺给旋转先验）和 RTK 融合。两边各自判健康：都健康时激光管帧到帧的运动，RTK 只消慢漂移；一边坏了另一边接管；两边都坏就标成"不可靠"，这些帧不标注、不进训练集。航向用激光加一个向 INS 慢慢靠拢的偏移，倾斜用激光的。
- **在线**（车上节点）：只用 INS 的位置和航向（扣掉杠杆臂），IMU 陀螺补两次定位之间的旋转，用来去畸变；倾斜靠检测系的地面找平。训练时用 `SweepPoseJitter` 模拟在线的位姿误差。

**效果**：
- 运动样例 bag，相隔 1 s 的帧对，激光配准量出的位姿偏差中位数 0.25 → 0.06 m（p90 0.55 → 0.14 m）；
- 0921 的 35 个场景，融合位姿的 `shift` p90：中位数 0.072 m，最大 0.155 m，没有超过 0.2 m 的（第一版时 4 个转角场景 0.49–0.61 m）；
- 预览：165659 是 22 OK / 6 WARN / 1 FAIL（FAIL 是坡道上的 `tilt_deg`，不是位姿），164127 全部 OK；
- 位姿不可靠的帧占 1.7%（178 帧），自动标注和数据集都跳过。

**还没解决的**：在线没有倾斜（大倾斜时在线点云和训练数据不一样）；在线的急停超前要等天线接正后的录制来确认还有没有；接反时的杠杆臂还有约 0.1 m 没解释（T10）；**车上配置还是 0921 的天线接法，上车前要改**（第 10 节）。

---

## 1. 坐标系、传感器和约定

### 1.1 坐标系

| 记号 | 名称 | 定义 |
|---|---|---|
| L | 雷达系 | XT32 自己的系（`frame_id: hesai_lidar`）：x 左、y 后、z 上 |
| B | 机身系 | 轴和 RTK IMU 相同：x 右、y 前、z 上，原点在雷达。L → B 绕 z 转 180°（`KYGI_LIDAR_RPY`）。机器人本体系（x 前、y 左）是 B 绕 z 转 +90° |
| W | 世界系 | ENU（东、北、天），原点是这份录制第一个 `ins_status = 3` 的定位点；同一份录制的所有场景共用。在线节点里叫 `camera_init` |
| D | 检测系 | 重力对齐，航向跟机身，原点在机身正下方的地面（机身原点下移 `base_height`），再用近处地面在线修正高度和倾斜 |

位姿都写成 `T_world_body`（B → W）。`odom.npy` 存的是 W 系里机身原点的位置、姿态和速度。

### 1.2 传感器（定位用到的）

| 传感器 | 话题 | 频率 | 用到什么 |
|---|---|---|---|
| 禾赛 XT32 | `/robot/topic/rdap/top_point_cloud2` | 10 Hz | 点云和每个点的绝对时间（`timestamp`，float64）；整帧时刻取最后一个点（扫描结束） |
| RTK IMU | `/rte/rdap/rtk/imus/data_raw` | 100 Hz | 角速度。**单位是 deg/s**，不是 sensor_msgs 规定的 rad/s；轴就是 B 系。加速度单位是 g，不用 |
| 组合导航 | `/rte/rdap/rtk/inspvax` | 100 Hz | `ins_status`、`pos_type`、经纬高、东北天速度、`azimuth` |

INSPVAX 的字段怎么用：
- `ins_status`：3（INS_SOLUTION_GOOD）才用，其余丢掉；
- `pos_type`：56 是 RTK 固定解（实测位置标准差 1.3 cm）。`kygi665_msgs` 里的常量名把 55 / 56 写反了，代码按 NovAtel 标准以 56 为固定解；
- 经纬高：转成 ENU（WGS-84 → ECEF → 原点处的 ENU 旋转），再按杠杆臂挪到机身原点；
- 速度：东、北、天，直接是 W 系；
- `azimuth`：从正北顺时针。机身航向 `yaw = 偏置 − azimuth`（换成弧度，W 系里从东向逆时针）。偏置 180°（天线接反，0921）或 0°（接正，1016 起）；
- `roll`、`pitch`：**不用**。姿态只有航向（四元数只有 z 分量）。

### 1.3 外参和常数（`dog_perception/inspvax.py`）

| 常数 | 值 | 来源 |
|---|---|---|
| `KYGI_LIDAR_RPY` | (0, 0, π) | 雷达在 B 系的朝向。多初值 ICP 测出（雷达系里狗往 −y 走，INS 说往 +y），和标定一致 |
| `R_BODY_KYGI_IMU` | 单位阵 | IMU 轴 = B 系轴。2026-10 用急刹时的地面倾斜和加速度计核实；之前把陀螺 x、y 取反了 |
| `KYGI_GYRO_SCALE` | π / 180 | deg/s → rad/s。按 deg/s 理解时，陀螺 z 积分和 INS 航向变化的相关是 0.99 |
| `KYGI_AZIMUTH_OFFSET_DEG` | 180 | 0921 的天线接法；接正后是 0 |
| `KYGI_INS_LEVER_ARM` | (−0.06, −0.54, 0) m | 0921（接反）INSPVAX 位置点在 B 系的位置，用 14 个转弯场景拟合，留一法变化 ≤ 3 cm |
| `KYGI_IMU_IN_BODY` | (0.0213, −0.18, 0) m | IMU 中心在 B 系的位置（机器人标定：雷达 → IMU 平移 (−0.0213, 0.18, 0.009) m，绕 z 转 180°）。天线接正后的默认杠杆臂（INS 输出点设在 IMU 中心，机器人团队 2026-10-22 确认） |
| `KYGI_GNSS_ANTENNA_*_IN_IMU` | 前 (0, 0.180, 0.037)、后 (0, −0.474, 0.037) m | 两个天线在 IMU 系的位置，基线 0.6546 m |
| `KYGI_STANDING_BASE_HEIGHT` | 0.72 m | 站立时雷达离地（0921 29 个场景中位数 0.71，样例 bag 0.72） |

杠杆臂怎么选（`default_lever_arm(偏置)`）：偏置 180 用拟合值，偏置 0 用 IMU 中心。接反时还能按天线位置推算出一个点（`ins_point_with_swapped_antennas()`）：INS 把另一个天线当成配置的主天线，再用转了 180° 的航向扣掉配置的天线偏移，两段水平偏移相加，落在 IMU 后方 0.474 − 0.180 = 0.294 m，B 系 (0.021, −0.474) m。这个推算值和拟合值差 0.10 m，方向一致；剩下的 0.1 m 还没解释，现在仍用拟合值（T10 可以比两者的残差）。

### 1.4 时间

- 所有话题都用消息头的时间（`header.stamp`、点的 `timestamp`），不用写进 bag 的时间。
- 雷达的 `header.stamp` 是一帧的开始，比写进 bag 的时间早约 0.10 s（晚一帧才写）；IMU、INS 两者几乎相同。
- 雷达和 INS 的时间差 0921 上验过没有：IMU 陀螺 z 和 INS 航向变化率在 0 延迟处相关最高（0.91）；激光偏航角速度和 INS 航向变化率，18 个转弯场景里 16 个正好在 0，其余 ±0.02 s。
- 有固定时间差时用 `lidar_time_offset` 修正（序列的 `meta.json` 和节点参数都有，第 4.5 节）。

---

## 2. 问题是怎么查出来的

数据本身是唯一可信的依据：单位、外参、时间、杠杆臂、倾斜，每一项都是先怀疑、再用数据验证的。下面按发现的顺序：

| # | 问题 | 怎么发现的 | 处理 |
|---|---|---|---|
| 1 | IMU 角速度是 deg/s | 按 deg/s 理解，陀螺 z 和 INS 航向变化率相关 0.99、比例 1.06 | `gyro_from_kygi_imu` 乘 π/180 |
| 2 | 雷达系和机身系差 180° | 多初值 ICP：雷达系里往 −y 走，INS 说往 +y。静止录制看不出来，运动时旧帧会挪错方向、静止物体被标成在动 | `KYGI_LIDAR_RPY` |
| 3 | 位姿外推只用最后一段，检测系原点跳了 6.6 m | 最后一帧的原点跳变：定位锚点和 IMU 样本时间挨得很近（0.1 ms），几厘米的锚点修正被当成几百 m/s 的速度外推 | `PoseBuffer._extrapolate`：速度取最近 20 ms 的运动，跳过短于 1 ms 的末段 |
| 4 | 机身和安装有倾斜 | 检测系里拟合的地面 pitch −1.7° | 检测系加 `auto_level` |
| 5 | 急停时 INS 超前 0.45 m | 激光帧两两配准：样例 bag 2.1 s 处超前 0.44 m。预览的 `align_m` 看不出（误差沿行进方向，墙大多和路平行，帧沿着墙滑动时墙照样对得上），所以加了 `shift` 检查 | `refine_poses.py`（第 5 节） |
| 6 | INS 位置点在雷达后方 0.54 m | 把每个场景的"INS − 激光"偏移拟合成"常量 + 慢漂移 + R(航向)·L"，14 个转弯场景的 L 一致落在 (−0.06, −0.54) m，残差减半 | 转换、融合、在线三处都扣杠杆臂 |
| 7 | 快速转弯后 INS 航向偏 4–10° | 融合比激光差 0.1 m 以上的帧对：位置只差 0–2 cm，航向在 1 s 内差 1–4°；按时间看，急转（60–90°/s）后 INS 航向偏、3–5 s 回来，这段 INS 位置自检照样"健康" | 第五版：航向用激光 + 慢变偏移 |
| 8 | IMU 的 x、y 轴之前取反了 | 急刹时地面平面显示低头 1.5°，陀螺 x 积分 1.6°（0.1–0.3 s 变化相关 0.85–0.94）；加速度计 y 跟激光的前向加速度走（相关 0.63），x 不相关 | `R_BODY_KYGI_IMU` = 单位阵；老序列 `sequence_meta.py --fix-imu-axes` |
| 9 | 狗倾斜比想象的大 | 第六版诊断：融合加上激光倾斜就和纯激光只差 0.003 m（中位）；激光去掉倾斜就和融合一样差 | 第六版：倾斜用激光的 |
| 10 | 天线接反 | 航向和行进方向差 179.5°；急刹时 INS 速度 1.9 → 2.5 m/s，激光显示 2.1 → 0.1 m/s；杠杆臂对不上标定 | 机器人团队确认并改接；转换器 `--azimuth-offset auto` |

---

## 3. 整体流程

### 3.1 离线：从 bag 到训练数据

```text
rosbag ──① bag_survey.py ──► segments.csv（按空档、RTK 状态切场景）
        │                    survey.png / survey.json（时钟、轨迹、速度）
        ▼
      ② rosbag_to_sequence.py ──► data/seq/<场景>/  scans/ imu.npy odom.npy meta.json
        │   INS：ins_status=3、共用 ENU 原点、航向偏置（自动）、扣杠杆臂、只有航向
        ▼
      ③ sequence_preview.py（第一遍，INS 位姿） ──► align / shift / ins_step / rtk ...
        │   [time_offset_check.py：转弯处扫雷达时间偏移，有固定偏差才写 lidar_time_offset]
        ▼
      ④ refine_poses.py ──► odom.npy（融合位姿）、odom_ins.npy（INS 原样）、meta.pose_refine
        │   KISS-ICP + 陀螺先验；两边健康判定；东北最小二乘；航向偏移；激光倾斜；闭合自检
        ▼
      ⑤ sequence_preview.py（第二遍，融合位姿） ──► shift_p90_m 要降下来
        ▼
      ⑥ split_check.py（训练 / 验证按空间分开，用 odom 的轨迹）
        ▼
      ⑦ run_sequence.py / batch_label.py（预处理 + 教师推理；每帧记 pose_bad）
        ▼
      ⑧ auto_label.py（离线跟踪；pose_bad 的帧不进跟踪、不出框）
        ▼
      ⑨ 审核 → export_dataset.py（跳过 pose_bad 的帧） → 训练（SweepPoseJitter 模拟在线位姿误差）
```

| 步骤 | 工具 | 读 | 写 | 和定位有关的 |
|---|---|---|---|---|
| ① 体检 | `tools/dog/bag_survey.py` | bag 的时间戳和 INSPVAX（不解点云） | `segments.csv`、`survey.*` | 空档（雷达 > 0.3 s、IMU > 0.1 s、INS > 0.2 s）和非 RTK 固定解的地方断开、整段丢掉；时钟检查（log − header） |
| ② 转换 | `tools/dog/rosbag_to_sequence.py` | bag | `data/seq/<场景>/` | 见 4.2 |
| ③ 预览 | `tools/dog/sequence_preview.py` | 序列 | `data/preview/<录制>/` | 见 4.4 |
| 时间差 | `tools/dog/time_offset_check.py` | 有转弯的序列 | `time_offset.*` | 见 4.5 |
| ④ 融合 | `tools/dog/refine_poses.py` | `scans/`、`imu.npy`、`odom_ins.npy`（没有就是 `odom.npy`） | `odom.npy`、`odom_ins.npy`、`meta.json` | 第 5 节 |
| ⑥ 划分 | `tools/dog/split_check.py` | `meta.json`、`odom.npy` | `split.csv`、`meta.segment.split` | 各场景的轨迹按 lat0 / lon0 换到同一 ENU，量空间重叠 |
| ⑦ 教师 | `tools/dog/run_sequence.py` | 序列 | `data/runs/<场景>/` | 和在线一样的预处理（去畸变、检测系）；每帧 `pose_bad` |
| ⑧ 自动标注 | `tools/dog/auto_label.py` | 教师结果 | `data/labels/<场景>/` | `pose_bad` 的帧给跟踪器空检测、不出框 |
| ⑨ 导出 | `tools/dog/export_dataset.py` | 标签 | 数据集 | 跳过 `pose_bad` 的帧 |

### 3.2 一份新录制的完整命令

```bash
export PYTHONPATH=.
BAG=/<路径>/rosbag2_...
NAME=<日期_时刻>                       # 场景名前缀
python tools/dog/bag_survey.py --bag $BAG --out data/survey/$NAME --prefix $NAME
python tools/dog/rosbag_to_sequence.py --bag $BAG --inspvax-topic /rte/rdap/rtk/inspvax \
    --lidar-topic /robot/topic/rdap/top_point_cloud2 --imu-topic /rte/rdap/rtk/imus/data_raw \
    --segments data/survey/$NAME/segments.csv --out data/seq
    # 打印 "INS azimuth offset ... deg" 和 "INS lever arm ... m"，确认是想要的接法
python tools/dog/sequence_preview.py --seq "data/seq/${NAME}_s*" --out data/preview/${NAME}_ins
python tools/dog/time_offset_check.py --seq <几个有转弯的场景> --out data/preview/time_offset   # 可选
python tools/dog/refine_poses.py --seq data/seq/${NAME}_s* --plots data/preview/poses
    # 看每行的 shift p90 ins / lo / fused、"pose bad" 行、最后的 "INS lever arm left over"
python tools/dog/sequence_preview.py --seq "data/seq/${NAME}_s*" --out data/preview/$NAME
```

`refine_poses.py` 总是从 `odom_ins.npy`（第一次运行时存下的 INS 原样）读，可以放心重跑；改了参数、改了杠杆臂，直接再跑一遍就行，不用重转 bag。

### 3.3 在线：车上节点

```text
INSPVAX ──► OnlineInputs.on_inspvax ──► pose_from_inspvax（ENU、偏置、杠杆臂）──┐
             （第一个好解定原点；HeadingCheck；pos_type 不是 56 告警）           │ 锚点（100 Hz）
IMU ──────► OnlineInputs.on_imu ──► gyro_from_kygi_imu ─────────────────────┐ │
                                                                           ▼ ▼
                                                     ImuPropagator ──► PoseBuffer（3 s）
点云 ──────► OnlineInputs.on_cloud ──► Preprocessor：去重、去自身点、去畸变（每 0.1 ms 一个位姿）
                                       → W 系 → 5 帧累积 → 检测系（航向 + 地面找平）→ 检测器
```

第 7 节细说。离线的 ⑦ 教师推理和在线走的是同一套代码（`PerceptionPipeline`），区别只在位姿来源（第 7.5 节）。

---

## 4. 数据链路：每一步对定位做了什么

### 4.1 体检

- 只读时间戳和 INSPVAX，不解点云，5–10 分钟读完 30 GB。
- 场景边界：雷达相邻两帧 > 0.3 s、IMU > 0.1 s、INS > 0.2 s，或 `pos_type ≠ 56` 的地方一定断开，这些时段丢掉；剩下的连续时段按 `--scene-length` 等分。
- 时钟：每个话题 log 时间减 header 时间。正常是雷达约 +0.10 s、IMU 和 INS 约 0，5–95% 区间宽 < 50 ms。几秒的偏差、区间很宽、前后两段不一样，说明有传感器没对上 GPS 时间，**要停下来**：后面的检查都发现不了。
- 不建议用 `--pos-types 56 55` 放宽到浮点解，位姿误差会直接进标签。

### 4.2 转换

INSPVAX 的每一行：
1. `ins_status ≠ 3` 丢掉；
2. 整份 bag 第一个好解作为 ENU 原点（`EnuOrigin`），所有场景共用，世界坐标对得上；
3. 经纬高 → ENU 位置；`yaw = 偏置 − azimuth` → 只有 z 分量的四元数；东北天速度原样；
4. **航向偏置**（`--azimuth-offset auto`，默认）：先按 180° 读完，再看狗是不是沿 B 系 +y 前进（`forward_speed`：速度 > 0.5 m/s 的行沿机身 +y 的速度中位数）。是正的就是 180，负的就把航向转 180° 改成 0。几乎不动时用 0.15 m/s 再试，还不够就保持 180 并告警。写入 `meta.ins_azimuth_offset_deg`；
5. **杠杆臂**（`--ins-lever-arm`，默认跟着偏置走：180 用拟合值，0 用 IMU 中心）：`apply_lever_arm` 把位置减去 R(航向)·L，速度减去这一项的时间导数（中心差分），挪到机身原点。写入 `meta.ins_lever_arm`（表示"已经扣过"，后面不会再扣一次）；
6. 场景前后各多留 1 s（`--pad`），去畸变和位姿缓存要用；
7. 每行记 `pos_type`，计数写入 `meta.inspvax_pos_type`。

IMU：角速度 × π/180，轴不转（`meta.imu_axes = "body"`）。同样前后多留 1 s。

离地高度：每 10 帧用该帧 B 系点云估一次（2–15 m 环内取 z 第 5 百分位附近点的中位数），取中位数。只看第一帧会受脚下地形影响：0921 29 个场景第一帧估出 0.59–1.05 m，真实约 0.72 m。

### 4.3 序列里和定位有关的文件、字段

| 文件 | 形状 | 内容 |
|---|---|---|
| `odom.npy` | (K, 11) float64 | `t, x, y, z, qx, qy, qz, qw, vx, vy, vz`：W 系里机身原点的位置、姿态、速度，100 Hz（INS 的时间点）。转换后是 INS（只有航向）；融合后东、北、航向、横滚俯仰和东北速度换成融合值，`t` 和 `z` 不变 |
| `odom_ins.npy` | 同上 | 第一次融合时存下的 INS 原样（扣没扣杠杆臂看 `meta.ins_lever_arm`），之后每次融合都从它读 |
| `imu.npy` | (M, 4) float64 | `t, wx, wy, wz`：B 系角速度，rad/s |

| `meta.json` 字段 | 写入者 | 含义 |
|---|---|---|
| `lat0` `lon0` `h0` | 转换 | ENU 原点 |
| `inspvax_pos_type` | 转换 | 场景内各 `pos_type` 的计数 |
| `ins_azimuth_offset_deg` | 转换 | 航向偏置 |
| `ins_lever_arm` | 转换 | 已经扣掉的杠杆臂。没有这个字段的老 INSPVAX 序列没扣过，融合时补上 |
| `imu_axes` | 转换 / `sequence_meta.py --fix-imu-axes` | `body`：陀螺按 B 系存。没有这个字段的老 kygi 序列 x、y 是反的 |
| `lidar_time_offset` | `sequence_meta.py`（可选） | 读的时候加到所有雷达时间上的秒数 |
| `base_height`、`base_height_sweeps` | 转换 / `sequence_meta.py` | 离地高度和估计统计 |
| `det_frame` | 转换 | kygi 是 `{"auto_ground": true, "auto_level": true}` |
| `pose_source` | 融合 | 没有：INS 位姿；`kiss-icp+ins`：融合位姿 |
| `pose_refine` | 融合 | 第 5.12 节 |

离线读序列时，`SequenceReader.events()` 把 IMU、定位、点云按时间排好依次送进流水线，和在线收消息的顺序一样；`odom_latency` 可以把定位推迟几毫秒到达，模拟 LIO 晚到、走 IMU 重新传播那条路。

### 4.4 预览的检查（`sequence_preview.py`）

每个场景每 5 帧做一次和线上一样的预处理，再检查：

| 检查 | 含义 | OK | WARN | FAIL |
|---|---|---|---|---|
| `shift_p90_m` | 相隔约 1 s 的两帧，取离地 0.3–3 m、1.5–30 m 内的点（墙、杆、树、车），在 ±0.8 m 内二维搜索把一帧叠到另一帧上的最佳平移（先 0.1 m 网格，再 0.02 m；代价是 2000 个点到对面的平均距离，封顶 0.4 m）。**这就是激光看到的两帧位姿误差**，沿行进方向也能测出来 | ≤ 0.20 m | ≤ 0.40 m | 更大 |
| `align_m` | 相隔约 0.5 s 的两帧，竖直结构（0.4 m 格子里高差 > 0.6 m）上的点的水平错位中位数 | ≤ 0.10 m | ≤ 0.20 m | 更大 |
| `ins_step_m` | INS 位置跳变：每个定位样本的位移减去速度 × Δt，取最大 | ≤ 0.15 m | ≤ 0.5 m | 更大 |
| `rtk` | 场景内（含前后 1 s）`pos_type` 全是 56 | 是 | 否 | |
| `odom_gap_s` | 定位最大间隔 | 0.1 s | 0.5 s | 更大 |
| `ground_z_m` / `tilt_deg` | 检测系里 3–12 m 地面的高度和倾角 | 0.10 m / 2° | 0.20 m / 4° | 更大 |

参考值：运动样例 bag 用 INS 位姿时 `shift` 0.51 m（FAIL），融合后 0.12 m（OK）；故意把雷达外参的 180° 改错，`align` 变成 1.03 m（FAIL）。

`shift` 和 `align` 的分工：`align` 对外参、时间差敏感（墙变双线），对沿路方向的位姿误差不敏感；`shift` 两个都能量，是判位姿的主要指标。

### 4.5 雷达和 INS 的时间差（`time_offset_check.py`）

- 原理：时间差 δ 让每帧用到 δ 秒前（后）的位姿。匀速转弯、直行时每帧偏得一样，帧间照样对得齐；角速度在变时（开始转、停止转、原地掉头），相隔 0.5 s 的两帧偏差不同，错位约为距离 × 角速度变化 × δ（10 m、90°/s、50 ms 时 0.8 m）。
- 做法：取角速度变化 ≥ 20°/s 的帧对，雷达时间平移 −0.15…+0.15 s，每个偏移重算错位，取最小的；角速度几乎不变的帧对作对照，应该平稳。
- 判断：几个场景都在同一个非 0 偏移处有明显最小值（|δ| ≥ 0.02 s、错位比 0 处小 1.5 倍以上），才是时钟偏差。写进序列 `sequence_meta.py --lidar-time-offset <δ>`，在线节点 `-p lidar_time_offset:=<δ>`。曲线平或最小值在 0，就不是时钟问题。
- 验证：仿真注入 50 ms 延迟，工具找到 −0.050 s，转弯帧对错位 0.77 → 0.007 m。

### 4.6 下游怎么用融合结果

- **教师推理**（`run_sequence.py`）：用融合后的 `odom.npy` 做去畸变、累积、检测系，每帧记 `pose_bad`（帧时刻落在 `pose_refine.bad_intervals` 里，`io.pose_bad`）；
- **自动标注**（`auto_label.py`）：`pose_bad` 的帧给跟踪器空检测（它们在世界系里位置不对，会把静止物体拉偏），也不给它们出框；
- **审核和导出**：SUSTechPOINTS 导出和 `export_dataset.py` 都跳过 `pose_bad` 的帧；
- **训练**（`det3d/datasets/pipelines/dog_aug.py` 的 `SweepPoseJitter`）：训练数据是用融合位姿叠的，在线却是 INS 位姿。按概率 0.3，把旧帧（dt > 0）按一个速度误差 v（σ 0.5 m/s，上限 1.5）和角速度误差 w（σ 3°/s，上限 10°）挪动：第 k 帧转 w·dt_k、平移 v·dt_k。当前帧和框不动，标签照样对，输入里却有坏位姿造成的重影。

---

## 5. 位姿融合的详细设计（`tools/dog/refine_poses.py`）

### 5.1 输入、输出和不变量

- 输入：一个序列的 `scans/`、`imu.npy`、`odom_ins.npy`（第一次运行时是 `odom.npy`），`meta.json`；
- 输出：`odom.npy`（格式不变，下游不用改）、`odom_ins.npy`（只在第一次写）、`meta.json` 的 `pose_source` 和 `pose_refine`（先写 `.tmp` 再 `os.replace`，不会写一半）；
- 不变量：
  - 永远从 INS 原样算起，重跑结果一样；
  - `odom.npy` 仍是 100 Hz 的 INS 时间点，扫描之间线性插值，所以去畸变和 IMU 传播的用法和以前完全一样；
  - 高度（`z`）不融合，保留 INS 的；检测系的高度由地面找平管。

每个场景的处理顺序：

```text
odom_ins ─► 0 杠杆臂 ─► odom_body ──────────────────────────────┐
scans + imu ─► 1 KISS-ICP（陀螺先验）─► 激光位姿 lo_poses        │
                                         │                      ▼
                       2a INS 健康（只看 INS 自己）◄─────────── INS 位置、航向（在扫描时刻插值）
                       3  航向偏移 heading_offset ─► 融合航向 yaw
                       4  激光步转到 ENU（用 yaw − yaw_lo）─► d_k
                       2b 激光健康 lo_health（要用健康的 INS 速度判"停住"）
                       5  坏激光段替换；东、北各解一次带 Huber 的三对角最小二乘
                       6  闭合自检：失败的缺口降权重解；列出不可靠 / 未验证时段
                       7  激光倾斜 lo_tilt
                       8  写回 odom.npy；诊断图、_diag.npz
```

### 5.2 杠杆臂

- 默认用 `meta.ins_lever_arm`（转换时扣过的）；没有这个字段的老 INSPVAX 序列用 0921 的拟合值。`--lever-arm` 指定时，只补"指定值 − 已扣值"，不会扣两次。
- `apply_lever_arm(odom, L)`：`p ← p − R·L`，`v ← v − d(R·L)/dt`。航向只有 z 分量，L 的高度不影响结果。
- **剩余杠杆臂**（`lever_check`，核对用）：在两边都健康的扫描上，最小二乘拟合

  ```text
  p_ins − lo_cum = c + v·t + R(yaw_ins)·L
  ```

  c 是常量偏移，v·t 是激光的慢漂移，L 是还没扣掉的杠杆臂。6 个未知数（c、v、L 各两维），迭代 5 次的稳健加权（权重 √min(1, 0.1 m / |残差|)）。转弯不到 45° 时 L 不可观，返回空。扣对了的话应该接近 0；0921 第五次融合时是 (0.03, 0.01) m。
- 这个拟合就是当初找到杠杆臂的方法：扣之前每个场景拟合出的中位数是 (−0.03, −0.58) m。

### 5.3 激光里程计（KISS-ICP）

- 每帧：去重（1 mm 网格）、去自身点（机身遮挡框）、加 `lidar_time_offset`，变到 B 系，带每个点的时间送进 KISS-ICP（体素 0.4 m，距离 1.5–60 m，开去畸变，去畸变到扫描结束 = 我们的帧时刻）。和在线预处理的去重、去自身点一样。
- **陀螺先验**（`--prior imu`，默认）：KISS-ICP 默认用上一帧的运动预测下一帧（匀速），转弯开始和停止时会预测错，配准落到错的局部最小：0921 上偏航角速度 < 5°/s 时 4% 的步、> 60°/s 时 53% 的步来回跳 0.5–1 m（s011 激光路程 105.6 m，INS 只有 36.8 m）。现在每帧配准前把 `last_delta` 的旋转换成两帧之间陀螺积分的旋转（`gyro_rotation`：逐个 IMU 样本 `R ← R·Exp(ω·Δt)`），平移仍按 B 系匀速。样例 bag 上激光每帧偏航和陀螺积分的相关从 0.89 升到 0.99；0921 激光路程和 INS 路程之比从最多 2.35 降到全部 1.00–1.16。
- 每 10 帧存一份配准用的场景点（离地 0.3–3 m、1.5–30 m，每 5 cm 一个点），给 5.10 的位姿检查用。
- 多场景并行时，每个进程的 KISS-ICP 线程数 = 核数 ÷ 场景数。

### 5.4 健康判定

**INS**（`ins_health`）：只看 INS 自己，不用激光判它（激光停住时会冤枉 INS）。
- 每个扫描时刻取 0.5 s 窗口，比较 INS 位置变化和它自己速度积分的变化：`rate = |Δp − Δ∫v| / 窗长`。急停时两者差约 1 m/s，正常走只差 0.1 m/s；
- `rate > 0.4 m/s` 判坏，前后各扩 0.3 s；
- 夹在两段坏之间、短于 1 s 的"好"小段也算坏（急停时 INS 可以短暂地"自洽地错"）；
- 整段都不健康时全部当健康，免得问题没有锚点。

**激光**（`lo_health`）：按步判，坏步前后各扩 0.2 s。
- **跳步**：单步速度 > 3.5 m/s（狗最快约 2.5 m/s）；
- **锯齿**：0.6 s 窗口内路径长 > 2 × 位移（位移不足 0.05 m 按 0.05 算），且路径 > 0.3 m。KISS-ICP 在转弯起止时来回跳就是这样；
- **停住**：INS 在这一步健康、显示速度 > 0.8 m/s，激光的路程却不到 INS 的 30%。沿着没有特征的长墙配准会这样。急停时 INS 速度冲过头，这时 INS 不健康，不拿来判激光。

### 5.5 航向融合（第五版起，`heading_offset`）

融合航向 = 激光航向 + 慢变偏移 s_k。s 用最小二乘求：

```text
min  Σ w_k (s_k − (yaw_ins,k − yaw_lo,k))² / σ_ins²  +  Σ (s_k+1 − s_k)² / σ_drift²
```

- σ_ins = 2°，只用 INS 健康的扫描（w = 1，否则 0）；
- 残差超过 1° 的按 Huber 降权（迭代 6 次）；
- σ_drift = 0.02° 每帧；时间常数 σ_ins / σ_drift = 100 帧 = 10 s；
- 用三对角求解（和位置同一个 `_solve`，步长 d 取 0）。

为什么这样：激光航向帧到帧是对的（它直接配准点云，又有陀螺先验），但会慢慢漂；INS 航向长期不漂，急转后却会偏 4–10° 好几秒，偏的时候位置自检照样健康，所以健康判定拦不住，只能让它"拉得很慢"。10 s 的时间常数远长于 3–5 s 的偏差，偏差被平均掉；又远短于激光航向漂移的尺度。

合成数据：INS 航向在转弯后偏 9°、3.5 s 回来，融合航向的起伏 < 0.3°。

### 5.6 位置融合

激光每步的位移先转到 ENU：激光自己的坐标系（第一帧的系）和 ENU 差一个航向角，用"融合航向 − 激光航向"逐步转，`d_k = Rz(yaw_k − yaw_lo,k)·Δlo_k`。

东、北两个方向分开，各解一次：

```text
min  Σ v_k (x_k+1 − x_k − d_k)² / σ_lo²  +  Σ w_k (x_k − p_ins,k)² / σ_ins²
```

- x_k：扫描时刻的位置；d_k：（可能被替换过的）激光步；p_ins：扣过杠杆臂、插值到扫描时刻的 INS 位置；
- 未知数是一串、只和相邻的耦合，正规方程是对称三对角矩阵，`scipy.linalg.solveh_banded` 直接解，300 帧的场景一次约 2 ms；
- **权重**：
  - σ_lo = 0.02 m（一步激光），σ_ins = 1 m（一个健康的 INS 点）：INS 只消慢漂移。平滑长度约 σ_ins / σ_lo = 50 帧 = 5 s；
  - 为什么 INS 这么"软"：0921 上 INS 即使自检健康，1 s 尺度上也有 0.2–0.5 m 的误差，激光没有。第二版时融合被 INS 拉坏（s008：INS 0.48、激光 0.12、融合 0.32 m）就是因为 INS 拉得太硬；
  - INS 不健康的地方 w = 0，激光把前后两段健康的 INS 连起来；
  - INS 残差超过 0.3 m 的按 Huber 降权（迭代 6 次）。
- **坏激光段**（连续的坏步）：
  - 这段 INS 整段健康：用 INS 速度积分当这几步（`vel_step`）；
  - 否则：用激光在这段的净位移均匀摊开（KISS-ICP 来回跳时净位移是对的；这时 INS 速度正是冲过头的那个）。记为 `bridged`；
  - 两种情况的步权重都降到 σ_lo_bad = 0.5 m。

### 5.7 闭合自检

- INS 不健康的缺口，两端都有健康 INS 时，看激光走完这段能不能和对端的 INS 闭合：`|Δlo_cum − Δp_ins|`；
- 差距超过 `gap_tol + gap_rel × 段长`（0.5 m + 5%）：这段的激光步降到 σ_lo_bad 重解，让差距在段内消化，不扩散到前后健康的轨迹；激光步本身保留，不换成 INS（0921 上这种缺口里每次都是激光对、INS 错）；这段记为**不可靠**；
- 缺口挨着序列开头或结尾，查不了，记为**未验证**（`unverified_intervals`），不算不可靠；
- 两种来源同时不健康 ≥ 0.2 s 的地方也记为不可靠。

### 5.8 倾斜（第六版起，`lo_tilt`）

- 激光位姿里机身 z 轴在激光系的方向，取全段中位数作为"平常的上方向"（狗大部分时间站平）；
- 把激光姿态转到这个参考下，读出每帧的横滚、俯仰；
- 默认（`--tilt lo`）写进 `odom.npy`：姿态 = 融合航向 + 激光的横滚、俯仰（ZYX 欧拉角，扫描之间线性插值，两端以外取端值）；
- 检测系的地面找平还开着，只修剩下的部分：雷达 / 机身相对地面的固定倾斜（164127_s002 平路上关掉找平，近处地面倾斜中位数 0.25° → 2.8°），以及激光看不出的小而快的倾斜（样例 bag 急刹时 1.5°，地面和陀螺都看到了，激光的俯仰一直在 +0.2° 左右）；
- 核实（s025）：激光倾斜和陀螺积分在 0.5 s 内的变化，鼻子方向相关 1.00、斜率 1.00、均方差 0.33°，侧倾方向相关 0.96。

### 5.9 写回（`refined_odom`）

- 扫描时段内：东、北、航向按扫描时刻的融合值线性插值到每个 INS 时间点；速度是融合轨迹的导数（`np.gradient`），同样插值；
- 扫描时段外（前后的 1 s 余量）：INS 原值加上第一帧 / 最后一帧的修正量；
- 高度、时间点不变。

### 5.10 诊断

**位姿检查**（`pose_checks`，和预览的 `shift` 同一个算法，`pose_shift`）：存下来的场景点（每 10 帧一份），相隔 1 s 的帧对，分别用几种位姿放到世界系里量平移：

| 名字 | 位姿 | 用来回答 |
|---|---|---|
| `ins` | INS（扣过杠杆臂） | INS 本身多差 |
| `lo` | 纯激光 | 激光本身在这个场景行不行（> 0.2 m 说明转角、开阔地、动态物体多） |
| `fused` | 写出去的那个 | 结果 |
| `lo_flat` | 激光去掉横滚俯仰 | 第六版诊断：差别是不是来自倾斜 |
| `fused_flat` / `fused_tilt` | 融合，只有航向 / 加上激光倾斜 | 同上 |

读法：纯激光好、融合反而差，说明融合有问题；纯激光也差，问题在激光（这时应该由 INS 接管或标不可靠）。

**输出**（`--plots <目录>`）：
- 每个场景一张三栏图：轨迹（INS 灰、融合蓝）；修正量、INS 不可信时段、被拒的激光步；各位姿的 `shift`；
- `<场景>_diag.npz`（几十到几百 kB）：扫描时刻、融合位置、激光位置 / 航向 / 完整姿态（`lo_R`）、INS 位置 / 航向、融合航向、健康判定、`rate_v`、坏步、`bridged`、倾斜、扣过杠杆臂的 INS（float64，unix 时间要 float64：第三版时存成 float32，时间精度只剩 128 s）、各位姿检查的结果，以及融合比激光差最多的 6 个帧对的场景点（float16）。**不用点云就能离线重放融合、复现最差的帧对**，第四、五版都是这样先在本机验证再上服务器的；
- `refine_summary.csv`：每个场景一行。

**屏幕输出**：每个场景一行（帧数、修正量中位 / p90 / 最大、INS 坏占比、激光坏占比、不可靠占比、激光 / INS 路程比、三种位姿的 `shift` p90），每个不可靠时段一行原因，最后一行 "INS lever arm left over"（转弯 > 45° 的场景的剩余杠杆臂中位数，应该约等于 0，否则加到 `--lever-arm` 上）。

### 5.11 参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--seq` | | 序列目录（可以多个） |
| `--jobs` | 0 | 同时处理几个场景，0 = 每核一个，1 = 不开子进程（调试） |
| `--plots` | 无 | 诊断输出目录 |
| `--sigma-lo` | 0.02 m | 一步激光的信任度 |
| `--sigma-ins` | 1.0 m | 一个健康 INS 点的信任度（只消漂移） |
| `--sigma-lo-bad` | 0.5 m | 坏激光步、闭合失败缺口里的步 |
| `--huber` | 0.3 m | INS 残差超过它就降权 |
| `--gap-tol` / `--gap-rel` | 0.5 m / 0.05 | 闭合允许的差距 |
| `--prior` | imu | KISS-ICP 的运动预测：陀螺旋转（imu）或它自己的匀速（none） |
| `--tilt` | lo | 写不写激光的横滚俯仰（none 只写航向） |
| `--lever-arm` | meta 里的，或 0921 拟合值 | INS 位置点在 B 系的位置 |

代码里固定的（改要改代码）：INS 健康窗口 0.5 s、阈值 0.4 m/s、扩 0.3 s、小段 1 s；激光健康 3.5 m/s、锯齿 2 倍 / 0.6 s、停住 30% / 0.8 m/s、扩 0.2 s；航向 σ 2°、Huber 1°、漂移 0.02°/帧；KISS-ICP 体素 0.4 m、1.5–60 m。

### 5.12 `meta.pose_refine` 的字段

| 字段 | 含义 |
|---|---|
| `scans` | 帧数 |
| `correction_median_m` / `_p90_m` / `_max_m` | 融合位置和 INS 的差 |
| `heading_correction_p90_deg` / `_max_deg` | 融合航向和 INS 航向的差 |
| `ins_unhealthy_share`、`lo_unhealthy_share` | 两边不健康的占比 |
| `lo_steps_rejected`、`lo_steps_bridged` | 坏激光步数，其中按净位移摊开的步数 |
| `lo_path_ratio`、`lo_path_m`、`ins_path_m` | 激光 / INS 路程（陀螺先验以后应在 1.0–1.2） |
| `lo_vs_ins_1s_median_m` / `_p90_m` | INS 健康时，激光 1 s 位移和 INS 的差 |
| `bad_intervals` | 不可靠时段 `[t0, t1, 闭合差 或 null]`（null = 两边都不健康）。**这些帧不要标注** |
| `unverified_intervals` | 挨着两端、查不了闭合的 INS 缺口 |
| `bad_share` | 不可靠帧的占比 |
| `lever_arm`、`lever_residual_m` | 用的杠杆臂；拟合出的剩余杠杆臂（转弯不够时 null） |
| `lo_tilt_ref_deg`、`lo_tilt_p90_deg`、`lo_tilt_max_deg`、`lo_tilt_change_1s_p90_deg` | 中位"上方向"和激光 z 轴的夹角；倾斜的 p90、最大；1 s 内倾斜变化的 p90 |
| `shift_<位姿>_p90_m` / `_median_m` | 5.10 的各位姿检查 |
| `prior`、`tilt`、`sigma_*`、`huber`、`gap_*` | 这次用的参数 |

### 5.13 耗时

- 每个场景的主要时间在 KISS-ICP 和位姿检查（KD 树查询）；求解本身可以忽略；
- 4 核机器上样例 bag 复制出的 4 个场景（各 27 帧）：每个场景 11.5 s → 4.1 s（串行）→ 2.7 s（4 个并行）；
- 服务器上 35 个 300 帧的场景：原来约 1 小时（每个约 100 s，60 s 在配准检查），现在按核数并行，16 核以上 5–10 分钟。结果和串行完全一样（每个场景固定随机种子）。

---

## 6. 迭代过程：六版

0921 的两份录制（164127 6 个场景、165659 29 个场景），每版在服务器上跑一轮（BAG_SPLIT 5.4–5.10），结果回来以后用 `_diag.npz` 离线分析、改代码、在合成数据上加测试，再上服务器。

| 版本（轮次） | 改了什么 | 为什么 | 0921 结果 |
|---|---|---|---|
| 第一版（5.4） | KISS-ICP + INS 健康判定 + 最小二乘；INS 不可信时激光接管 | 急停超前 0.45 m | 多数场景 `shift` 下降，但转角场景仍 FAIL（s020 0.59、s026 0.61、s024 0.52、164127_s003 0.49 m），s011、s027 反而变差。大 bag 里 INS 不可信的时间比样例多得多（s020 约 60%），激光要单独接管 5–10 s |
| 第二版（5.5） | 激光步"失效"改成和 INS 速度积分、前后几步激光自己**都**对不上才算；替换用 INS 速度积分 | 原来和 INS 位置步长比，INS 往回跳时正常的激光步被拒，换成了带跳变的步 | 三种位姿各自打分：纯激光明显最好（s008 INS / 激光 / 融合 0.48 / 0.12 / 0.32 m，s028 0.59 / 0.11 / 0.37 m）。融合被健康的 INS 拉坏；激光在转弯时来回跳（> 60°/s 时 53% 的步） |
| 第三版（5.6） | 陀螺先验；两边各自判健康；INS 只消漂移（σ_ins 1 m）；闭合失败的缺口激光降权重解 | 激光转弯时跳；INS 拉得太硬 | 激光路程比全部 1.00–1.16；纯激光 `shift` p90 中位 0.08 m。融合中位 0.14 m，14 个 > 0.2 m、6 个 > 0.4 m；预览 165659 13 / 8 / 8。分析出：闭合失败的 8 个缺口里激光是对的却被换成了 INS；根源是杠杆臂（拟合出 (−0.06, −0.54) m） |
| 第四版（5.7） | 杠杆臂（转换、融合、在线三处一致）；坏激光段在 INS 不全健康时用净位移摊开；闭合失败只降权不替换 | 杠杆臂；冲过头的正是 INS 速度 | 剩余杠杆臂 (0.03, −0.04) m；闭合失败 8 → 0；> 0.4 m 6 → 0、> 0.2 m 14 → 10；预览 14 / 14 / 1。剩下的差距来自航向：急转后 INS 航向偏 4–10° |
| 第五版（5.8） | 航向 = 激光 + 慢变偏移；IMU 轴修正（陀螺 x、y 不再取反） | INS 航向偏；陀螺先验的横滚俯仰 | 融合中位 0.136 → 0.092 m，> 0.2 m 10 → 4；纯激光也变好（0.084 → 0.064 m）；164127_s003 0.289 → 0.076 m（航向修正最大 9.9°）；预览 15 / 13 / 1；不可靠 1.7%。剩 5 个场景融合比激光差 0.1 m 以上，两者只差横滚俯仰 |
| 第六版（5.9 诊断、5.10 定稿） | 激光倾斜写进 `odom.npy`（默认） | 诊断证实差别就是倾斜：`fused_tilt` 和 `lo` 只差中位 0.003 m、最多 0.037 m；狗最大倾斜 15–24° | 融合中位 0.072 m、最大 0.155 m，没有 > 0.2 m 的；和纯激光的差都在 0.05 m 内；预览 165659 22 / 6 / 1（剩下的 WARN 是坡道的 `tilt_deg` 和少量非固定解），164127 全 OK。**位姿到此定稿** |

几条经验：
- **先找对指标**：`align_m` 看不出沿路方向的误差，加了 `shift` 才看见急停的问题；
- **每个来源单独打分**：第二版把 INS、激光、融合分开量，才知道"融合比激光差"，否则只会继续调融合的权重；
- **诊断文件离线重放**：几十 kB 的 `_diag.npz` 就能重跑融合，一轮服务器结果可以在本机验证好几种改法（第四版"闭合失败 8 → 0"、第五版"航向差 0.53° → 0.02°"都是先离线算出来的，服务器结果和它一致）；
- **把异常追到物理原因**：杠杆臂、航向、倾斜每一项都先用数据拟合出数值，再和标定、天线位置对照，最后追到"天线接反"。

---

## 7. 在线定位（车上节点）

在线不能用激光里程计融合（没有未来的数据，也不想在 Orin 上多跑一个配准），用的是 INS + 陀螺 + 地面找平。Python（`dog_perception/online.py`）和 C++（`lidar3d/online.hpp`）逐位一致。

### 7.1 消息处理（`OnlineInputs`）

- **INSPVAX**：
  - 等第一个 `ins_status = 3` 的解定 ENU 原点（日志 "ENU origin ..."），之前每 2 s 告警一次 "waiting for a good INS solution"；
  - `pose_from_inspvax`：ENU 位置、`yaw = ins_azimuth_offset_deg − azimuth`、位置扣杠杆臂（`ins_lever_arm`）；
  - 解不好时丢掉并告警；`pos_type ≠ 56` 每 5 s 告警 "RTK not fixed"，但照样用；
  - `HeadingCheck`：数运动中（> 0.5 m/s）速度沿机身 +y 和 −y 的样本，往后的 ≥ 300 个、且是往前的 4 倍以上，就报一次 error "the robot walks backwards in the body frame"，说明偏置和天线接法对不上；
- **IMU**：kygi 约定（`imu_convention: kygi`）下 × π/180；
- **LIO 里程计**（`localization: odom`，别的机器人）：`nav_msgs/Odometry`，twist 可以是 body、world 或没有；
- **雷达时间**：每个点和帧时刻加 `lidar_time_offset`；
- **时间戳倒退**（T29 复核加的）：某个话题比它上一条早 1 s 以上（bag 重播、时钟回跳），整个重来（原点、位姿缓存、IMU、累积、地面、跟踪全清），日志一行 warn；别的话题还属于旧时间线的迟到消息丢掉，不算第二次倒退。不重来的话，位姿缓存把新位姿都当乱序丢掉，超过 3 s 后再也没有位姿。

### 7.2 位姿缓存和 IMU 传播（`PoseBuffer`、`ImuPropagator`）

去畸变要 IMU 频率的位姿：步态让机身以几 Hz 俯仰、横滚，30°/s 的俯仰角速度乘 0.1 s 扫描就是 3°，20 m 处约 1 m；对 10 Hz 位姿插值会把这些抹平。

- **锚点**：每条定位（INSPVAX 100 Hz）是一个锚点 `(t, R, p, v)`。比当前锚点旧的直接丢掉；
- **传播**：锚点之后，旋转逐个 IMU 样本积分 `R ← R·Exp((ω − b)·Δt)`，平移 `p0 + v0·(t − t0)`；
- **重新传播**：定位常常比更新的 IMU 晚到。新锚点到达时，缓存里它之后的样本截掉，用最近 1 s 存着的 IMU 从新锚点重新传播（FAST-LIO 内部也是这样）；
- **缓存**：保留 3 s；按时间插值，旋转 SLERP、平移线性；超出范围最多外推 0.05 s，外推速度取最近 20 ms 的运动，并跳过短于 1 ms 的末段（问题 3：两个来源时间挨得很近时，那一小步是修正，不是运动）。再远就抛 `PoseOutOfRange`，这一帧不出结果（"no pose for scan yet"）。

INSPVAX 速度在线没有扣杠杆臂（位置扣了）：两者差"角速度 × 杠杆臂"，30°/s 时 0.3 m/s。锚点 100 Hz，一次外推最多 10 ms，影响约 3 mm，可以忽略；离线转换时 `apply_lever_arm` 两个都扣了。

### 7.3 去畸变（`deskew_points`）

每个点从它自己发射时刻的机身位姿，变到帧时刻（扫描结束）的机身位姿：

```text
x_B(t_ref) = T_WB(t_ref)⁻¹ · T_WB(t_i) · T_BL · x_L
```

XT32 每 50 µs 发一组 32 线，点按 0.1 ms 分组共用一个插值位姿，几乎精确，又把位姿插值从每点一次降到每组一次。仿真里 8° 俯仰、0.5 rad/s 转向，墙面点误差 59 cm → 0.8 cm（传感器噪声 1 cm）。

### 7.4 检测系和地面找平

- `det_frame_from_body`：D 的旋转只取机身航向 `Rz(yaw)`，原点 = 机身原点下移 `base_height − ground_offset`，再按估计的地面倾斜转一下；
- `GroundHeightEstimator`，每帧只用最新一帧的点（dt = 0）：
  - 取 D 系里 1.5–8 m 环内、|z| < 0.35 m 的点，少于 200 个就不更新；
  - `auto_ground`：残差中位数按 0.2 的增益累加进高度偏移，限 ±0.3 m；
  - `auto_level`：两遍修剪最小二乘拟合平面 z = a·x + b·y + c（第二遍去掉离平面 > 8 cm 的点），横滚 = atan(b)、俯仰 = −atan(a)，同样 0.2 增益累加，限 ±5°；
- kygi 预设：`base_height` 0.72 m，两个都开（`robot:=kygi`，车上配置同样）。原来的节点没开、离地高度 0.45 m，和训练数据不一致，BAG_SPLIT 3.3 的复核才改过来。

### 7.5 在线和离线的差别

| | 离线（训练数据、教师） | 在线（车上） |
|---|---|---|
| 位置 | 激光 + RTK 融合 | INS（扣杠杆臂） |
| 航向 | 激光 + 慢变偏移 | INS |
| 横滚、俯仰 | 激光（相对中位上方向）+ 地面找平 | 只有地面找平（最多 5°，每帧跟 20%） |
| 两次定位之间 | 陀螺积分 | 陀螺积分 |
| 位姿不可靠时 | 不标注、不进数据集 | 照常输出 |

后果和对策：
- 急停、急转时在线的多帧叠加有重影：训练时 `SweepPoseJitter` 让模型见过这种输入；天线接正后 INS 本身应该好很多，要用新录制确认（第 9 节）；
- 大倾斜（路沿、台阶、坡，> 5° 或变化快）时，在线点云的检测系没放平，和训练数据不一样：以后可以用陀螺 + 加速度计在线估倾斜（第 10 节）；
- 想看在线位姿到底差多少：把 bag 用 `tools/deploy/cpp_parity.py python --bag` 回放（在线的处理），和同一段融合后的序列比 `shift`。

### 7.6 节点参数（`dog_perception/node_config.py`，车上配置 `$L/config/lidar_3d_detection.yaml` 的 `node:` 一节）

| 参数 | kygi 预设 | 含义 |
|---|---|---|
| `localization` | inspvax | inspvax 或 odom（LIO） |
| `inspvax_topic` / `imu_topic` | `/rte/rdap/rtk/inspvax` / `/rte/rdap/rtk/imus/data_raw` | |
| `imu_convention` | kygi | kygi：deg/s、B 系轴；ros：rad/s |
| `ins_azimuth_offset_deg` | 180 | 天线接反 180，接正 0 |
| `ins_lever_arm` | Python 里是 nan（跟着偏置走）；导出到车上配置时写成具体值 (−0.06, −0.54, 0) | INS 位置点在 B 系的位置 |
| `lidar_time_offset` | 0 | 雷达时间加多少秒 |
| `base_height`、`auto_ground`、`auto_level`、`ground_*` | 0.72、true、true、见 7.4 | 检测系 |

注意：车上配置是 `export_cpp_config.py --robot kygi` 导出的，`ins_lever_arm` 已经按导出时的偏置（180）解析成具体值。**只改偏置不会跟着换杠杆臂**，两项要一起改（第 10 节）。

### 7.7 C++ 和 Python 的一致性

- `lidar3d/localization.hpp`（ENU、INSPVAX、`HeadingCheck`）、`pose_buffer.hpp`、`online.hpp` 和 Python 一一对应，位姿用 float64；
- golden 数据里带样例 bag 前 8 帧的在线回放，`Online.MessageLogMatchesPythonBagReplay` 每次 CTest 都测；样例 bag 27 帧上 C++ 读消息日志，点逐位相同、位姿差约 1e-15；
- 板上 ROS 读 bag 和读消息日志逐位相同（T29；T32 用 deb 里的两个回放再验一次）。

---

## 8. 检查和验收

### 8.1 一份录制处理完要看的数

| 指标 | 在哪 | 期望 |
|---|---|---|
| 时钟（log − header） | 体检 | 雷达约 +0.10 s，IMU、INS 约 0，区间宽 < 50 ms |
| `rtk` | 预览 | 场景内全是 56（边上少量非固定解可以保留） |
| 航向偏置 | 转换输出 | 和天线接法一致（接正应自动选 0） |
| 融合的 `shift` p90 | `refine_poses.py`、预览 | ≤ 0.2 m；0921 定稿时中位 0.072 m |
| 融合和纯激光的差 | `shift_fused` − `shift_lo` | ≤ 0.05 m（大了说明融合有问题） |
| 剩余杠杆臂 | "INS lever arm left over" | 约 0（0921：(0.03, 0.01) m）；不是 0 就加到 `--lever-arm` 上重跑 |
| 激光 / INS 路程比 | `lo_path_ratio` | 1.0–1.2 |
| 闭合失败 | "pose bad ... misses the INS" 行 | 0 个 |
| 不可靠占比 | `bad_share` | 0921 是 1.7% |
| 航向修正 | `heading_correction_max_deg` | 0921 最大 9.9°；天线接正后应小很多 |
| 时间差 | `time_offset_check.py` | 最小值在 0 附近 |

### 8.2 测试

| 测试 | 测什么 |
|---|---|
| `tests/dog/test_refine_poses.py` | 合成数据（有真值）：急停超前被修正到 < 6 cm（`test_fuse_bridges_braking_error`）；INS 往回跳时保留好的激光步；INS 健康时接管坏激光；陀螺积分；激光停住被报出来；INS 缺口里的激光锯齿（< 0.15 m）；急转后航向跟激光（起伏 < 0.3°）；倾斜相对中位上方向；"8"字行走时杠杆臂被测出并去掉（< 2 cm）；样例 bag 上 `shift` 融合前后 |
| `tests/dog/test_time_offset.py` | 注入的时间差被找出来；`meta.lidar_time_offset` 读的时候生效 |
| `tests/dog/test_bag_split.py` | 体检给出场景和验证集缓冲；一遍切分和整个 bag 转换一致；预览对好数据通过、外参改错时 FAIL |
| `tests/dog/test_ros2_node.py` | 样例 bag 回放进节点，检测系点云和训练序列一致 |
| `$L/tests/test_io_geometry.cpp`、`test_online.cpp` | C++ 对 Python：`Localization.InspvaxMatchesPython`（ENU、航向、杠杆臂）、`Online.InspvaxWaitsForAGoodFix`、`Online.MessageLogMatchesPythonBagReplay`、`Online.ReplayedBagStartsOver`（时间戳倒退） |

---

## 9. 天线接正以后的新录制

1016 起天线已经接正（方位角和行进方向只差约 9°，转换器自动选了偏置 0）。下一批 XT32 的录制到了以后：

1. **转换**：`--azimuth-offset auto` 应打印 0，杠杆臂默认用 IMU 中心 (0.0213, −0.18, 0)。打印的是 180 或者告警"walks backwards"，先停下查接线；
2. **融合**：照常跑 `refine_poses.py`，重点看三项：
   - "INS lever arm left over"：接近 0 说明 INSPVAX 报的就是 IMU 中心；否则把剩下的量加到 `--ins-lever-arm` / `--lever-arm` 上；
   - `shift_ins_p90_m`（INS 自己）：预期比 0921 小很多；
   - `heading_correction_max_deg`、`ins_unhealthy_share`：急刹和急转后的 INS 误差应该基本没有了；
3. **决定还要不要融合**：INS 自己的 `shift` 如果已经和融合接近（例如 p90 ≤ 0.1 m、没有急停超前），在线和离线的差别就小了，训练里的 `SweepPoseJitter` 也可以调小；融合照样跑（不亏，还能标出不可靠的帧）；
4. **时间差**：在几个转弯场景上跑一次 `time_offset_check.py`；
5. **车上节点**：配置改成接正的参数（第 10 节第 1 条）。

---

## 10. 已知问题和下一步

1. **车上配置还是 0921 的天线接法**：`$L/config/lidar_3d_detection.yaml` 里 `ins_azimuth_offset_deg: 180.0`、`ins_lever_arm: [-0.06, -0.54, 0.0]`。车上天线已经接正的话，航向会差 180°：旧帧往反方向挪（静止物体被叠出重影、被当成在动），杠杆臂也扣错方向；节点会报 "walks backwards" 的 error，但照常输出。**上车前两项一起改**成 `0.0` 和 `[0.0213, -0.18, 0.0]`（或者在 Python 预设里改默认、`check_deploy_deb.py --write` 重新导出），再用一段接正后的 bag 离线回放确认没有 error。改哪一处、什么时候改，由人定。
2. **在线没有倾斜**：大倾斜时在线点云没放平。可以加一个在线倾斜估计：陀螺积分管快变化，加速度计（静止或匀速时指向重力）慢慢拉回，和地面找平互补；先在 0921 上和激光倾斜（`lo_tilt`）比。
3. **在线的急停超前**：0921 的 INS 本身没法补救；天线接正后有没有，等新录制（第 9 节）。还有的话，在线可以加一个简单的健康判定（同 5.4 的 INS 判定，只用过去的数据），不健康时改用"上一个健康锚点 + 陀螺航向 + 慢下来的速度"。
4. **接反时的杠杆臂还差 0.1 m**：推算值 (0.021, −0.474) 和拟合值 (−0.06, −0.54) 差 0.1 m。T10（可选）用两个值各融合一次比残差；影响只在 0921 的数据。
5. **高度不融合**：用的是 INS 高度，检测系高度由地面管。坡道上 INS 高度和地面一起变，没问题；INS 高度跳变时会传给多帧叠加的 z，目前没观察到。
6. **激光也不行的地方**：开阔地、没有竖直结构的场景，KISS-ICP 会停住或跳；这时 INS 接管，两边都不行就标不可靠（0921 最长 3 s，在 s019 开头）。
7. **`kygi665_msgs` 的 `pos_type` 常量名写反了**（55 / 56）：代码按 NovAtel 标准用数字 56，没有用常量名。以后车上代码用常量名时要注意。
8. **1016 的数据不用**：点云录成了前置 Mid-360（线上用的是 XT32）。天线、相机的结论仍然有效。

---

## 11. 代码和文档索引

| 文件 | 作用 |
|---|---|
| `dog_perception/inspvax.py` | INSPVAX → ENU 位姿；外参和杠杆臂常数；`apply_lever_arm`、`forward_speed`、`default_lever_arm`、`HeadingCheck` |
| `dog_perception/pose_buffer.py` | `PoseBuffer`（插值、外推）、`ImuPropagator`（锚点 + 陀螺传播 + 重新传播） |
| `dog_perception/preprocess/deskew.py` | 逐点去畸变 |
| `dog_perception/preprocess/frames.py` | 检测系、地面高度和倾斜估计、离地高度估计 |
| `dog_perception/online.py` | 在线消息处理（INSPVAX、IMU、里程计、点云、时间戳倒退） |
| `dog_perception/node_config.py` | 节点参数和 kygi 预设 |
| `dog_perception/io.py` | 序列读写、`events()`、`pose_bad` |
| `tools/dog/bag_survey.py` | 体检、切场景 |
| `tools/dog/rosbag_to_sequence.py` | 转换（偏置、杠杆臂、离地高度） |
| `tools/dog/sequence_preview.py` | 预览检查（`shift`、`align`、`ins_step` ...），`pose_shift` |
| `tools/dog/time_offset_check.py` | 雷达和 INS 的时间差 |
| `tools/dog/sequence_meta.py` | 改 `meta.json`：离地高度、`lidar_time_offset`、`--fix-imu-axes` |
| `tools/dog/refine_poses.py` | 位姿融合（第 5 节） |
| `tools/dog/split_check.py` | 训练 / 验证的空间划分 |
| `det3d/datasets/pipelines/dog_aug.py` | `SweepPoseJitter` |
| `tests/dog/test_refine_poses.py`、`test_time_offset.py`、`test_bag_split.py`、`test_ros2_node.py` | 第 8.2 节 |
| `$L/include/lidar3d/localization.hpp`、`pose_buffer.hpp`、`online.hpp` | C++ 版 |
