# 从 ROS 数据到下游：调用链与数据 shape

本文按一帧数据的实际流向逐段说明：从 ROS 消息进来，到给规划和可视化的输出为止，列出每一段的调用链路和数据 shape。

表中的 shape 和 dtype 是**实测值**，测量条件如下：
- 数据：样例 bag `samples/rosbag2_2026_09_21-16_41_27` 的第 7 帧，此时已凑满 5 帧点云；
- 模型：`work_dirs/pretrained/nusc_pp.pth`（nuScenes CenterPoint-Pillar）；
- 硬件：CPU。

插桩脚本的做法是在每个函数外包一层，记录输入输出。方案背景见 [DOG_PIPELINE.md](DOG_PIPELINE.md)。

记号说明：`N` 表示点数，`V` 表示非空柱或体素数，`D` 表示检测数，`T` 表示轨迹数，`K` 表示预测模态数。

---

## 0. 总览

```
 ROS topic (频率)                     回调                                   状态 / 输出
 ─────────────────────────────────────────────────────────────────────────────────────────────
 sensor_msgs/Imu        100–400 Hz ─► pipe.on_imu ──► ImuPropagator ─────► PoseBuffer (IMU 频率 T_world_body)
 nav_msgs/Odometry 或                                                          ▲
 kygi665 INSPVAX        10–100 Hz ─► pipe.on_odometry ─► 重新设锚点、重传播 ──┘
                                                                               │ 插值
 sensor_msgs/PointCloud2  10 Hz ───► scan_from_msg ─► LidarScan               │
                                        │                                      │
                                        ▼  PerceptionPipeline.on_scan          │
   [1] Preprocessor.process   去重 → 自体滤除 → 逐点去畸变 ◄───────────────────┘
                              → 转世界系 → 入 5 帧缓存 → 检测系 → 拼 5 帧     (N,5) [x,y,z,i,dt]
   [2] CenterPointDetector    z 裁剪 → 虚拟 nuScenes 系 → 体素化 → 网络 → 解码
                              → 10→3 类 → 回检测系 → 几何过滤                  Detections (D,7)+(D,2)
   [3] MultiObjectTracker     转世界系 → IMM 预测 → 两阶段关联 → 更新 / 新建 / 删除  TrackState × T
   [4] IMMPredictor / LearnedPredictor   每条轨迹 3 s 外推                     Prediction × T
                                        │
                                        ▼
   下游：pipe.to_local() 给规划（检测系） │ ros2_node.publish()：MarkerArray（世界系）
         │ run_bag_mcap / McapViz：Foxglove MCAP
```

`PerceptionPipeline`（`dog_perception/pipeline.py`）是唯一的编排者，三个入口脚本都只负责把消息喂给它。

---

## 1. 坐标系

| 名称 | 定义 | 由谁给出 | 用在 |
|---|---|---|---|
| L 雷达系 | 驱动输出的点云坐标 | PointCloud2 | 原始点、自体滤除 |
| B 机体系 | LIO / INS 位姿所描述的刚体 | `T_body_lidar` 外参（4×4） | 去畸变的参考系 |
| W 世界系 | LIO 世界系，或 INSPVAX 的局部 ENU（以第一帧良好解为原点） | `PoseBuffer` 中的 `T_world_body` | 多帧缓存、跟踪、预测 |
| D 检测系 | 重力对齐，航向跟随机身，原点在机身正下方的地面 | `det_frame_from_body(T_world_body, base_height)` | 拼好的点云、检测输出、给规划 |
| M 网络系 | 虚拟 nuScenes 雷达：`x_M = Rz(90°)·x_D + [0,0,−1.84]` | `ModelFrame` | 只在网络输入输出处出现 |

---

## 2. 三个入口

| 入口 | 适用场景 | 位姿来源 | 调用链 |
|---|---|---|---|
| `tools/dog/ros2_node.py` | 在线运行 | `nav_msgs/Odometry`（LIO） | `on_imu:91` → `pipe.on_imu`；`on_odom:95` → `pipe.on_odometry`；`on_cloud:106` → `scan_from_msg` → `pipe.on_scan` → `publish:118` |
| `tools/dog/run_bag_mcap.py` | 离线回放带 INSPVAX 的 bag | `pose_from_inspvax`（ENU，`yaw = 180° − azimuth`，roll/pitch 置 0） | 按消息记录时间顺序读 `AnyReader`，其余与在线相同（`:96`、`:117`），最后 `McapViz.add`（`:124`） |
| `rosbag_to_sequence.py` → `run_sequence.py` | 批量评测、保存帧数据和跟踪日志 | 先把 bag 转成序列目录，再用 `SequenceReader.events()` 回放 | 与在线相同，另外保存 `results.pkl`、`track_log.npz` 和 `frames/*.npy` |

---

## 3. 位姿通路（高频，与点云异步）

```
Imu.angular_velocity          ─► ImuPropagator.on_imu(t, gyro(3,))             pose_buffer.py:156
                                   └─ _propagate: R ← R·Exp((ω−b)·dt),  p ← p0 + v0·(t−t0)   :166
                                      └─ PoseBuffer.add(t, R(3,3), p(3,))
Odometry / INSPVAX            ─► ImuPropagator.on_odometry(t, R(3,3), p(3,), v(3,)|None)   :132
                                   ├─ 比当前锚点旧的消息直接丢弃
                                   ├─ PoseBuffer.truncate_from(t)，写入新锚点
                                   └─ 用 1 s 内缓存的 IMU 从新锚点重新传播
```

| 数据 | shape / dtype | 说明 |
|---|---|---|
| INSPVAX 转换结果 `pose_from_inspvax` → `(t, R, p, v)` | `float`, `(3,3) f64`, `(3,) f64`, `(3,) f64` | 只有 `ins_status==3` 的消息才用，否则返回 None |
| `PoseBuffer` 内部数组 | `t (S,) f64`, `R (S,3,3) f64`, `p (S,3) f64` | 实测 S=74，最多保留 3 s（`max_age`） |
| `PoseBuffer.interpolate(times)` | 输入 `(Q,)`，输出 `R (Q,3,3)`、`p (Q,3)` | SLERP 插值，允许外推 0.05 s |

---

## 4. 点云通路：逐段调用与 shape

### 4.1 ROS 消息 → `LidarScan`（`ros_utils.py`）

```
scan_from_msg(msg, "timestamp", "absolute")                       ros_utils.py:30
  └─ cloud_to_array(msg)   按 fields / point_step 生成结构化 dtype      :17
```

| 步骤 | shape / dtype（实测） |
|---|---|
| `PointCloud2.data` | `(3 328 000,) uint8`，即 128 000 点 × `point_step` 26 字节 |
| `cloud_to_array` | `(128000,)` 结构化数组：`x,y,z,intensity <f4`、`ring <u2`、`timestamp <f8` |
| 去掉 NaN 和全零点后得到 `LidarScan` | `xyz (108220,3) f32`、`intensity (108220,) f32`、`point_times (108220,) f64`，`stamp = max(point_times)`（扫描结束时刻） |

### 4.2 预处理 `Preprocessor.process`（`preprocess/pipeline.py:115`）

```
Preprocessor.process(scan)
├─ sweep_to_body(scan)                                              :97
│   ├─ dedup_returns(scan)             双回波去重，1 mm 网格           :43
│   ├─ SelfFilter.__call__(xyz_L)      最小距离 + 机体盒 + 近场离群     self_filter.py:82
│   └─ deskew_points(xyz_L, t_i, PoseBuffer, t_ref=stamp, T_BL)      deskew.py:14
│        按 0.1 ms 分桶 → PoseBuffer.interpolate → T_B(t_ref)⁻¹ · T_WB(t_i) · T_BL
├─ T_WB = PoseBuffer.pose_at(stamp)                                  pose_buffer.py:107
├─ SweepAccumulator.push(stamp, T_WB·xyz_B, intensity)               accumulator.py:29
├─ T_WD = det_frame_from_body(T_WB, base_height, ground_offset)      frames.py:29
├─ points = SweepAccumulator.build(T_WD, stamp)                      accumulator.py:34
└─ GroundHeightEstimator.update(当前帧点)   (auto_ground 开启时)      frames.py:43
```

| 步骤 | 输入 | 输出（实测） | 说明 |
|---|---|---|---|
| `dedup_returns` | `xyz (108220,3) f32` | `(53253,3) f32` | 双回波模式下同一坐标会重复 |
| `SelfFilter` | `(53253,3)`（雷达系） | `keep (53253,) bool`，保留 39 731 点 | 去掉 0.4 m 内的无效回波和机体盒内的点 |
| `deskew_points` | `xyz (39731,3) f32`、`point_times (39731,) f64` | `xyz_body (39731,3) f64` | 统一到扫描结束时刻的机体系 |
| 转世界系后 `push` | `xyz_world (39731,3) f64`、`features (39731,1) f32` | 写入 deque，最多保留 5 帧 | 世界系用 float64，避免 ENU 大坐标丢精度 |
| `det_frame_from_body` | `T_world_body (4,4) f64` | `T_world_det (4,4) f64` | 只保留 yaw，原点下移 `base_height`（样例 bag 实测 0.31 m） |
| `build` | 5 帧 | **`points (198690,5) f32`**，列为 `[x, y, z, intensity, dt]`（检测系） | 最新帧在前；dt ∈ {0, 0.1, …, 0.4} |
| 输出 `PreprocessedFrame` | | `points (198690,5)`、`T_world_det (4,4)`、`T_world_body (4,4)`、`stamp`、`timings` | |

### 4.3 检测 `CenterPointDetector.__call__`（`detection/centerpoint.py:186`）

```
CenterPointDetector.__call__(points_det, stamp)
├─ _crop               检测系 z ∈ [−1, 3] m                               :149
├─ ModelFrame.points_to_model   D → M                                     frame_adapter.py:35
├─ build_example(points_M)                                                :164
│   └─ VoxelGenerator.generate(points)      det3d/core/input/voxel_generator.py
├─ net(example, return_loss=False)          PointPillars.forward          det3d/.../point_pillars.py:32
│   ├─ extract_feat                                                       :21
│   │   ├─ reader    PillarFeatureNet       点特征装饰到 10 维 → 2 层 PFN   readers/pillar_encoder.py:116
│   │   ├─ backbone  PointPillarsScatter    柱特征铺到 BEV                  :182
│   │   └─ neck      RPN                    3 级下采样 + 上采样拼接          necks/rpn.py:150
│   ├─ bbox_head.forward   shared_conv + 6 个 SepHead                      bbox_heads/center_head.py:236
│   └─ bbox_head.predict   sigmoid、解码、中心点 circle NMS                  :294 / :451
├─ postprocess(box9, scores, labels)                                      :208
│   ├─ label_map：nuScenes 10 类 → 3 类（barrier、cone 映射为 −1，丢弃）
│   ├─ det3d_to_standard   [x,y,z,w,l,h,vx,vy,r] → [x,y,z,l,w,h,yaw] + v   boxes.py:87
│   ├─ ModelFrame.boxes_from_model   M → D                                frame_adapter.py:48
│   └─ classwise_circle_nms   合并类别后的去重（例如 car 和 truck 头重复出框）  boxes.py:120
└─ filter_detections   点数 / 贴地 / 跨类去重                              filters.py:77
```

| 步骤 | shape / dtype（实测） | 说明 |
|---|---|---|
| `_crop` | `(198690,5)` → `(164191,5) f32` | |
| `points_to_model` | `(164191,5) f32` | 只改 xyz：绕 z 转 90°，z 减 1.84；intensity 和 dt 不变 |
| `VoxelGenerator.generate` | `voxels (4607,20,5) f32`、`coordinates (4607,3) int32`（z,y,x）、`num_points (4607,) int32` | 柱大小 0.2×0.2×8 m，网格 408×408×1，每柱最多 20 点 |
| det3d example 字典 | `voxels (V,20,5)`、`coordinates (V,4)`（在最前面补 batch 列）、`num_points (V,)`、`num_voxels (1,) i64`、`shape [array([408,408,1])]`、`points [(164191,5)]` | TTA 打开时 batch=4（原图 + 3 种翻转） |
| `reader` PillarFeatureNet | `(4607,20,5)` → 内部装饰为 `(4607,20,10)` → **`(4607,64)`** | 10 维 = 原始 5 维 + 相对柱内点均值 3 维 + 相对柱中心 2 维 |
| `backbone` Scatter | **`(1,64,408,408)`** | 按 `y·nx + x` 写入画布 |
| `neck` RPN | **`(1,384,102,102)`** | 3 个尺度各 128 通道拼接，输出步长 4，0.8 m/格 |
| `shared_conv` | `(1,64,102,102)` | |
| 6 个 SepHead，每个输出 | `reg (1,2,102,102)`、`height (1,1,…)`、`dim (1,3,…)`、`rot (1,2,…)`、`vel (1,2,…)`、`hm (1,C_t,…)` | 各 task 的 `C_t` 为 1,2,2,1,2,2（car / truck+cv / bus+trailer / barrier / moto+bike / ped+cone） |
| `predict` | 每个 task 有 102×102=10 404 个候选，经 score>0.1、范围过滤和 circle NMS 后每 task 最多 83 个 → 合并为 `box3d_lidar (128,9) f32`、`scores (128,)`、`label_preds (128,) i64` | box9 = `[x,y,z,w,l,h,vx,vy,r]`（M 系），`r = −yaw − π/2` |
| `postprocess` | `boxes (103,7) f64`、`velocities (103,2) f64` | 3 类，D 系，yaw 为逆时针、从 +x 起算 |
| `filter_detections` | `(103,7)` → **`(55,7)`** | 仍保留低分框，第二阶段关联要用 |
| 输出 `Detections` | `boxes (55,7) f64`、`velocities (55,2) f64`、`scores (55,) f64`、`labels (55,) i64`（0=vehicle，1=pedestrian，2=cyclist） | `last_raw_count = 103` |
| `Detections.above(DEFAULT_SCORE_THRESHOLDS)` | `(15,7)` | 用于上报 / 可视化，阈值为车 0.35、人 0.3、骑行 0.4 |

### 4.4 跟踪 `MultiObjectTracker.step`（`tracking/tracker.py:212`）

```
dets_w = Detections.transform(T_world_det, "world")      boxes.py:59   (55,7)，速度一并旋转
MultiObjectTracker.step(dets_w, stamp)
├─ Track.predict(stamp)  → IMM.predict(dt)：模态混合 + CV/CT 各自预测    tracker.py:115 / imm.py:149
├─ 按类别阈值拆分：high = score ≥ 阈值；low = 0.1 ≤ score < 阈值
├─ _associate(high, 所有轨迹)       代价矩阵 (N_high, M) 为中心距离，同类且在门限内   :200
├─ _associate(low, 未匹配的已确认轨迹)
├─ Track.update(box, vel, score) → IMM.update(z=[x,y,vx,vy], H(4,5), R(4,4))    :126 / imm.py:164
├─ 未匹配的 high 检测 → 新建 Track（tentative）；未匹配的轨迹 → mark_missed
├─ 删除：tentative 连续漏检超过 0.15 s，或已确认轨迹 coasting 超过 max_age
└─ outputs() = [t.to_state() for t in reported_tracks()]                 :255
```

| 数据 | shape / dtype | 说明 |
|---|---|---|
| 每个 `Track.imm` | `x (M,5)`、`P (M,5,5)`、`mu (M,)`，M=2 | 状态为 `[px, py, vx, vy, ω]`（世界系）；车和骑行者用 CV+CT，行人用两个噪声不同的 CV |
| 输出 `TrackState`（实测 16 条） | `position (3,)`、`velocity (2,)`、`yaw` float、`size (3,)`（l,w,h）、`score`、`covariance (4,4)`、`mode_probs (2,)`、`age`、`hits`、`coasting` bool、`history (K,6)` | `history` 每行为 `[t, x, y, vx, vy, observed]`，最多保留 3 s（实测 K=6） |

### 4.5 预测

第一档 `IMMPredictor.__call__`（`prediction/kinematic.py:71`）：

```
for t in tracker.reported_tracks():
    IMM.rollout(horizon=3.0, step=0.5)   → mode_means (M,6,5)、mode_covs (M,6,5,5)、mode_probs (M,)   imm.py:192
    merge_modes(...)                     轨迹重合的模态按矩匹配合并（行人两个 CV 会合成一条）   kinematic.py:37
    → Prediction
```

| 字段 | shape（实测） |
|---|---|
| `times` | `(6,)`，取值 0.5, 1.0, …, 3.0 s |
| `modes` | `(K,6,2)`，世界系 xy；K=1（合并后）或 2（车、骑行者的直行 / 转弯） |
| `probs` | `(K,)` |
| `covs` | `(K,6,2,2)` |

第二档 `LearnedPredictor.__call__`（`prediction/learned.py:37`）：历史不足 0.5 s 的轨迹回退到第一档。

| 张量 | shape | 说明 |
|---|---|---|
| `agent_hist` | `(B,20,6)` | 过去 2 s，10 Hz，列为 `[x,y,vx,vy,observed,valid]`，目标自身坐标系 |
| `agent_class` | `(B,3)` | one-hot |
| `nbr_hist` / `nbr_static` / `nbr_mask` | `(B,16,20,6)` / `(B,16,4)` / `(B,16)` | 最近的 16 个邻居，包含机器狗自身（`is_robot` 标志） |
| `raster`（可选） | `(B,1,64,64)` | 0.5 m 分辨率的障碍物栅格 |
| 网络输出 | `traj (B,6,6,2)`、`log_scale (B,6,6,2)`、`logits (B,6)` | 6 个模态 × 6 个时刻 |
| 转成 `Prediction` | `modes (6,6,2)`（世界系）、`probs (6,)`（softmax）、`covs (6,6,2,2)`（由 Laplace 尺度换算） | |

### 4.6 下游输出

| 输出 | 函数 | 坐标系 | 内容 / shape |
|---|---|---|---|
| **给规划** | `PerceptionPipeline.to_local(out)`（`pipeline.py:91`） | D（重力对齐，航向跟随机身，原点在地面） | 轨迹：`[{id, label, position(3,), velocity(2,), yaw, size(3,), score, coasting}] × T`；预测：`Prediction × T`，其中 `modes (K,6,2)` |
| ROS 2 可视化 | `DogPerceptionNode.publish`（`ros2_node.py:118`） | W（`world_frame` 参数） | `~/tracks`：每条轨迹一个 CUBE 加一个 TEXT；`~/predictions`：每个模态一条 6 点的 LINE_STRIP；`~/cloud`（可选）：`(N,5) f32`，20 字节/点 |
| Foxglove MCAP | `McapViz.add`（`mcap_viz.py:76`） | map = W | `/lidar`：当前帧，最多 20 000 点，16 字节/点；`/detections`：`above()` 之后的框；`/tracks`；`/pose` |
| 离线文件 | `run_sequence.py` | D / W | `results.pkl`（逐帧的 Detections / TrackState / Prediction）、`track_log.npz`（`rows (R,9)`、`ego (F,4)`、线上 IMM 预测）、`frames/*.npy`（`(N,5)`） |

**缺口**：ROS 2 节点目前只发可视化用的 MarkerArray，**还没有给规划的正式消息**。`to_local()` 已经给出了数据，但需要定义一个 ObjectArray 消息，包含 id、类别、位姿、尺寸、速度、协方差和多模态轨迹。

---

## 5. 延迟（样例 bag，x86 CPU，实测）

| 段 | ms | 说明 |
|---|---|---|
| 去畸变（含去重、自体滤除） | 30.0 | Python 参考实现，上板应改用 C++/CUDA |
| 多帧拼接 | 7.5 | |
| 体素化 | 13.1 | numba 实现 |
| 网络 | 488.9 | **CPU 上的数字**；GPU 或 BPU 上是另一个量级，需上板实测 |
| 后处理（解码映射 + 过滤） | 17.2 | |
| 跟踪 | 5.6 | 16 条轨迹 |
| IMM 预测 | 16.6 | 16 条轨迹 |
| **合计** | 578.9 | 其中网络占 85% |

---

## 6. 其他分支的 shape

**CenterPoint-Voxel**（按配置推算。这里没有 CUDA，前向没有实测）：

| 步骤 | shape |
|---|---|
| 体素化 | 体素 0.075×0.075×0.2 m，网格 1088×1088×40；`voxels (V,10,5)` |
| `VoxelFeatureExtractorV3`（体素内取均值） | `(V,5)` |
| `SpMiddleResNetFHD`（稀疏 3D 卷积，z 方向 41 → 2） | `(1,128×2,136,136) = (1,256,136,136)` |
| RPN | `(1,512,136,136)` |
| shared_conv | `(1,64,136,136)` |
| 各 head | 136×136，0.6 m/格；解码及之后的流程与 Pillar 相同 |

**BPU / ONNX 部署**（`pillar_export.py`，`tools/dog/export_pillar_onnx.py`）：

```
CPU  体素化 → decorate_pillars (V,20,10) → pad_pillars → (1,10,P_max=30000,20)
NPU  pfn.onnx        (1,10,30000,20) → (1,64,30000,1)
CPU  scatter_pillars → (1,64,408,408)
NPU  rpn_head.onnx   (1,64,408,408) → 36 张图（6 task × [reg, height, dim, rot, vel, hm]），每张 (1,C,102,102)
CPU  decode_centerpoint → box9 (D,9) → 与 4.3 节后半段相同
```

**训练数据**（`export_dataset.py` → `DogDataset`）：

```
frames/*.npy (N,5) D 系 ─► ModelFrame.points_to_model ─► lidar/*.npy (N,5) M 系
labels.pkl  box (7,) D 系 ─► boxes_to_model + standard_to_det3d ─► infos[i]["gt_boxes"] (M,9) det3d 格式
```
之后进入 det3d 的训练管道：`LoadPointCloudFromFile` → `Preprocess`（GT 采样等增强）→ `Voxelization` → `AssignLabel`（热力图 `(C_t,102,102)`）。
