# 轮足机器狗 XT32 感知四段流水线：方案评审与实现

本文档包含三部分：(1) 对原方案的评审和优化点；(2) 优化后的方案与本仓库中的实现对应关系；(3) 使用方法、仿真验证结果与已知局限。

代码位于 `dog_perception/`（四段流水线库）、`tools/dog/`（离线/在线工具）、`configs/dog/`（微调配置）、`det3d/datasets/dog/`（det3d 数据集）、`tests/dog/`（39 个测试，其中 Voxel 前向在无 CUDA 时跳过）。

---

## 1. 评审结论

原方案的主体方向是对的：四段式结构，部署选 Pillar 路线，用 CenterPoint 自带的速度头配合贪心匹配做跟踪，预测先上 Kalman、有数据后再换学习模型。下面按影响大小列出需要修正或补充的地方。

| # | 原方案 | 问题 / 优化 | 影响 |
|---|---|---|---|
| 1 | 去畸变"一步解决" | 位姿必须是 **IMU 频率**（≥200 Hz），不能对 10 Hz LIO 位姿做线性插值。步态的俯仰振动恰好会被插值抹平。另外需要 PTP 时间同步 | 高 |
| 2 | "重力对齐是否有必要？" | **有必要，而且几乎零成本**。LIO 给出的是重力对齐的世界系位姿，检测器需要的是"重力对齐、航向跟随机身、原点在地面"的检测系。这一步只是一个 4×4 变换，合并进多帧拼接的变换里即可 | 高 |
| 3 | z 按重力系 [-1,3] | 需要固定原点，并把检测系映射到**虚拟 nuScenes 雷达系**（离地 1.84 m，前向为 +y）。这样预训练模型学到的绝对 z / 高度先验可以直接沿用 | 高 |
| 4 | 未提及 | **近场垂直盲区**：XT32 视场 -16°~+15°，装在约 0.6 m 高处时，行人要在 4.1 m 以外才能完整进入视场。1 m 处只能看到 0.31–0.87 m 高度的一段。nuScenes 数据里没有这种样本 | 高 |
| 5 | 3~5 帧累积，"32 线变百线" | 固定为 **5 帧**（dt 范围 0–0.4 s，对应 nuScenes 的 0–0.45 s，点数约 32 万，nuScenes 约 34 万）。"变百线"的说法偏乐观：狗速度慢，相邻帧的光束几乎重合。多帧累积的实际收益在于与预训练分布对齐，以及让速度头可用 | 中 |
| 6 | 检测范围 ±40 m | 改为 **±40.8 m**。网格尺寸需要是网络最大步长的整数倍（Pillar 为 408 格，Voxel 为 1088 格）。另外 Voxel 模型的 z 网格**不能改**，稀疏主干需要 41 个 z 格 | 中（否则直接报错或权重失配） |
| 7 | 3 类，nuScenes 预训练后微调 | 不要随机初始化 3 类 head。做 **head 权重迁移**：vehicle 继承 car head，pedestrian 继承 pedestrian 通道，cyclist 继承 bicycle 通道。car/truck/bus 映射到同一类后需要再做一次类内 NMS | 中 |
| 8 | OpenPCDet 训练 | 本仓库就是 CenterPoint 原版（det3d），已加入 `DogDataset`、3 类配置和 GT 采样库，**不需要迁移框架** | 低 |
| 9 | 离线大模型教师 | 在逐帧检测之外，补一步**轨迹级非因果精修**：类别投票、尺寸取中位数、RTS 平滑补漏检、用运动方向消除朝向歧义。仿真中行人 AP@0.5 m 从 0.59 提升到 0.99 | 中 |
| 10 | 跟踪：中心距离匹配 | 必须在 **LIO 世界系**里跟踪。速度头的输出直接作为 KF 量测。关联采用**两阶段**（高分检测先匹配，低分检测只用于延续已有轨迹）。行人保活时间放宽到 1.5 s | 高 |
| 11 | 第一档 IMM (CV/CTRV) | 改用 CV + CT（笛卡尔速度的协调转弯模型），两者共享 5 维状态，没有角度回绕问题。跟踪器的滤波器与预测器共用这一个 IMM，每个模态单独外推，自然得到多模态输出 | 中 |
| 12 | 第二档：自监督 | 要把**机器狗自身**作为邻居输入。输入使用跟踪器的因果估计，真值使用 RTS 平滑后的轨迹。对比时必须用**线上 IMM 的真实输出**作为基线 | 中 |

### 1.1 预处理

**去畸变。** 旋转是畸变的主要来源：30°/s 的步态俯仰角速度乘以 100 ms 扫描时间等于 3°，在 20 m 处造成约 1 m 的位移；而 1 m/s 平移在 100 ms 内只有 0.1 m。所以旋转必须用陀螺仪积分。本实现 `ImuPropagator` 的做法：以 LIO 位姿为锚点，两次锚点之间做陀螺积分，平移按 LIO 速度外推。LIO 结果通常在 IMU 数据之后才到，届时会从新锚点重新传播。仿真条件为 8° 俯仰、0.5 rad/s 转向，墙面点误差从 59 cm 降到 0.8 cm（传感器噪声为 1 cm）；由 10 Hz 里程计加 400 Hz 陀螺积分得到的位姿，效果同样是 0.8 cm。

- XT32 支持 PTP（IEEE 1588）和 GPS PPS，**必须和 IMU 共用同一时钟**，否则去畸变反而会引入误差。逐点时间取 Hesai 驱动输出的 `timestamp` 字段。
- 如果 LIO（例如 FAST-LIO2 的 `cloud_registered_body`）已经输出去畸变后的点云，可以设置 `PreprocessConfig(deskew=False)`，避免重复处理。

**重力对齐。** LIO 的 body 系点云会跟着狗身俯仰；world 系点云的航向和位置则不跟随机器人。检测系（det frame）是 `Rz(yaw_body)`，原点为机身原点下移 `base_height`，由 `preprocess/frames.py` 构造。仿真中地面点 z 的标准差在机体系下为 51 cm，在检测系下为 6 cm。坡道和蹲姿造成的残余偏差可以用 `auto_ground` 在线修正。

**虚拟 nuScenes 雷达系**（`detection/frame_adapter.py`）。nuScenes 预训练模型学到的是"雷达离地 1.84 m、车头朝 +y"的绝对 z 先验。映射关系为 `x_M = Rz(90°)·x_D + [0,0,-1.84]`。微调数据使用同一映射，因此预训练的 z 头和高度头在微调后依然有效。

**自体滤除**（`preprocess/self_filter.py`）：
- 在**机体系**中做：腿跟着机身运动，而不是跟着重力方向。
- 放在去畸变**之前**：只需要静态外参，计算便宜。
- 包围盒按腿和轮的**摆动包络**设置，而不是站立姿态。如果包络太大，可以通过 `dynamic_boxes` 接口传入由关节编码器算出的实时盒子。
- 半径离群滤波只在近场（默认 3 m 内）做，否则会误删远处本来就稀疏的行人点。

**多帧累积**（`preprocess/accumulator.py`）。过去的帧存成世界系坐标，每帧变换到当前检测系，并附加 dt 通道。最新一帧排在最前面：pillar 每柱最多容纳 20 个点，溢出时保留下来的是最新的点，这与 det3d 推理时的顺序一致。

**近场盲区**（原方案没有提到，是最大的风险）。雷达离地约 0.6 m 时：
- 地面最近照射点在 0.6 / tan16° ≈ **2.1 m** 处；
- 1.7 m 高的行人要在 (1.7 − 0.6) / tan15° ≈ **4.1 m** 以外才能完整入视场；
- 1 m 处只能看到 0.31–0.87 m 高度的一段，也就是腿和腰。

对应措施：
- 微调数据要专门采集 4 m 内的近距离交互场景；
- 跟踪器的尺寸做轨迹级平滑，保留远处观测到的完整尺寸；
- **1.5 m 内的安全不能依赖学习检测器**，规划侧需要用几何占据或最近点做兜底；
- 结构上如果允许，把雷达下倾 5°–10° 或者装高一些，收益会非常明显。

**10 Hz。** 同意选 10 Hz。补充一个量化参考：30 m 处一个 0.6 m 宽的行人约占 6 列（0.18°）、3 条线（1°）；40 m 处只剩约 2 条线。因此 ±40 m 基本是 32 线雷达的有效上限，远距离召回主要靠跟踪来维持。

### 1.2 检测

**BPU 部署切分**（`detection/pillar_export.py`、`tools/dog/export_pillar_onnx.py`）。scatter 和 top-k/NMS 都是数据相关的操作，不适合放进 NPU 图。切分方式如下：

```
CPU  体素化 + pillar 特征装饰(10维)      numpy
NPU  PFN (Linear/BN1d 改写为 1x1 Conv/BN2d, 静态形状 1x10xP_maxxN_max)   pfn.onnx
CPU  scatter 到 BEV                       numpy
NPU  RPN neck + CenterHead (纯稠密 2D 卷积)                              rpn_head.onnx
CPU  解码 + circle NMS                     detection/decode.py
```

验证结果：ONNX（onnxruntime）与 PyTorch 在全部 head 特征图上的最大误差为 4.8e-7；NumPy 解码器与 det3d 原版 `CenterHead.predict` 的结果逐框一致（有单测覆盖）。

**NMS。** 使用中心距离的 circle NMS，替代原来依赖 CUDA 编译算子的旋转 IoU NMS。它可以直接在 ARM 上运行，CenterPoint 官方的 nuScenes 提交用的也是这种方式。

**范围与 z。** Pillar 只有一个 z 柱，z 范围只起裁剪作用，可以自由调整。Voxel 模型的 z 网格必须保持 [-5, 3]，也就是 41 格；否则稀疏主干输出的 BEV 通道数和预训练 neck 对不上。因此代码对点云做 z 裁剪，但网格本身不变。

**类别收敛**（`detection/boxes.py: NUSC_TO_DOG`）：
- car / truck / bus / trailer / construction_vehicle 映射为 vehicle；
- bicycle / motorcycle 映射为 cyclist；
- barrier 和 traffic_cone 丢弃，交给占据图处理。

由于 car、truck、bus 属于不同的 task head，映射后同一物体可能出现两个框，所以之后要再做一次类内 NMS。另外，nuScenes 的 bicycle 类包含停放的无人自行车；这对狗来说是静态障碍物，可以通过速度来区分。

**微调**（`configs/dog/dog_centerpoint_{pp,voxel}_xt32.py`）：
- 每个类别一个 task。
- 用 `tools/dog/convert_nusc_ckpt.py` 从 nuScenes 权重迁移 head。
- GT-sampling 开启。在重力对齐、原点在地面的检测系下，采样粘贴的目标才会贴地。
- 增加 ±2° 的俯仰/横滚增强（`global_pitch_roll_noise`，新加到 det3d 的 `Preprocess` 中）。
- 训练数据直接保存**预处理后的多帧点云**，与线上使用同一份预处理代码，从源头保证训练和部署一致。

**Pillar 与 Voxel 的对比方法。** 把同一份预处理输出同时喂给两个模型（`run_sequence.py --detectors pillar,voxel`），然后在人工校正过的标注上，按 0–10 / 10–20 / 20–40 m 分段比较 AP、ATE、AVE 和延迟（`compare_detectors.py`）。**不要用 Voxel 教师自己生成的标签去评估 Voxel。** 建议的定位：线上用 Pillar，因为它能部署到 BPU、延迟可控；Voxel 做离线教师，并提供精度上限作为参照。

### 1.3 标注

教师模型用 Voxel 加双翻转 TTA（`--tta`）。离线跟踪之后做轨迹级非因果精修（`dog_perception/autolabel.py`）：

- 类别：按分数加权投票；
- 尺寸和 z：取加权中位数；
- 中心点：RTS 平滑，并插值补上 0.5 s 以内的漏检；
- 朝向：运动时取速度方向，静止时取消歧后的环形均值；
- 过滤：丢弃太短或平均分太低的轨迹；
- 低置信轨迹导出到 `*_review.csv`，交给人工处理。

在仿真中，对同一个带噪声的教师（每帧 0.5 个虚警、10% 朝向翻转），行人指标变化如下：

| | AP@0.5m | ATE | AVE |
|---|---|---|---|
| 逐帧教师 | 0.59 | 0.19 m | 0.31 m/s |
| 轨迹级精修后 | **0.99** | **0.07 m** | **0.14 m/s** |

关于数据量：5k–10k 帧在 10 Hz 下只相当于 8–17 分钟的录制。相邻帧高度相关，所以**场景数量比帧数更重要**，建议至少 30 段不同场景，并按录制段划分训练集和验证集。从第二轮开始，可以用自己微调好的 Voxel 模型当教师，做自训练迭代。

### 1.4 跟踪

实现见 `tracking/tracker.py` 和 `tracking/imm.py`。

- **在 LIO 世界系中跟踪。** 狗原地转向时，机体系里静止的行人看起来会动。CenterPoint 输出的速度已经是对地速度（多帧拼接时做了自运动补偿），旋转到世界系就可以直接作为量测。单测：机器人以 1 rad/s 原地转 5 s，静止行人的估计速度小于 0.2 m/s。
- **速度头作为 KF 量测** `[x, y, vx, vy]`，而不只是在关联时用来反推位置。这样新轨迹在第一帧就有速度。
- **两阶段关联**：高分检测（≥0.35）先和所有轨迹匹配；低分检测（0.1–0.35）只用来延续已确认的轨迹，不新建轨迹。这对 32 线雷达下远处或被遮挡的行人收益很大。
- **门限**：按类别设定（行人 1.5 m、骑行者 2.5 m、车辆 3 m），并随预测协方差自适应放大。
- **生命周期**：
  - tentative 状态命中 2 次后确认，期间容忍 1 帧漏检；
  - 保活时间：行人 1.5 s，车辆和骑行者 1.0 s。狗的视角低，遮挡时间长，原方案的 5–10 帧（0.5–1 s）对行人来说偏短；
  - 保活期间输出 `coasting` 标志和随时间膨胀的协方差，由规划侧决定是否使用。
- **框属性平滑**：z 和尺寸用 EMA；车辆朝向做 180° 消歧，速度超过 1 m/s 时与运动方向对齐；行人朝向在行走时取速度方向。

### 1.5 预测

**第一档**（`prediction/kinematic.py`）：
- 车辆和骑行者：CV + CT 双模型 IMM。每个模态单独外推，得到"直行"和"持续转弯"两条轨迹，并附带模态概率。
- 行人：低噪声 CV + 高噪声 CV。两条轨迹重合时按矩匹配合并，结果就是"用过程噪声膨胀表达不确定性"。
- 输出 3 s、步长 0.5 s 的均值和协方差。

**第二档**（`prediction/model.py`、`features.py`、`dataset.py`）：
- 网络结构：GRU 或小 Transformer 编码过去 2 s；社交注意力作用于邻居，并**把机器狗自身作为一个邻居**（行人会避让机器人，这是狗场景里最强的交互信号）；BEV 占据图可选接入（`use_raster`）；K=6 个模态，训练方式为 WTA 下的 Laplace NLL 加模态分类交叉熵。
- 数据：输入用跟踪器的**因果**估计，与线上一致；真值用 RTS 平滑后的**非因果**轨迹；丢弃未来被遮挡超过 20% 的窗口。
- `TrackLogger` 会同时记录**线上 IMM 的预测**，保证两档在完全相同的样本上比较。我最初用"在跟踪器输出上重新跑一遍 IMM"作为基线，这相当于双重滤波，速度估计滞后，结果比 CV 还差；改用线上真实输出后，基线才可信。
- 指标：多模态看 minADE、minFDE 和 MR@2m；单模态看 ADE1 和 FDE1。WTA 训练出的 top-1 模态对应的是某一个具体的机动假设，单独拿出来往往不如 CV，所以**规划应该使用全部模态和它们的概率**。
- 上线门槛：第二档在同一份 held-out 数据上的 minFDE 和 MR 都要优于第一档，才切换。线上历史不足 0.5 s 的轨迹会自动回退到 IMM。

---

## 2. 模块与文件对应

```
LidarScan(XT32, 逐点时间) ─┐        IMU 400Hz ─┐   LIO 10Hz ─┐
                          ▼                   ▼              ▼
             [1] preprocess/  self_filter → deskew(PoseBuffer/ImuPropagator)
                              → det frame(重力对齐) → 5帧累积 [x,y,z,i,dt]
                          ▼
             [2] detection/   ModelFrame(虚拟nuScenes雷达系) → CenterPoint-Pillar | Voxel (det3d)
                              → nuScenes10类→3类 + 类内NMS → Detections(det frame)
                          ▼  T_world_det
             [3] tracking/    IMM(CV/CT) + 两阶段关联 + 生命周期 (LIO world frame)
                          ▼
             [4] prediction/  第一档 IMM 外推 | 第二档 GRU/Transformer 多模态 (IMM 兜底)
                          ▼
             pipeline.to_local() → 重力对齐的局部系, 给规划
```

| 路径 | 内容 |
|---|---|
| `dog_perception/pose_buffer.py` | IMU 频率位姿缓冲（SLERP），LIO + 陀螺积分传播 |
| `dog_perception/preprocess/` | 去畸变、自体滤除、检测系、地面高度估计、多帧累积、`Preprocessor` |
| `dog_perception/detection/centerpoint.py` | det3d Pillar/Voxel 统一封装，支持范围/NMS 配置和双翻转 TTA |
| `dog_perception/detection/pillar_export.py`, `decode.py` | BPU/ONNX 切分和 NumPy 解码 |
| `dog_perception/detection/evaluate.py` | 中心距离 AP（含分距离段）、ATE、AVE |
| `dog_perception/tracking/` | IMM、贪心/匈牙利匹配、多目标跟踪器 |
| `dog_perception/prediction/` | 第一档 IMM 外推、特征构造、自监督数据、模型、训练/评估 |
| `dog_perception/autolabel.py` | 离线轨迹级精修 |
| `dog_perception/pipeline.py` | 四段串联 `PerceptionPipeline` |
| `dog_perception/sim.py` | XT32 + 足式步态仿真器（测试和演示用） |
| `det3d/datasets/dog/` + pipeline 改动 | `DogDataset`、GT-sampling、俯仰/横滚增强 |
| `configs/dog/` | 3 类 Pillar/Voxel 微调配置（±40.8 m、5 帧） |
| `tools/dog/rosbag_to_sequence.py` | ROS1/ROS2 bag 转序列（无需安装 ROS，依赖 `rosbags`） |
| `tools/dog/run_sequence.py` | 录制数据回放，Pillar/Voxel 并行跑，输出结果、跟踪日志和帧数据 |
| `tools/dog/auto_label.py`, `export_dataset.py` | 教师自动标注，导出为 DogDataset 并建 GT 库 |
| `tools/dog/compare_detectors.py` | Pillar 与 Voxel 对比（AP、分段 AP、延迟、轨迹碎片化） |
| `tools/dog/convert_nusc_ckpt.py` | nuScenes 10 类到 3 类的 head 权重迁移 |
| `tools/dog/export_pillar_onnx.py` | 导出 BPU/TRT 用的两段 ONNX 并校验 |
| `tools/dog/train_predictor.py` | 第二档训练，与 CV/IMM 对比 |
| `tools/dog/ros2_node.py` | 在线 ROS 2 节点（MarkerArray 可视化） |
| `tools/dog/run_sim_demo.py` | 仿真端到端演示，输出各段指标 |

---

## 3. 使用流程

所有命令都在仓库根目录执行，并设置 `PYTHONPATH=.`。

```bash
# 0) 录制数据 -> 序列（base_height 为机身原点离地高度，extrinsic 为雷达在 LIO body 系下的外参）
python tools/dog/rosbag_to_sequence.py --bag rec_001/ --out data/rec_001 \
    --lidar-topic /lidar_points --imu-topic /imu/data --odom-topic /Odometry \
    --extrinsic-t 0.2 0 0.15 --extrinsic-rpy 0 0 0 --base-height 0.45

# 1) 用 nuScenes 预训练的 Pillar 和 Voxel 直接跑（两者输入完全相同）
python tools/dog/run_sequence.py --seq data/rec_001 --detectors pillar,voxel \
    --ckpt-pillar nusc_pp.pth --ckpt-voxel nusc_voxel.pth --save-frames --out work_dirs/rec_001

# 2) 离线教师自动标注 -> 人工修正 review.csv 中的轨迹 -> 对比两个检测器
python tools/dog/auto_label.py --seq data/rec_001 --ckpt-voxel nusc_voxel.pth --tta --out data/rec_001/labels.pkl
python tools/dog/compare_detectors.py --labels data/rec_001/labels.pkl \
    --results pillar=work_dirs/rec_001/pillar/results.pkl voxel=work_dirs/rec_001/voxel/results.pkl

# 3) 微调：导出数据集 + GT 库，迁移 head，用 det3d 训练
python tools/dog/export_dataset.py --recording work_dirs/rec_001 data/rec_001/labels.pkl \
    --recording work_dirs/rec_002 data/rec_002/labels.pkl --val-recordings rec_002 --out data/dog --gt-db
python tools/dog/convert_nusc_ckpt.py --src nusc_pp.pth --dst work_dirs/dog_pp_init.pth
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/dog/dog_centerpoint_pp_xt32.py
python tools/dist_test.py configs/dog/dog_centerpoint_pp_xt32.py --work_dir work_dirs/dog_pp --checkpoint work_dirs/dog_centerpoint_pp_xt32/latest.pth

# 4) 部署 Pillar：导出两段 ONNX 并与 PyTorch 对齐校验（之后交给 OpenExplorer / TensorRT）
python tools/dog/export_pillar_onnx.py --config configs/dog/dog_centerpoint_pp_xt32.py \
    --checkpoint work_dirs/dog_centerpoint_pp_xt32/latest.pth --out deploy/ --check frame.npy

# 5) 第二档预测：用多段录制的跟踪日志训练，并与 IMM 比较
python tools/dog/train_predictor.py --logs "work_dirs/rec_*/pillar/track_log.npz" \
    --val-logs work_dirs/rec_009/pillar/track_log.npz --encoder gru --out work_dirs/pred/gru.pt

# 在线（ROS 2）
python3 tools/dog/ros2_node.py --ros-args -p detector:=dog_pillar -p checkpoint:=pp.pth \
    -p predictor_model:=work_dirs/pred/gru.pt -p extrinsic_t:="[0.2,0.0,0.15]" -p base_height:=0.45

# 仿真演示和测试
python tools/dog/run_sim_demo.py --frames 80 --plot 3 --out work_dirs/sim_demo
python -m pytest tests/dog -q
```

---

## 4. 仿真验证结果

仿真器（`dog_perception/sim.py`）模拟的内容：
- XT32 几何：32 线，-16°~+15°，2000 列 × 10 Hz，逐点时间，1 cm 测距噪声；
- 卷帘畸变：每一列都从它自己时刻的位姿发射光线，运动目标也按该时刻的位置摆放；
- 步态：6° 俯仰、3° 横滚、2 cm 起伏，速度 1.2 m/s，转向 0.15 rad/s；
- 机身载荷的自遮挡；
- 400 Hz 陀螺和晚到 30 ms 的 10 Hz 里程计（带噪声）。

检测器使用 **oracle**，即 GT 加噪声。没有真实权重时，这是验证跟踪和预测的唯一办法。检测精度本身需要用真实数据配合 `compare_detectors.py` 来评估。

**预处理**（80 帧，每 5 帧统计一次）

| 指标 | 结果 |
|---|---|
| 墙面点距误差 RMS（未去畸变 → 去畸变） | 0.113 m → **0.027 m**（残差主要来自注入的里程计噪声：1 cm / 0.1°；用无噪声 IMU 频率位姿时为 0.8 cm） |
| 地面点 z 标准差（机体系 → 检测系） | 0.52 m → **0.055 m** |
| 自体点（机身载荷） | 2412 → **0** 点/帧 |

**跟踪**（oracle 检测，16 个运动目标，80 帧；只统计点数 ≥5 的可见目标）

| 指标 | 结果 |
|---|---|
| MOTA | 0.903 |
| 召回 | 0.906（oracle 的检测率本身约 0.9） |
| ID 切换 | 2 |
| 位置 RMSE | 6.9 cm（检测噪声 10 cm） |
| 速度 RMSE | 0.23 m/s（检测速度噪声 0.25 m/s） |

**延迟**（x86 4 vCPU，Python）：预处理 23 ms，跟踪 6 ms，IMM 预测 13 ms（约 16 条轨迹）。

**预测**：用 10 段 × 60 s 仿真跟踪日志训练第二档（9 段训练、1 段验证，约 2.8 万个窗口，CPU 上训练 25 个 epoch）。

**(a) 留出的仿真跟踪日志**（2849 个窗口，第一档用线上记录的 IMM 输出）：

| 预测器 | minADE | minFDE | ADE1 | FDE1 | MR@2m |
|---|---|---|---|---|---|
| CV（最后速度外推） | 0.808 | 1.656 | 0.808 | 1.656 | 0.309 |
| 第一档 IMM (CV/CT) | 0.722 | 1.494 | 0.802 | 1.638 | 0.265 |
| 第二档 GRU（K=6） | 0.350 | 0.658 | 0.720 | 1.489 | 0.033 |
| 第二档 小 Transformer（K=6） | **0.328** | **0.617** | **0.691** | **1.437** | **0.025** |

**(b) 激光仿真端到端**（80 帧，对比目标**真实**的未来 3 s；机器人速度 1.2 m/s，和训练日志的 0.4 m/s 不同，属于分布外测试）：

| 预测器 | minADE | minFDE | ADE1 | FDE1 | MR@2m |
|---|---|---|---|---|---|
| 第一档 IMM | 0.734 | 1.507 | **0.822** | **1.703** | 0.302 |
| 第二档 GRU | 0.449 | **0.867** | 0.911 | 1.867 | **0.083** |
| 第二档 Transformer | **0.441** | 0.868 | 0.932 | 1.925 | 0.103 |

结论：
- 多模态指标（minFDE、MR）第二档大幅领先。
- 在分布外场景下，第二档概率最大的单条轨迹（FDE1）**比 IMM 差**。这正是第 1.5 节强调的：规划侧应使用全部模态及其概率；是否从第一档切换到第二档，要看它在自己留出数据上的表现是否过门槛。

注意：仿真中的目标按"分段恒定速度/转向率"随机机动，**没有社交交互**，而第二档的主要优势（对机器人的避让等交互）在仿真里体现不出来。表中第二档的领先主要来自多模态覆盖，真实效果需要用实车日志重新评估。

---

## 5. 对原仓库的兼容性修改

新环境（Python ≥3.10、新版 numba、无 CUDA 的 CPU 机器）下原仓库无法导入或训练，做了以下最小修复：

- `det3d/models/__init__.py`、`backbones/__init__.py`：`import importlib.util`（Python 3.11 下 `importlib.util` 不会被隐式导入）；缺少 iou3d 编译算子时，two-stage 的 roi_heads 改为可选导入。
- `det3d/torchie/trainer/checkpoint.py`：spconv 改为可选依赖，Pillar 模型不再需要它。
- `det3d/solver/*.py`、`torchie/parallel/collate.py`：`collections.Iterable/Sequence/Mapping` 改为 `collections.abc`，这些名字在 Python 3.10 中已被移除，而 Ubuntu 22.04 / JetPack 6 自带的就是 3.10。
- `det3d/core/bbox/geometry.py: points_in_convex_polygon_jit`：新版 numba（nopython 默认开启）不再支持列表花式索引，改写为显式循环，已对照原公式验证。GT-sampling 的碰撞检测依赖这个函数。

## 6. 已知局限（没有在本环境中验证的部分）

- **没有真实 XT32 数据和 nuScenes 预训练权重**（官方权重托管在 OneDrive，本环境无法下载）。所有检测相关的数字只验证了管线和一致性，Pillar 与 Voxel 的精度和延迟对比需要在你的机器上用步骤 1–2 实测。
- Voxel 模型的前向需要 CUDA 版 spconv（CPU 版 spconv 不支持带 bias 的卷积），本环境只验证了模型构建和网格尺寸。
- 本环境没有 ROS，`ros2_node.py` 没有实际运行过。bag 转换器已用合成的 ROS2 bag 测试。
- BPU 量化（OpenExplorer / hb_mapper）没有做，流程只到 ONNX 为止。PFN 中的 max-pool 和 concat 是否被 Nash 工具链完全支持，需要在 S100P 上确认。
- 预处理、跟踪和预测是 Python 参考实现。在 x86 4 核上实测：预处理约 21 ms（5 帧、约 22 万点），跟踪约 4 ms，IMM 预测约 15 ms（16 个目标）。上板时建议把预处理用 C++/CUDA 重写，目标在 5 ms 以内。
- 仿真中的目标没有社交交互，所以第二档在真实数据上的收益（主要来自对机器人的避让等交互）需要用实车日志重新评估。
