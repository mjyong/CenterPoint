# CenterPoint 代码流程与架构

用鸟瞰图上的中心点做 3D 检测和跟踪。论文：[Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275)（Yin, Zhou, Krähenbühl, CVPR 2021）。

本仓库在上游检测代码之外，加了轮足机器狗 XT32 的四段流水线，见 `dog_perception/`。方案和命令在 [docs/DOG_PIPELINE.md](docs/DOG_PIPELINE.md)。

不枚举锚框朝向。点云先压成鸟瞰图，`CenterHead` 在图上找物体中心，再在中心处回归尺寸、高度、朝向和速度。跟踪把当前中心加上速度外推，和上一帧中心做最近点匹配。

## 目录

| 路径 | 职责 |
|---|---|
| `configs/` | 把 reader、backbone、neck、head、数据和训练策略拼成一次实验。nuScenes 在 `configs/nusc/`，Waymo 在 `configs/waymo/`，机器狗微调在 `configs/dog/` |
| `det3d/models/` | 网络。`registry.py` 登记模块名，`builder.py` 按配置里的 `type` 实例化 |
| `det3d/datasets/` | 读点云、预处理、体素化、把 GT 画成 heatmap |
| `det3d/torchie/` | 训练循环、分布式、checkpoint、hook。入口在 `torchie/apis/train.py` |
| `det3d/ops/` | 体素化、旋转 IoU NMS、可变形卷积。需要 CUDA 编译 |
| `tools/train.py` `tools/dist_test.py` `tools/create_data.py` | 训练、评测、把官方数据集转成 info pkl |
| `tools/nusc_tracking/` `tools/waymo_tracking/` | 检测结果出来之后的离线跟踪 |
| `dog_perception/` `tools/dog/` | 机器狗四段流水线，不改检测器内部，包在外面 |

## 检测器怎么串起来

配置里的 `model.type` 决定类。`SingleStageDetector`（`det3d/models/detectors/single_stage.py`）按固定顺序建四个子模块：

```
reader  →  backbone  →  neck  →  bbox_head
```

两条常用检测器只是 `extract_feat` 不同：

- `PointPillars`（`detectors/point_pillars.py`）：数据管道已经体素化。`PillarFeatureNet` 把一个柱里的点编成一个向量，`PointPillarsScatter` 按柱坐标铺成 BEV，`RPN` 做 2D 卷积。
- `VoxelNet`（`detectors/voxelnet.py`）：`VoxelFeatureExtractorV3` 提体素特征，`SpMiddleResNetFHD`（`backbones/scn.py`）用稀疏 3D 卷积压成 BEV，再进同一个 `RPN`。

`forward(example, return_loss)` 训练和测试共用这一条：

```
example → extract_feat → CenterHead.forward
              ├─ return_loss=True  → CenterHead.loss
              └─ return_loss=False → CenterHead.predict
```

`TwoStageDetector`（`detectors/two_stage.py`）在这一阶段框的基础上，用 BEV 特征再修一次（`models/second_stage/`、`models/roi_heads/`）。第一阶段已经能出框，两阶段是可选精修。

## 一次前向

以 nuScenes Pillar 配置 `configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py` 为例。范围 `[-51.2, 51.2]` 米，柱大小 `(0.2, 0.2, 8)`，最多 10 帧。

数据管道（同文件里的 `train_pipeline`）：

```
LoadPointCloudFromFile
  → LoadPointCloudAnnotations
  → Preprocess          多帧拼到当前帧，并做增强
  → Voxelization        点云变成柱
  → AssignLabel         GT 中心画成高斯 heatmap，并记下回归目标
  → Reformat            收成模型吃的 example 字典
```

网络（同文件 `model` 字典）：

```
PillarFeatureNet     每个柱：点的坐标装饰 + 两层 PFN，得到 64 维
PointPillarsScatter  按柱的 (batch, x, y) 涂到 BEV
RPN                  三次下采样再上采样，拼成多尺度 BEV
CenterHead           每个 task 一组输出
```

`CenterHead`（`det3d/models/bbox_heads/center_head.py`）先过一层共享卷积，再按 `tasks` 分头。nuScenes 把 10 类收成 6 个 task（车一类、卡车和工程车一类，等等），每个 task 输出：

| 键 | 含义 |
|---|---|
| `hm` | 该类中心的 heatmap |
| `reg` | 中心相对格子的亚像素偏移 |
| `height` | 中心 z |
| `dim` | 长宽高 |
| `rot` | 朝向，用 sin/cos 两个通道 |
| `vel` | 鸟瞰速度。没有这一项时框是 7 维，有则是 9 维 |

损失：heatmap 用 `FastFocalLoss`，只在 GT 中心和其邻域算；框用 `RegLoss`，只在有物体的格子上算。`code_weights` 把朝向两个通道的权重降到 0.2。

推理 `CenterHead.predict`：heatmap 取局部峰值，用 `reg/height/dim/rot/vel` 解回 3D 框，再按 `test_cfg` 做分数阈值和 NMS。`double_flip` 时把原图和三种翻转拼在 batch 里，解框前翻回同一坐标系。

## 训练和评测从哪进

`tools/train.py` 读配置，`build_detector` 建模型，`build_dataset` 建数据，然后交给 `det3d.torchie.apis.train_detector`。优化器、学习率和 hook 都在配置末尾，不在模型类里。

```bash
# 仓库根目录，且已按 docs/INSTALL.md、docs/GETTING_START.md 准备好数据
python tools/train.py configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py

python tools/dist_test.py configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py \
    --work_dir work_dirs/nusc_pp --checkpoint work_dirs/nusc_pp/latest.pth
```

换数据集或换 Pillar / Voxel，改的是配置文件，不是 `forward`。新模块放到对应目录并 `@READERS.register_module`（或 `BACKBONES` / `NECKS` / `HEADS` / `DETECTORS`），配置里写同一个 `type` 字符串即可。

## 跟踪

检测已经回归出速度，跟踪不再单独做运动模型。`tools/nusc_tracking/pub_tracker.py` 的 `PubTracker.step_centertrack`：

1. 用 `中心 + 速度 × (-time_lag)` 把当前检测投回上一帧时刻。
2. 和已有轨迹中心算距离。类别不同，或距离超过该类速度误差门限的，代价设成无穷大。
3. 默认贪心匹配；`hungarian=True` 时改用匈牙利算法。
4. 没配上的检测新建轨迹，没配上的轨迹年龄加一，超过 `max_age` 删除。

Waymo 的对应实现在 `tools/waymo_tracking/tracker.py`。

## 机器狗流水线

`dog_perception/` 不替换 `det3d` 的检测器，只规定检测器前后各做什么。`PerceptionPipeline.on_scan`（`dog_perception/pipeline.py`）的顺序：

```
IMU / LIO 位姿
  → Preprocessor     自体滤除、按陀螺去畸变、变到重力对齐的检测系、累积 5 帧
  → detector         CenterPoint Pillar 或 Voxel，输入是检测系点云
  → 变到 LIO 世界系
  → MultiObjectTracker   IMM（匀速 + 协调转弯）+ 两阶段关联
  → IMMPredictor 或 LearnedPredictor
  → to_local()       再变回规划用的局部系
```

和上游跟踪的差别：关联和滤波在世界系做，速度头当卡尔曼量测，而不只是用来平移中心。部署把 Pillar 切成两段 ONNX（PFN、RPN+Head），scatter 和 NMS 留在 CPU，见 `dog_perception/detection/pillar_export.py`。

仿真测试：

```bash
PYTHONPATH=. python -m pytest tests/dog
```

## 建议阅读顺序

1. `configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py`：看 `tasks`、`model`、`train_pipeline`、`test_cfg` 四块如何对应到类。
2. `det3d/models/detectors/point_pillars.py` 的 `forward`：确认 example 字典里有哪些键。
3. `det3d/models/bbox_heads/center_head.py` 的 `forward`、`loss`、`predict`：检测器的实际输出。
4. `det3d/datasets/pipelines/preprocess.py` 与体素化：点怎么变成柱，heatmap 在哪画。
5. `tools/train.py` 再跳到 `det3d/torchie/apis/train.py`：配置如何进训练循环。
6. `tools/nusc_tracking/pub_tracker.py`：速度头怎么被用来关联。
7. `dog_perception/pipeline.py`，然后 `docs/DOG_PIPELINE.md`：机器狗是怎么把同一检测器接到 XT32 上的。
