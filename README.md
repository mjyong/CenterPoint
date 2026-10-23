# CenterPoint 代码流程与架构

用鸟瞰图上的中心点做 3D 检测和跟踪。论文：[Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275)（Yin, Zhou, Krähenbühl, CVPR 2021）。

本仓库在上游检测代码之外，加了轮足机器狗 XT32 的四段流水线，见 `dog_perception/`。方案和命令在 [docs/DOG_PIPELINE.md](docs/DOG_PIPELINE.md)。用自采 bag 做自动标注、审核和微调，见 [docs/DOG_FINETUNE.md](docs/DOG_FINETUNE.md)；第一次可以先照 [docs/LABELING_TUTORIAL.md](docs/LABELING_TUTORIAL.md) 用样例 bag 走一遍。从数据到训练、评估对比、部署的整轮开发流程见 [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md)。

不枚举锚框朝向。点云先压成鸟瞰图，`CenterHead` 在图上找物体中心，再在中心处回归尺寸、高度、朝向和速度。跟踪把当前中心加上速度外推，和上一帧中心做最近点匹配。

## 目录


| 路径                                                           | 职责                                                                                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `configs/`                                                   | 把 reader、backbone、neck、head、数据和训练策略拼成一次实验。nuScenes 在 `configs/nusc/`，Waymo 在 `configs/waymo/`，机器狗微调在 `configs/dog/` |
| `det3d/models/`                                              | 网络。`registry.py` 登记模块名，`builder.py` 按配置里的 `type` 实例化                                                                |
| `det3d/datasets/`                                            | 读点云、预处理、体素化、把 GT 画成 heatmap                                                                                         |
| `det3d/torchie/`                                             | 训练循环、分布式、checkpoint、hook。入口在 `torchie/apis/train.py`                                                                |
| `det3d/ops/`                                                 | 体素化、旋转 IoU NMS、可变形卷积。需要 CUDA 编译                                                                                     |
| `tools/train.py` `tools/dist_test.py` `tools/create_data.py` | 训练、评测、把官方数据集转成 info pkl                                                                                             |
| `tools/nusc_tracking/` `tools/waymo_tracking/`               | 检测结果出来之后的离线跟踪                                                                                                       |
| `dog_perception/` `tools/dog/`                               | 机器狗四段流水线，不改检测器内部，包在外面                                                                                               |
| `third_party/SUSTechPOINTS/`                                 | 3D 点云标注工具（GPL-3.0），用来逐框审核自动标注，见 `VENDORED.md` |

## 文档

| 文档 | 内容 |
|---|---|
| [DOG_PIPELINE.md](docs/DOG_PIPELINE.md) | 四段流水线（预处理、检测、跟踪、预测）的设计、模块和文件对应、Orin 部署 |
| [DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) | 一轮迭代：数据 → 标注 → 训练 → 评估对比 → 部署，每步的进入条件 |
| [DOG_FINETUNE.md](docs/DOG_FINETUNE.md) | 从 bag 到微调权重每一步的参考说明；车辆分 head、只检车辆、遮挡 / 待审核 / 运动判断、SUSTechPOINTS 逐框审核 |
| [LABELING_TUTORIAL.md](docs/LABELING_TUTORIAL.md) | 用两段样例 bag 实际走一遍标注，带命令输出和 SUSTechPOINTS 截图 |
| [SERVER_TASKS.md](docs/SERVER_TASKS.md) | 服务器上用 0921 数据训练车辆模型的任务清单、结果和复核（当前进度看这里） |
| [DATA_COLLECTION_PLAN.md](docs/DATA_COLLECTION_PLAN.md) | 厂区下一轮：教师用哪个模型、采什么数据（运动的货车、可控场景、挑哪些来标） |
| [BAG_SPLIT.md](docs/BAG_SPLIT.md) | 长 bag 体检、切场景、位姿融合、训练 / 验证划分；0921 数据准备的记录 |
| [DATA_LINEAGE.md](docs/DATA_LINEAGE.md) | 从 rosbag 话题字段到训练集：每个脚本读写什么、审核改了哪些字段（附 [思维导图](docs/data_lineage_mindmap.html)） |
| [SENSORS.md](docs/SENSORS.md) | XT32、组合导航和天线、鱼眼相机的参数和外参 |
| [DOG_DATAFLOW.md](docs/DOG_DATAFLOW.md) | 线上一帧的调用链和每一步的数据 shape、延迟 |
| [DETECTION_POSTPROCESS.md](docs/DETECTION_POSTPROCESS.md) | 检测后处理逐步讲解（真实一帧） |
| [TRACKING_PREDICTION.md](docs/TRACKING_PREDICTION.md) | 跟踪（含运动状态）和预测的原理与代码 |
| [CENTERPOINT_NETWORK.md](docs/CENTERPOINT_NETWORK.md) / [CENTERPOINT_VOXEL_NETWORK.md](docs/CENTERPOINT_VOXEL_NETWORK.md) | Pillar / Voxel 网络逐层详解 |
| [NUSC_CLIP_EVAL.md](docs/NUSC_CLIP_EVAL.md) | nuScenes 片段上的检测、跟踪、预测评估 |
