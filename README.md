# CenterPoint 代码流程与架构

用鸟瞰图上的中心点做 3D 检测和跟踪。论文：[Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275)（Yin, Zhou, Krähenbühl, CVPR 2021）。

本仓库在上游检测代码之外，加了轮足机器狗 XT32 的四段流水线，见 `dog_perception/`。方案和命令在 [docs/DOG_PIPELINE.md](docs/DOG_PIPELINE.md)。

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


