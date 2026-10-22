# 用自采 bag 微调检测器：提取 → 自动标注 → 审核 → 训练

从 kygi665 录制（Hesai XT32 + RTK 组合导航）到微调后的 Pillar 模型的完整流程。所有命令在仓库根目录执行，设置 `PYTHONPATH=.`。通用 LIO 里程计的写法见 [DOG_PIPELINE.md](DOG_PIPELINE.md) 第 3 节。

训练之后的评估、对比、验收和部署，以及整轮迭代的做法，见 [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)。

传感器（XT32、组合导航和天线、鱼眼相机）的参数和外参见 [SENSORS.md](SENSORS.md)。

每一步读写哪些文件、用到 bag 的哪些字段、审核改了哪些字段，见 [DATA_LINEAGE.md](DATA_LINEAGE.md)（附思维导图 [data_lineage_mindmap.html](data_lineage_mindmap.html)）。

**第一次做，先照着 [LABELING_TUTORIAL.md](LABELING_TUTORIAL.md) 用仓库里的两段样例 bag 走一遍**：每一步的命令、应有的输出、SUSTechPOINTS 的界面截图和操作都在里面。

中间产物都放在 `data/` 下（`data/seq`、`data/runs`、`data/labels`、`data/dog`），这个目录被 .gitignore 忽略，点云不会被误提交。

```
bag ─① rosbag_to_sequence ─► 序列 ─② run_sequence --save-frames ─► frames/*.npy（模型输入）+ 教师结果
   ─③ auto_label ─► labels.pkl + labels_tracks.csv
   ─④a review_labels（Foxglove 里看，按轨迹改）─► labels_reviewed.pkl
   ─④b SUSTechPOINTS（逐框改、补漏标，验证集必须做）─► labels_sustech.pkl
   ─⑤ export_dataset ─► data/dog（DogDataset + GT 库）─⑥ convert_nusc_ckpt + train ─► 微调权重
   ─⑦ 评估 ─⑧ 部署（ONNX → TensorRT）─⑨ 用微调后的模型当教师，下一轮
```

## 0. 采集

- 场景多比帧数多重要：至少 30 段不同场景，总共 5k–10k 帧（10 Hz 下 1 万帧只有约 17 分钟）。相邻帧高度相关。
- 专门录 4 m 以内的近距离交互（人从狗身边走过、绕行、停下）。
- 留出整段录制做验证集，训练集和验证集按录制段划分，不按帧。
- RTK 要固定解。第 ① 步会打印 `pos_type`（样例 bag 是 56，按 NovAtel 的定义是 INS_RTKFIXED）。不是固定解时世界系轨迹会漂，自动标注的轨迹就会断、会抖。

## ① bag → 序列

```bash
python tools/dog/rosbag_to_sequence.py --bag <bag 目录> --out data/seq/rec_001 \
    --lidar-topic /robot/topic/rdap/top_point_cloud2 \
    --imu-topic /rte/rdap/rtk/imus/data_raw --inspvax-topic /rte/rdap/rtk/inspvax
```

加了 `--inspvax-topic` 之后，雷达 180° 外参、IMU 的 deg/s 单位、机身离地高度估计和地面找平都自动处理（见 DOG_PIPELINE.md 第 9 节）。每段 bag 一个序列目录，目录名就是录制名，不能重复。

几分钟以上的长 bag（几十 GB）先体检、再一次性切成多个 30 s 左右的场景并逐个检查，见 [BAG_SPLIT.md](BAG_SPLIT.md)（`bag_survey.py` → `rosbag_to_sequence.py --segments` → `sequence_preview.py`）。每个场景就是一个录制名。

## ② 预处理 + 教师推理 + 保存帧

```bash
python tools/dog/run_sequence.py --seq data/seq/rec_001 --detectors voxel --tta \
    --ckpt-voxel work_dirs/pretrained/nusc_voxel.pth --save-frames --out data/runs/rec_001
```

- `data/runs/rec_001/frames/<stamp>.npy` 是预处理后的 5 帧拼接点云（检测系），和狗上线时模型的输入完全一样。训练直接用它，训练和部署走的是同一套预处理。
- 教师用 Voxel + 双翻转 TTA。nuScenes Voxel 权重在仓库里用 Git LFS 存（`work_dirs/pretrained/nusc_voxel.pth`，103 MB），克隆后要 `git lfs install && git lfs pull` 才是真文件，否则只是 134 字节的指针。
- Voxel 必须在 GPU 上跑：CPU 上 spconv 直接报错（`cpu don't support act and bias`）。
- 没有 GPU 时可以先用 Pillar 试跑：`--detectors pillar --ckpt-pillar work_dirs/pretrained/nusc_pp.pth`。

## ③ 自动标注

```bash
python tools/dog/auto_label.py --results data/runs/rec_001/voxel/results.pkl --out data/labels/rec_001/labels.pkl
```

对整段录制离线跟踪，再逐条轨迹非因果精修（`dog_perception/autolabel.py`）：

- 类别按分数加权投票；尺寸和 z 取加权中位数；
- 车的子类（car / truck / bus，即教师是哪个 head 出的框）在车的观测里同样加权投票，写进 `subclass` 字段，训练时用作 head 类别（第 12 节）；行人、骑行者的 `subclass` 与类别相同；
- 中心 RTS 平滑，0.5 s 以内的漏检插值补上；
- **静止 / 运动**：某一帧只有在轨迹速度（±0.5 s 差分）> 0.4 m/s **并且**教师速度头 > 0.15 m/s 时才算运动，否则速度为 0。再去掉毛刺：运动中短于 0.5 s 的停顿算运动，短于 1 s 的运动段算静止（除非占了轨迹的一半以上，第 14 节）。从头到尾都没动的轨迹，如果 90% 的框离中位数不超过车 1.0 m / 人 0.5 m / 骑行 1.5 m，位置固定为加权中位数；散得更开说明轨迹漂了或跟错了物体，保留平滑后的路径；
- 朝向：运动快（> 1 m/s）时取运动方向；否则取框朝向的环形均值，180° 歧义用运动方向消解，从没动过就用分数最高的那个框；
- 太短（< 5 次观测）或平均分太低（< 0.3）的轨迹丢掉；平均分 < 0.5、并且教师给 ≥ 0.6 分的帧不到 5 帧的，标为待审核（第 14 节）；
- 每个框记下这一帧教师的分数 `det_score`（插值的帧为 0），以及框里的点数 `points`（当前帧）/ `points_all`（模型输入的 5 帧，第 14 节）。

输出 `labels.pkl`（每帧的框，检测系）和 `labels_tracks.csv`（每条轨迹一行，待审核的在前）。

### 为什么要判静止

狗走动时，一辆停着的车被看到的那一面在变，教师给的框中心跟着漂。原来直接对平滑后的轨迹求导，这个漂移就变成了速度，朝向也跟着速度走：

- 只看轨迹：停着的车也有零点几 m/s，朝向跟着漂移方向走；
- 只看教师速度头：静止物体几乎都是 0，但慢速运动读得偏小：nuScenes 上被误判为静止的运动车，真值中位数 0.77 m/s，速度头只给 0.18 m/s，轨迹给 0.84 m/s；
- 所以两个都要过阈值。轨迹阈值 0.4 m/s 挡住漂移，速度头阈值 0.15 m/s 挡住静止物体，又不漏掉慢速运动。

**运动样例 bag**（狗在走，教师为预训练 Pillar），标签速度中位数和 > 1 m/s 的比例：

| | vehicle | pedestrian | cyclist |
|---|---|---|---|
| 修改前 | 0.42 m/s，17% | 0.41 m/s，0% | 0.40 m/s，28% |
| 修改后 | 0.00 m/s，7% | 0.00 m/s，0% | 0.00 m/s，3% |

修改后还有超过 1 m/s 的框，来自三条 30 m 以外、平均分 0.31–0.37 的轨迹（两条车、一条骑行），都在待审核之列；没有真值，是不是真在动要人看。

**nuScenes mini_val**（2 个场景、81 个关键帧，自车在开，停着的车同样是从变化的视角看到的）。用 `tools/dog/nusc_autolabel_eval.py` 和真值对比，匹配距离 2 m；AVE / AOE 是匹配上的框的平均速度 / 朝向误差（m/s / rad），"静止"指真值速度 < 0.2 m/s，"运动"指 > 0.5 m/s：

| | 版本 | TP / FP | ATE | AVE | AOE | 静止物体标成 > 0.5 m/s | 运动物体 AVE |
|---|---|---|---|---|---|---|---|
| vehicle | 教师原始框 | 1406 / 65 | 0.21 | 0.17 | 0.26 | 2% | 1.14 |
| | 修改前 | 1553 / 93 | 0.22 | 0.23 | 0.15 | 9% | 0.33 |
| | 修改后 | 1553 / 93 | 0.21 | **0.08** | **0.12** | **0%** | 0.36 |
| pedestrian | 教师原始框 | 842 / 219 | 0.17 | 0.23 | 0.43 | 1% | 0.31 |
| | 修改前 | 954 / 360 | 0.16 | 0.15 | 0.46 | 6% | 0.14 |
| | 修改后 | 953 / 361 | 0.16 | **0.13** | **0.37** | **1%** | 0.14 |
| cyclist | 教师原始框 | 101 / 36 | 0.22 | 0.12 | 0.84 | 2% | 0.52 |
| | 修改前 | 153 / 22 | 0.28 | 0.36 | 0.41 | 20% | 0.24 |
| | 修改后 | 155 / 21 | 0.24 | **0.06** | **0.28** | **0%** | 0.24 |

教师原始框取上报阈值以上的。自动标注比教师原始框多匹配上 10%（车）、13%（人）、53%（骑行）的真值：低分框经过跟踪保留了下来，漏检帧也插值补上了。代价是行人 FP 从 219 增加到 361，审核时要重点看行人。剩下的一个退步是运动车辆的速度误差 0.33 → 0.36：约 6% 的运动车帧仍被判为静止，多是 1 m/s 左右的慢速帧。

复现（教师结果缓存后，换参数重跑只要几秒）：

```bash
python tools/dog/nusc_autolabel_eval.py --dataroot data/nuscenes_mini --scenes mini_val \
    --cache work_dirs/nusc_autolabel/teacher.pkl --out work_dirs/nusc_autolabel/report.md
```

## ④ 人工审核

自动标注不能直接拿来训练：漏标的物体会被当成背景，错标的类别和朝向会被模型学进去。分两级：

- **④a 按轨迹**（下面）：训练集至少做这一级，处理待审核的轨迹；
- **④b 逐框**（SUSTechPOINTS，见第 10 节）：验证集必须做，补漏标、改框。没被标为待审核的轨迹也会错：样例 bag 里一个分数 0.50 的"骑行者"其实是三脚架（[LABELING_TUTORIAL.md](LABELING_TUTORIAL.md) 第 5 节）。

### ④a 按轨迹

**看**：生成审核用的 MCAP，在 Foxglove 里打开。

```bash
python tools/dog/review_labels.py mcap --labels data/labels/rec_001/labels.pkl \
    --frames data/runs/rec_001/frames --results data/runs/rec_001/voxel/results.pkl \
    --out data/labels/rec_001/rec_001_review.mcap
```

| topic | 内容 |
|---|---|
| `/lidar` | 当前帧点云（`--all-sweeps` 画 5 帧拼接，文件大很多） |
| `/labels` | 标签框，按类别着色；待审核的轨迹为品红色粗框；插值补出来的框更淡；文字是轨迹号、类别（车带子类，如 `vehicle/truck`）、分数、速度 |
| `/teacher` | 教师原始框（灰色，达到上报阈值的），用来对照 |
| `/tf` | `world → det`。3D 面板显示系选 `world` 时，静止物体应该不动，拖尾（过去 3 s）是一个点 |
| `/log` | 每条待审核轨迹第一次出现的时间 |

**改**：在 `labels_tracks.csv` 的 `action` 列填动作，一行一条轨迹：

| 动作 | 含义 |
|---|---|
| `ok` | 确认无误，清除待审核标记 |
| `drop` | 删掉（误检、重复框、跟错的物体） |
| `vehicle` / `pedestrian` / `cyclist` | 改类别（改成别的类时车的子类清空；只是确认 `vehicle` 则保留子类） |
| `car` / `truck` / `bus` | 是车，并指定子类（表里 `subclass` 列是教师投票的结果，错了就改） |
| `flip` | 朝向转 180° |
| `static` | 物体没动：速度置 0，位置和朝向固定为整段的中位数 / 均值 |
| `ignore` | 看不清（点太少、被挡住一半，分不出是不是车）：训练时作忽略框，既不当物体也不当背景，评估时不计分 |

- 多个动作用 `+` 连起来，例如 `cyclist+flip`。
- `from` / `to` 列（秒，从录制开始算，对应表里的 `t_first` / `t_last`）把动作限制在一段时间内。例如轨迹在 3.2 s 后跳到了另一个物体上：这一行填 `drop`、`from` 填 3.2；再复制一行，`ok`、`to` 填 3.2。各行按从上到下的顺序执行。
- 除 `drop` 外，任何动作都会清除待审核标记。
- 表里 `train_as` 列是导出时这条轨迹的去向（训练类别名，或 `ignore`）；`length_m`、`height_m` 是框的长、高（中位数）。只训练车辆时用 `review_labels.py table --classes vehicle` 生成只有车的表，要定的（`train_as = ignore`）排在前面（第 13 节）。

**应用**：

```bash
python tools/dog/review_labels.py apply --labels data/labels/rec_001/labels.pkl \
    --edits data/labels/rec_001/labels_tracks.csv --out data/labels/rec_001/labels_reviewed.pkl
```

没审核过、仍标着待审核的轨迹，导出时默认作忽略框（`--review-policy ignore`）：既不当物体，也不当背景。以前是直接丢掉（`--skip-review`，仍可用）：丢掉的如果是真车，训练时就成了"这里没有车"的负样本。

按轨迹改不能**补**漏标的物体，也不能逐帧调框，这些在 ④b 里做（第 10 节）。

## ⑤ 导出数据集和 GT 采样库

```bash
python tools/dog/export_dataset.py \
    --recording data/runs/rec_001 data/labels/rec_001/labels_reviewed.pkl \
    --recording data/runs/rec_002 data/labels/rec_002/labels_sustech.pkl \
    --val-recordings rec_002 --out data/dog --gt-db
```

**忽略框**（`gt_ignore`）：没审核过的待审核轨迹（`--review-policy ignore`，默认；`drop` 丢掉，`keep` 当正样本）和审核时标了 `ignore` 的轨迹。训练时热图上照样画出它的高斯（于是周围不算负样本），但不给中心点和回归目标，所以网络在这里出不出框都不奖不罚；GT 采样库不收；评估时它既不算漏检，落在它上面的检测也不算误检（类似 KITTI 的 DontCare）。实现：`det3d/datasets/pipelines/preprocess.py`（`AssignLabel`）、`dog_perception/detection/evaluate.py`。

点云和框都从检测系转到虚拟 nuScenes 雷达系（`ModelFrame`），和预训练模型的 z 先验一致。输出 `data/dog/infos_{train,val}.pkl`、`lidar/` 和 GT 采样库，路径与 `configs/dog/dog_centerpoint_pp_xt32.py` 一致。

类别名用模型 head 的 5 个类：car / truck / bus / pedestrian / cyclist（取标签的 `subclass`）。只训练车辆的配置（`*_veh.py`，第 13 节）加 `--classes vehicle`：只导出车，按尺寸分 car / truck。没有子类的车（子类出现之前做的标签，或用旧的 3 类模型当教师）按尺寸补一个（长 ≥ 10 m 为 bus，长 ≥ 6.5 m 或高 ≥ 2.8 m 为 truck，否则 car），脚本会打印有多少个，最好重新跑一遍 `auto_label.py` 或在 ④ 里核对。旧的 3 类配置（`*_3cls.py`）要加 `--classes coarse`，类别名为 vehicle / pedestrian / cyclist。

## ⑥ head 权重迁移 + 训练

```bash
python tools/dog/convert_nusc_ckpt.py --src work_dirs/pretrained/nusc_pp.pth --dst work_dirs/dog_pp_init.pth
python tools/train.py configs/dog/dog_centerpoint_pp_xt32.py --work_dir data/train/dog_pp_v001
```

- 迁移：car ← car，truck ← truck（nuScenes task 1 的 truck 通道），bus ← bus（task 2 的 bus 通道），pedestrian ← pedestrian，cyclist ← bicycle；其余层原样。task 从 `--config` 读（默认 `configs/dog/dog_centerpoint_pp_xt32.py`，Voxel 的 head 布局相同），旧的 3 类配置传 `--config configs/dog/dog_centerpoint_pp_xt32_3cls.py`（vehicle ← car）。配置里 `load_from` 指向迁移后的权重。
- 配置：5 个 task（car、truck、bus、行人、骑行者各一个），输出时 car / truck / bus 合并为 vehicle，见第 12 节；20 个 epoch，one-cycle 最大学习率 5e-4，GT 采样（car 2、truck 2、bus 2、人 4、骑行 4；GT 库里没有的类别跳过），±45° 旋转、±5% 缩放、±2° 俯仰/横滚增强；另有 30% 的帧把历史 4 帧按随机的速度误差（σ 0.5 m/s，最多 1.5 m/s）和角速度误差（σ 3°/s，最多 10°/s）挪动（`SweepPoseJitter`，`det3d/datasets/pipelines/dog_aug.py`）：线上用 INS 位姿拼帧，急刹、急转时历史帧会错开 0.2–0.6 m、几度，训练帧用的是修正后的位姿，这样网络见过这种重影。当前帧和标注框不动，所以标注仍然对。
- 单卡（RTX 4090）直接运行；中断后用 `--resume_from <work_dir>/epoch_N.pth` 续训。多卡命令（`torch.distributed.launch`）在 torch 2.x 上没有验证过。没有 GPU 时同一条命令会在 CPU 上跑，只适合检查流程。
- 配置要点、日志怎么看、怎么选 epoch，见 [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) 第 4 节。

## ⑦ 评估

```bash
python tools/dist_test.py configs/dog/dog_centerpoint_pp_xt32.py --work_dir data/train/dog_pp_v001/eval \
    --checkpoint data/train/dog_pp_v001/latest.pth
```

- 这是模型级评估（网络本身，验证集 `data/dog/infos_val.pkl`）。决定能不能替换线上模型，要看流水线级对比：预训练模型和微调模型都用 `run_sequence.py` 在**同一批人工逐框审核过的验证录制**上跑，再用 `compare_detectors.py --labels <验证集 labels_sustech.pkl>` 逐类、按 0–10 / 10–20 / 20–40 m 分段比。
- 另外在 nuScenes mini_val 上查回归（`nusc_clip_eval.py`），在新 bag 上人眼看（`run_bag_mcap.py`）。
- 不要用教师自己生成的标签评估教师同架构的模型。
- `dist_test.py` 对 5 个 head 各出一行 AP，另外以 `merged/` 开头的几行是合并成 vehicle / pedestrian / cyclist、做完跨 head 去重之后的结果，也就是检测器实际输出的类别。
- 各工具都按权重自动识别配置（nuScenes 10 类 / 狗 5 个 head / 旧的狗 3 类 `*_3cls`），不用再手动指定。
- 四层评估的命令、指标含义和验收标准，见 [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) 第 5 节。

## ⑧ 部署到 Orin

`tools/dog/export_pillar_onnx.py --checkpoint <权重> --out deploy/<名字> --check <一帧 .npy>` 导出两段 ONNX 并与 PyTorch 逐段校验，再在 Orin 上用 `trtexec --fp16` 建引擎，见 [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) 第 6 节和 DOG_PIPELINE.md 第 8 节。

## ⑨ 迭代

从第二轮开始，用自己微调好的 Voxel 模型当教师重新标注（自训练）。教师越准，待审核的轨迹越少。

## 10. 标注工具：SUSTechPOINTS 还是 Xtreme1

按轨迹改（第 ④ 步）解决的是"自动标注大体对、个别轨迹要修"。验证集需要逐框核对、补漏标，就要一个 3D 标注工具。两个开源候选（2026-10 查看）：

| | [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS) | [Xtreme1](https://github.com/xtreme1-io/xtreme1) |
|---|---|---|
| 形态 | 轻量网页工具，本地起一个 Python 服务即可（也有 Docker） | 完整平台：Docker Compose 起 MySQL、Redis、MinIO、后端、前端 |
| 数据 | 文件夹形式：点云（.pcd / .bin）+ 每帧一个标注 JSON | 点云要求 .pcd，按规定的目录结构打包上传；结果按其格式导出 |
| 3D 框 | 9 自由度编辑、按点自动贴合、自动预测朝向、批量模式（一个物体连续多帧一起改）、物体 ID 跨帧保持 | LiDAR 与 LiDAR-相机融合标注，2D/3D 检测和 3D 分割 |
| 协作 | 单人用，没有任务分配和质检流程 | 用户、任务、本体（类别 / 属性）管理，质检，自带 OpenPCDet 预标注模型 |
| 维护 | 最后一次提交 2024-06，之后没有更新 | 持续更新（2026-09 仍有提交），LF AI & Data 项目 |
| 许可 | GPL-3.0（内部使用不受影响） | Apache-2.0 |

**建议先用 SUSTechPOINTS。** 我们的数据只有激光，没有相机；预标注来自自己的教师，不需要平台自带的模型；标注量是一两个人校正验证集、补漏标。SUSTechPOINTS 是纯文件格式，用 `tools/dog/sustechpoints_convert.py` 来回转换（见下文），轨迹号保留为物体 ID，批量模式正好对应"一条轨迹连续多帧一起改"。它的问题是已经两年多没有维护，遇到 bug 得自己修。

**以后换 Xtreme1 的时机**：标注的人多了，需要分配任务、质检、统计；或者要加相机做融合标注；或者想要一个长期有人维护的平台。代价是部署重，预标注要转成它的导入格式。

### SUSTechPOINTS 的导入导出

SUSTechPOINTS 的源码放在仓库里（`third_party/SUSTechPOINTS`，上游 2024-06 的最新版本），`git pull` 之后就能运行，也可以直接改。上游版本、去掉了哪些文件、本地改了什么，见其中的 `VENDORED.md`；以后再改它的代码，也请在那张表里记一笔（GPL-3.0 要求注明修改和日期）。

```bash
pip install cherrypy jinja2     # 只有它自带的自动标注按钮需要 TensorFlow

# 导出：标签 + 点云 -> third_party/SUSTechPOINTS/data/rec_001。--every 5 = 10 Hz 里取 2 Hz
python tools/dog/sustechpoints_convert.py export --labels data/labels/rec_001/labels_reviewed.pkl \
    --frames data/runs/rec_001/frames --scene rec_001 --every 5

# 启动标注工具，浏览器打开 http://127.0.0.1:8081
python tools/dog/sustechpoints_convert.py serve
#   等价于上游的 cd third_party/SUSTechPOINTS && python main.py（监听所有网卡）；serve 可用 --host / --port

# 标完导回来，得到和 labels.pkl 一样格式的文件，直接给 export_dataset.py
python tools/dog/sustechpoints_convert.py import --scene rec_001 --out data/labels/rec_001/labels_sustech.pkl
```

导出的场景在 `third_party/SUSTechPOINTS/data/` 下，被 .gitignore 忽略，不会提交到仓库（点云很大）。标好的结果以导回后的 `labels_sustech.pkl` 为准，另行保存。

转换规则（`dog_perception/sustechpoints.py`）：

- **坐标系**：点云和框都在每一帧自己的检测系（重力对齐），不需要标定文件，也不写 `ego_pose`（SUSTechPOINTS 的 ego_pose 是 NovAtel 的 UTM 格式，并写死了它自己车上的雷达安装）。z 整体下移 1.8 m（地面在 z ≈ −1.8）：SUSTechPOINTS 假定雷达离地 1.8 m，新建的框底放在 z = −1.8；不下移的话新框会埋在地下、贴合不到点。导回时再移回来（`dog_meta.json` 里的 `z_offset`）。
- **点云**：默认导出 5 帧拼接（模型的输入），远处稀疏的物体更容易看清；运动物体会拖影，框要放在拖影最前端（当前帧）。`--current-sweep` 只导当前帧。强度除以 255。
- **帧名**：时间戳的微秒数（16 位数字，按名字排序就是时间顺序）。`dog_meta.json` 保存每帧的时间戳、`T_world_det` 和自动标注的速度、分数，导入时要用，不要删。
- **框**：position 为框中心，scale 为（长、宽、高），rotation.z 为航向；导入时忽略 rotation.x / y。
- **类别**：车按子类导出为 Car / Truck / Bus（没有子类的车导出为 Car），行人为 Pedestrian，骑行者为 BicycleRider；待审核的轨迹 `obj_attr` 为 `review`，在工具里会显示出来。导入时按 `TYPE_TO_SUBCLASS` 映射：Car、Van、PoliceCar、TourCar 为 car；Truck、ConcreteTruck 和工程机械（Excavator、ForkLift、Crane、RoadRoller、Bulldozer）为 truck（皮卡没有单独的类型，选 Truck，和 nuScenes 一致）；Bus、Tram、LongVehicle 为 bus（和 nuScenes 一样，工程车归 truck，挂车这类长车归 bus）；三者都是 vehicle。Pedestrian、RoadWorker、Child 为 pedestrian；BicycleRider、MotorcyleRider、ScooterRider 以及停着的 Bicycle、Motorcycle、Scooter 为 cyclist（和 nuScenes 的映射一致），三轮车 Trimotorcycle、FreightTricycle 也是 cyclist（小三轮车、代步车不算车辆，2026-10-21）。其他类型（Cone、TrafficBarrier 等）跳过并计数；`--type-map` 可以补充或改写。**标车的时候把类型选对**（Car / Truck / Bus），它决定训练哪个 head。
- **轨迹号**：数字 `obj_id`（自动标注的轨迹）原样保留；工具里新建、不是数字的 ID 分配新号。
- **速度**：框没被挪动（中心偏移 ≤ 5 cm）就保留自动标注的速度，静止物体的处理不受影响；挪动过或新加的框，用同一 `obj_id` 在世界系的中心差分（±0.5 s）重算，低于 0.2 m/s 置 0。所以要改一条轨迹的位置时，用批量模式把整条轨迹一起挪；只挪其中一帧，那几帧的速度会算出一个跳变。

工具本身要注意的几点（详细操作见 [LABELING_TUTORIAL.md](LABELING_TUTORIAL.md) 第 5 节）：

- **自动保存**：这一版默认打开自动保存（`config.js` 里 `autoSave = true`），每次修改约 0.5 s 后写回 `label/*.json`，Ctrl+S 也能手动存。它的中文 README 说"暂不支持自动保存"，已经过时。想放弃全部修改，重新 export 覆盖这个场景即可。
- **新建框**（右键 → New → 类别，或 Ctrl + 拖框）：会从鼠标下的点自动生长、贴合，朝向由服务端 `/predict_rotation` 给出。没装 TensorFlow 时，仓库里的版本用点的主方向（PCA）代替，朝向可能差 180°，按 `g` 翻转；贴合时可能把旁边的东西也包进来（试验中一个骑行者的框高到了 3.1 m），要在侧视图里调。
- 批量窗口里的 Auto / Auto (no rotation) 需要 TensorFlow 和它的模型文件（上游没有提供），没有时返回 503；Interpolate、Finalize 等不受影响。
- 右键菜单打开时按 Esc 会触发它自己的一个脚本错误（不影响结果），点空白处关闭菜单即可。

在运动样例 bag 上用真实的 SUSTechPOINTS（2024-06 的最新版本）验证过：导出的 27 帧场景能正常加载（一帧 12 个框、18.7 万个点，网页没有报错，待审核的框显示 `review`）；不做修改直接导回，框、类别、速度与原标签完全一致；在工具里把一辆车挪 1 m 并改成 Van、新加一个行人，再用工具自己的保存功能存盘，导回后车仍是 vehicle、位置偏移 1.00 m，新行人成为一条新轨迹。

## 11. 在样例 bag 上跑通的结果

完整的操作过程（两段样例 bag、按轨迹 + 逐框审核、导回、生成训练数据）见 [LABELING_TUTORIAL.md](LABELING_TUTORIAL.md)。下表是最早用运动 bag 做的冒烟测试。

`samples/rosbag2_2026_09_21-16_41_27_moving`（27 帧，狗在走），教师用 nuScenes 预训练 Pillar（CPU）：

| 步骤 | 结果 |
|---|---|
| ① bag → 序列 | 27 帧点云、310 条 IMU、310 条 INS；RTK 固定解（pos_type 56）；机身离地 0.72 m |
| ② 教师推理 | 27 帧，网络中位数 511 ms/帧（CPU） |
| ③ 自动标注 | 327 个框，17 条轨迹，13 条待审核 |
| ④ 审核 | MCAP 121 条消息、14 MB；`drop`、部分 `drop`、`static`、改类别 + `flip`、`ok` 都按预期生效 |
| ⑤ 导出 + GT 库 | vehicle 190 / cyclist 90 / pedestrian 47 |
| ⑥ head 迁移 | 295 个张量全部加载，无形状不匹配 |
| ⑥ 训练（CPU，6 步） | loss 9.5 → 5.1；同样几帧上，迁移起训 5.8，随机初始化 18.1 |

17 条轨迹里 13 条平均分低于 0.5、被标为待审核：预训练模型在 32 线数据上的分数普遍偏低。换成 Voxel 教师、再做自训练后，这个比例应该会下降。

## 12. 车辆分成 car / truck / bus 三个 head

**结论**：大车（卡车、公交）的框更准，数据少的时候召回也更稳；小车和合并后的 vehicle mAP 不变；代价是计算量多 12–14%。下面是做法、原因、实验和代价。

**做法。** 检测头有 5 个 task：car、truck、bus、pedestrian、cyclist，各 1 类（`configs/dog/dog_centerpoint_{pp,voxel}_xt32.py`）。输出时 car / truck / bus 合并成 vehicle（`SUBCLASS_TO_CLASS`），再按 2 m 做一次类内去重；跟踪、预测、评估和 ROS 看到的仍然是 vehicle / pedestrian / cyclist 三类。子类按 nuScenes head 的分组来分，也就是按尺寸：

| 子类 | nuScenes 类别 | SUSTechPOINTS 类型 | 典型长度 | 迁移自 nuScenes |
|---|---|---|---|---|
| car | car | Car、Van、PoliceCar、TourCar | 4–5 m | task 0（car） |
| truck | truck、construction_vehicle | Truck、ConcreteTruck、Excavator、ForkLift、Crane、RoadRoller、Bulldozer | 6–10 m | task 1 的 truck 通道 |
| bus | bus、trailer | Bus、Tram、LongVehicle | 10–18 m | task 2 的 bus 通道 |

子类沿着整条数据链传下去：
1. 教师：nuScenes 模型本来就有这几个 head，检测结果里带 `subclasses`；
2. 自动标注：每条轨迹在车的观测里按分数加权投票，写进 `subclass` 字段（③）；
3. 审核：表里多一列 `subclass`，错了用 `car` / `truck` / `bus` 动作改（④a）；SUSTechPOINTS 里导出为 Car / Truck / Bus，改类型即改子类（第 10 节）；
4. 导出：`export_dataset.py` 把子类写成训练类别名；没有子类的车按尺寸补，并打印个数（⑤）。

在运动样例 bag 上重新跑教师（nuScenes Pillar）和自动标注：17 条轨迹、327 个框与之前完全一致，车的轨迹都带上了子类。7.2 m 长、3.2 m 高的 T12 为 truck，4.5 m 的 T7、T14 为 car。另有两条 2.2–2.5 m 长、2.5 m 高的轨迹（T3、T18）也由 truck head 给出（nuScenes 的工程车也归在 truck）。细看它们都是一面高墙墙脚的东西，不是车，教程里在 SUSTechPOINTS 中删掉了（[LABELING_TUTORIAL.md](LABELING_TUTORIAL.md) 第 5.3 节）。所以审核时要看一眼 `subclass` 列：尺寸和子类对不上，往往说明根本不是车；子类拿不准时保留教师的。

**为什么更准：**
1. **热启动不丢东西。** 旧布局只有一个 vehicle head，只能从 nuScenes 的 car head 继承，而 car head 训练时把卡车、公交当负样本。新布局三个 head 各自继承，起点和 nuScenes 模型一样（下表"起点"两行）。
2. **每个 head 只管一个尺寸段。** CenterPoint 一个 task 内共用回归分支，尺寸用 log 后的 L1 回归。4.5 m 的轿车和 12 m 的公交放在一个 head 里，尺寸分布是双峰的。nuScenes 按尺寸分 task 就是这个原因。
3. **各 head 有自己的 NMS 半径**：car 2 m、truck 3.5 m、bus 3.2 m。
4. **狗的数据里确实有大车。** 样例 `rec_static` 的停车场里就有 6 辆长约 12 m、高约 4 m 的车（[LABELING_TUTORIAL.md](LABELING_TUTORIAL.md) 第 7 节）。

**实验**（nuScenes mini，Pillar，2026-10-14，CPU）。
- **数据**：训练用 mini_train 的 8 个场景（323 个关键帧），验证用 mini_val 的 2 个场景（81 帧，预训练模型没见过）；10 帧叠加，±40.8 m。
- **对比的两种 head**：单 vehicle head（旧布局，从 car head 迁移）和 car / truck / bus 三个 head（新布局）。两者都从 nuScenes Pillar 权重迁移，配置相同，只差 head；训练 6 个 epoch，关掉 GT 采样和按类重采样，保证看到的数据完全一样。
- **评估方式**：走检测器封装（合并成三类、2 m 去重、几何过滤），只算 40 m 内、至少有 1 个点的真值。车的真值共 1816 个框：car 1710，truck 77（4 辆），bus 29（1 辆）。召回、尺寸误差（1 − 对齐后的 IoU）和朝向误差（rad）按车的上报阈值 0.35、2 m 内匹配计算。

| | vehicle mAP | car 召回 / 尺寸误差 / 朝向误差 | truck 召回 / 尺寸 / 朝向 | bus 召回 / 尺寸 / 朝向 |
|---|---|---|---|---|
| 起点：单 vehicle head（= car head） | 0.718 | 0.739 / 0.168 / 0.26 | 0.338 / 0.220 / 0.03 | 0.000 / – / – |
| 起点：三个 head | 0.746 | 0.740 / 0.169 / 0.26 | 0.935 / 0.198 / 0.22 | 1.000 / 0.162 / 0.02 |
| 微调后：单 vehicle head | 0.736 | 0.725 / 0.177 / 0.31 | 0.935 / 0.229 / 0.05 | 0.966 / 0.347 / 0.19 |
| 微调后：三个 head | **0.738** | 0.733 / 0.179 / 0.30 | 0.922 / **0.194** / 0.44 | **1.000 / 0.113 / 0.02** |

结论：
- **小车和总分一样。** 微调后合并的 vehicle mAP 是 0.738 对 0.736，差别在噪声以内：验证集里 94% 是小车，两种布局的小车 head 都从 car head 来。作为噪声的量级参考，两次训练行人 / 骑行者 mAP 也差了 0.016 / 0.019，而这两类的 head 两边完全一样。
- **大车的框更准。** 公交尺寸误差 0.11 对 0.35：单 head 给 12 m 的公交平均画 13.4 m。公交朝向误差 0.02 对 0.19；卡车尺寸误差 0.19 对 0.23。微调后两者的大车召回都在 0.92–1.0。
- **起点差别很大。** 不微调时，单 head 只找回 34% 的卡车、0% 的公交；三个 head 和 nuScenes 模型一样（vehicle mAP 0.746 对 0.718）。mini_train 里有 46 辆大车，6 个 epoch 就把单 head 的召回补回来了。狗的数据里大车少得多，补不回来的风险更大，这时三个 head 的优势最明显。
- **唯一变差的是一辆 5 m 长的小卡车。** car 和 truck 两个 head 都会出框（30 帧里 22 帧由 car head 出），其中 9 帧朝向反了 180°，所以 truck 的平均朝向误差是 0.44。介于两类之间的车（皮卡、小厢式货车）会在两个 head 之间分票。按 nuScenes 的定义，轿车、SUV、面包车是 car，皮卡和各种货车是 truck；标注时照这个来，点太稀、分不清时保留教师给的子类。运动中的 180° 翻转，跟踪和自动标注会按运动方向纠正。
- **样本少**：验证集只有 2 个场景、4 辆卡车、1 辆公交，大车的数字只说明方向，要在自己的数据上用 `compare_detectors.py` 再确认。

**代价：**
- **计算量**：多 2 个 task。Pillar 从 34.1 增加到 38.9 GMAC（+14%），Voxel 从 67.5 增加到 75.9 GMAC（+12%），参数多 0.46 M。Orin 上 head 在 TensorRT 引擎里，延迟要实测。
- **标注**：车的类型要选对（Car / Truck / Bus）。教师会先给出子类，审核时只需要改错的。
- **样本少**：卡车、公交的框少，对应的 head 主要靠迁移来的权重和 GT 采样（truck 2、bus 2）。如果录制里一辆公交都没有，bus head 在微调中只见到负样本，会越来越不出框。这种情况下旧布局同样检测不到公交（它的 vehicle head 来自 car head），但要知道新布局的优势只在有大车样本时才能保持，录数据时要包含一些有大车的场景。

**旧布局**：配置保留为 `*_3cls.py`（预设名 `dog_pillar_3cls` / `dog_voxel_3cls`）。迁移时加 `--config configs/dog/dog_centerpoint_pp_xt32_3cls.py`，导出加 `--classes coarse`。已经按旧布局训练的权重照常能用，各工具按权重里的 task 数自动识别。

## 13. 只检车辆的版本：car / truck 两个 head（2026-10）

**要求**：行人、骑行者暂时不检，接口保留；车辆必须检出，而且尽量准。0921 的录制里大货车不少，公交基本没有，教师（nuScenes Voxel）却把大货车多半判成了公交。

### 0921 自动标注里的车

5.11 的自动标注，35 个场景，每条轨迹取框尺寸的中位数：

| 教师给的子类 | 轨迹 | 框 | 长度 p10 / 中位 / p90 (m) | 高度中位 (m) | 平均分中位 | 待审核 |
|---|---|---|---|---|---|---|
| car | 199 | 19,161 | 4.4 / 4.7 / 5.0 | 1.85 | 0.59 | 30% |
| truck | 132 | 12,874 | 2.7 / 5.9 / 12.4 | 2.68 | 0.42 | 70% |
| bus | 46 | 3,498 | 7.3 / 12.4 / 16.0 | 4.02 | 0.36 | 91% |

- **"公交"其实是大货车。** 中位长 12.4 m、高 4.0 m、宽 3.0 m。教师的 truck 里 ≥ 10 m 的还有 26 条，也就是说同样的大货车，教师有时给 truck，有时给 bus。按教师的子类分 head 训练，两个 head 会拿到互相矛盾的目标。
- **小车很干净。** 长 4.4–5.4 m，高全部低于 2.5 m。所有 ≥ 6.5 m 的车都高于 2.5 m。
- **教师的 truck 里混着不像货车的东西。** 45 条长不到 5 m，其中 28 条长 2.4–3.3 m、宽 1.4–1.7 m、高约 2.2 m，几乎都不动：可能是电动三轮车、岗亭之类，要人看。

### 做法

1. **head 只留 car、truck**（`configs/dog/dog_centerpoint_{pp,voxel}_xt32_veh.py`，预设 `dog_pillar_veh` / `dog_voxel_veh`，各工具按权重里的 task 数自动识别）。bus head 没有数据，留着只会在微调里学成"不出框"；行人、骑行者的 head 去掉。两个 head 分别从 nuScenes 的 car、truck head 迁移：在 0921 上，教师 truck head 给 ≥ 10 m 货车的平均分中位是 0.53，bus head 给它判成公交的那些只有 0.36。少了 3 个 task 的 head，按第 12 节的测量（每个 task 的 head：Pillar 约 2.4 GMAC，Voxel 约 4.2 GMAC）估算，Pillar 约 31.7 GMAC（5 个 head 时 38.9），Voxel 约 63.3 GMAC（75.9）。
2. **按尺寸分 head，不再用教师的子类**：长 ≥ 5.5 m 或高 ≥ 2.5 m 为 truck，否则为 car（`dog_perception/detection/boxes.py` 的 `vehicle_head`）。公交、挂车、工程车都归 truck。同样大小的车永远进同一个 head，审核时也不用再改 car / truck / bus。
3. **拿不准的轨迹作忽略框**（第 ⑤ 步，`gt_ignore`）：既不当正样本，也不当负样本，评估时不计分。哪些算拿不准：
   - 还标着待审核（平均分 < 0.5）、没人审过的轨迹；
   - 没人审过、教师子类和尺寸对不上的（教师说 truck / bus，尺寸却是小车）；
   - 审核时填了 `ignore` 的（SUSTechPOINTS 里 attribute 写 `ignore`）。

   以前是把待审核轨迹直接丢掉（`--skip-review`）。丢掉的如果是真车，训练时就成了"这里没有车"，正好和"车必须检出"相反：0921 的大货车 70–91% 都是待审核。
4. **类平衡采样补回无车帧**：类平衡采样让 car、truck 各占一半，但会把不含车的帧全部排除，而 0921 训练集有一半的帧没有车。配置里 `empty_frames=0.25` 再补回相当于四分之一的无车帧，让网络也见到只有墙、树、行人的场景。
5. **GT 采样**：car 2、truck 3（货车少）。其余（20 个 epoch、学习率、旋转缩放、`SweepPoseJitter`）和 5 个 head 的配置一样。
6. **接口不变**：检测器输出的类别仍是 vehicle / pedestrian / cyclist，子类 car / truck 跟着输出。跟踪、预测、评估和 ROS 都不用改，只是不会出现行人和骑行者。以后要加回来：
   - `labels.pkl` 里行人、骑行者的自动标注一直保留着；
   - 配置里加 pedestrian、cyclist 两个 task，`convert_nusc_ckpt.py` 按配置里的 task 从 nuScenes 迁移；
   - 审核行人和骑行者，再训练。

### 按这个规则，0921 未审核的自动标注能进训练的车

按场景划分（`data/survey/split/split.csv`），跳过位姿不可靠的帧：

| | car 正样本 | car 忽略 | truck 正样本 | truck 忽略 |
|---|---|---|---|---|
| 训练（25 个场景，7,311 帧） | 121 条轨迹 / 12,902 框 | 79 / 4,422 | 13 / 1,668 | 56 / 4,084 |
| 验证（10 个场景，2,874 帧） | 18 / 3,043 | 35 / 3,055 | 18 / 3,215 | 37 / 3,144 |

其中 4,261 个框是因为"教师子类和尺寸对不上"被忽略的。不审核的话，训练集里 truck 的正样本只有 13 条轨迹，truck head 基本还是迁移来的样子。所以：

### 审核只看车

要人定的轨迹（表里 `train_as = ignore`）：训练集 135 条，验证集 72 条。原来所有类别要审 718 条。

- **按轨迹**（训练集和验证集都做）：
  ```bash
  python tools/dog/review_labels.py table --classes vehicle --labels data/labels/$r/labels.pkl \
      --out data/labels/$r/labels_tracks_vehicle.csv
  python tools/dog/review_labels.py mcap --vehicles-only --labels data/labels/$r/labels.pkl \
      --frames data/runs/$r/frames --results data/runs/$r/voxel/results.pkl
  ```
  表里只有车，要定的排在前面，带 `length_m`、`height_m`。每行：是车填 `ok`，不是车填 `drop`，看不清填 `ignore`（留空效果一样）。朝向反了加 `+flip`，停着的车加 `+static`。不用改 car / truck / bus，导出按尺寸分。
- **逐框**（只做验证集）：`sustechpoints_convert.py export --vehicles-only ... --every 5`，补漏标的车、调框。导回的 `labels_sustech.pkl` 只有车；以后做行人、骑行者时，验证集要从 `labels.pkl` 重新导出逐框审核。
- **规则（2026-10-21 定）**：小三轮车、老年代步车不算车辆，算骑行者，动作填 `cyclist`；单独停着的挂车厢算车辆（`ok`，按尺寸进 truck）。SUSTechPOINTS 里三轮车选 Trimotorcycle / FreightTricycle（导回为 cyclist），挂车厢选 Truck 或 LongVehicle。

### 训练和评估

服务器上的步骤见 [SERVER_TASKS.md](SERVER_TASKS.md)：T2–T4 先用未审核的自动标注跑通一版（v0），H1 人工审核后 T5–T7 正式训练（v1），T8 导出 ONNX。要点：

```bash
python tools/dog/convert_nusc_ckpt.py --src work_dirs/pretrained/nusc_voxel.pth --dst work_dirs/dog_voxel_veh_init.pth \
    --config configs/dog/dog_centerpoint_voxel_xt32_veh.py
python tools/dog/export_dataset.py $REC --val-recordings $VAL --classes vehicle --out data/dog_veh --gt-db
python tools/train.py configs/dog/dog_centerpoint_voxel_xt32_veh.py --work_dir data/train/dog_voxel_veh_v0
python tools/dist_test.py configs/dog/dog_centerpoint_voxel_xt32_veh.py --work_dir data/train/dog_voxel_veh_v0/eval \
    --checkpoint data/train/dog_voxel_veh_v0/latest.pth
python tools/dog/compare_detectors.py --classes vehicle --labels <验证场景的标签> --results teacher=... v0=...
```

- 迁移后、微调前，两个 head 的输出和 nuScenes 模型的 car、truck head 一样：0921_164127_s005 的一帧上（Pillar，分数 ≥ 0.2），车辆模型的 2 个框在 nuScenes 模型里都有、中心完全重合；nuScenes 模型多出的 1 个来自 construction_vehicle 通道（迁移只取 truck 通道）。
- 评估只算车（`merged/vehicle`，`compare_detectors.py --classes vehicle`），忽略框不计分。重点看召回（分距离段 AP、上报阈值处的召回），其次是框（ATE、ASE、AOE）。上报阈值（现在 vehicle 0.35）要按验证集重新定。
- 验证集没有逐框审核时，标签就是教师给的，对教师有利，只能说明训练没跑坏；要不要替换线上模型，看审核过的验证集。

## 14. 遮挡、待审核和静止判断（2026-10-22）

### 遮挡和远处：按框里的点数决定

A 车被路过的车挡住时，跟踪会延续 A 的轨迹，自动标注也会把 0.5 s 以内的空档插值补上。但单帧检测器在那一帧看不到 A，硬要它在那里出框，就是在教它猜。远处只有几个点的车也一样。

现在按框里的点数决定（`dog_perception/visibility.py`）：

| 框里的点 | 训练 | 评估 |
|---|---|---|
| 模型输入的 5 帧里一个点都没有 | 不要这个框（那里就是背景） | 不算真值 |
| 当前帧不到 5 个点（车；人、骑行 3 个） | 忽略框：不当正样本，也不当负样本 | 不计分，检出也不算误检 |
| 其余 | 正样本 | 正常计分 |

- 点数在导出时对着导出的那一帧数（`export_dataset.py`），审核时挪过的框也按新位置数。框的长、宽各放宽 0.25 m，框底以上 0.2 m 以内的点算地面，不计。
- `auto_label.py` 把点数写进 `labels.pkl`，`compare_detectors.py` 用同样的规则评估。SUSTechPOINTS 导回时加 `--frames data/runs/<场景>/frames` 也会写上。
- 轨迹表多一列 `few_points`：这条轨迹有多少帧点数不够。这些帧不管审核填什么，都按上表处理，审核时不用管。
- **只影响训练目标，不影响在线跟踪。** 线上跟踪照样在遮挡时延续轨迹，狗仍然记得那里有车。

### 待审核：平均分一般，但有几帧很确定

以前平均分 < 0.5 就要审。远处的车大部分时间只有 0.3–0.4 分，等狗走近了才有几帧 0.6 以上；这种车其实可以确定。现在的规则是：**平均分 < 0.5，并且 ≥ 0.6 分的检测不到 5 帧**，才标为待审核（`AutoLabelConfig.confident_score` / `confident_frames`）。误检轨迹偶尔也会有一帧高分，但很少连续好几帧，所以要求 5 帧，而不是只看最高分。

0.6 和 5 是先定的值，要在服务器上校准：

```bash
python tools/dog/review_labels.py confidence --vehicles-only --labels data/labels/0921_*/labels.pkl \
    --spot-check data/reports/review_spot_check.csv
```

这条命令打印不同阈值（0.5 / 0.6 / 0.7 分 × 3 / 5 / 10 帧）下还剩多少条待审核轨迹、多少个框，并列出按 (0.6, 5) 不再需要审核的轨迹。从中抽 20 条，在 Foxglove 里看是不是真车：全是真车就用这个规则；有不是车的，就提高分数或帧数。

另外两点，用来减少审核量：

- **短轨迹可以不审。** 现在要人定的 201 条车辆轨迹里，不到 20 帧的有 60 条（30%），但只有 628 个框（4%）。不审的轨迹仍是忽略框，不会被当成背景，损失很小。审核时按 `frames` 从多到少审，审不完的留空即可。
- **"教师子类和尺寸对不上"仍然要审。** 只因为这条被忽略的 13 条轨迹里，8 条是长 2.5–3.1 m、高约 2.3 m、平均分 0.5–0.68 的"truck"，很可能是电箱、岗亭或三轮车。分数高，但不一定是车。

### 静止 / 运动

狗遇到运动的车要停，遇到静止的车不停，所以静止判断直接决定狗的行为。

**标注（离线）。** 原来某一帧只要轨迹速度和教师速度头都过阈值就算运动，0921 上有运动帧的 104 条车辆轨迹里，45 条只有短于 1 s 的零星运动帧。这些多半是停着的车：狗走过长货车时框沿车身滑动，或者是轨迹最后几帧。这些运动帧带来两个问题：

- 轨迹没能固定到一个位置，框跟着漂；
- 那几帧的速度标签不是 0。

现在去掉这种毛刺（`autolabel._clean_moving`）：

- 运动中短于 0.5 s 的停顿，算运动；
- 短于 1 s 的运动段，算静止，除非这一段占了轨迹的一半以上（例如车只在视野边上出现了一下）。

**在线（跟踪器）。** 以前只输出速度，没有状态。现在每条轨迹有 `motion`（`TrackState.motion`，ROS 的轨迹文字和 mcap 里也显示），有三种状态：

| 状态 | 什么时候进入 | 狗怎么做（建议） |
|---|---|---|
| `moving` | 最近 5 次检测里有 3 次"快"：位移速度和速度头都 > 0.7 m/s（行人 0.5） | 停 |
| `stopped` | 运动之后，1 s 内没有一次"快"，并且位移速度 < 0.3 m/s | 仍按运动处理：刚停下的车（让行、路口）可能马上再走 |
| `static` | 一开始就是；或者 `stopped` 持续 10 s | 当障碍物绕开，不停 |

位移速度是这条轨迹最近 1 s 检测框中心的直线拟合斜率（不足 0.3 s 时不算），速度头是检测器每帧给的速度。

- **为什么两个信号都要过阈值**，和标注的道理一样：
  - 停着的车，框会随视角漂移，位移速度会有零点几 m/s；框沿长车滑动时更大；
  - 速度头只看一帧（5 个扫描拼起来），位姿一抖就跳。
  - 测试里构造了"框滑 1.2 m"和"速度头读 1.2 m/s 持续 0.5 s"两种情况，只看任何一个都会误判为运动，两个都要求就不会（`tests/dog/test_tracking.py`）。
- **为什么不用 IMM 速度**：IMM 是用第一帧的速度头初始化的，开头几帧和速度头其实是同一个信号。运动样例 bag 开头，一辆停着的车速度头读到 12、8、5 m/s，IMM 也跟着读到 12、10、8 m/s，第 3 帧就报了运动。
- **为什么要有 `stopped`**：速度头逐帧噪声大，如果要求"连续 1 s 每帧都慢"，在噪声下几乎等不到。所以改成：1 s 内没有一帧"快"，并且位移速度也慢。
- **延迟**（测试里的仿真，中心噪声 0.15 m）：
  - 从静止起步到 3 m/s，0.4 s 报运动；
  - 第一次看到时就在开的车，0.5 s 报运动；
  - 停下后 0.9 s 报 `stopped`。
- **运动样例 bag**（27 帧，教师为预训练 Pillar，三类都算）：标注里静止的 16 条轨迹，被在线报过运动的：

  | 版本 | 报过运动的轨迹 | 静止帧里报成运动的比例 |
  |---|---|---|
  | 用 IMM 速度 | 4 条 | 12.3% |
  | 用位移速度 | 2 条 | 3.5% |

  剩下的 2 条来自 1.8–2.3 s 的一次位姿跳变：几个静止物体在世界系里一起偏了 0.4–0.7 m，速度头也读到 1.5–4.6 m/s。两个信号都被骗了，运动规则改不了，只能靠位姿。服务器上用的是离线修正过的位姿，线上用 INS，跳变会更多，所以线上还要用线上位姿再评一次。
- **阈值在服务器上调**：`tools/dog/motion_eval.py` 在验证场景上重跑跟踪，逐帧和标注的 `moving` 比，扫 `--on-speed`、`--on-frames`。给出：
  - 停着的车有多少被报过运动（每小时几次）；
  - 开始运动的车多久被报出来（中位数 / p90），漏了几段。

  选法：先保证漏报为 0、延迟 p90 不超过 0.5 s，再让停着的车尽量少被报运动。


### 服务器上同一天的改动（commit 15fdedc）

- **一辆长货车只开一条轨迹**：长货车常被检成前后两个框，以前第二个框会开一条新轨迹。现在，一个没匹配上的高分框，如果和已有的同类轨迹"一个的中心落在另一个框里"，就不开新轨迹（`tracking/tracker.py` 的 `_second_box_on_track`）。对小车和行人基本没有影响：中心落在对方框里时，距离已经小于关联门限，本来就会关联上。只对长于约 6 m 的车起作用。
- **教师的 bus、trailer head 记为 truck 子类**（`NUSC_TO_SUBCLASS`）。这只影响 5 个 head 的旧布局；车辆版本按尺寸分 head，不受影响。0921 是用旧的 `results.pkl` 重做的标注，子类还是推理时记下的，所以标签里还有 139 个 bus 框。
