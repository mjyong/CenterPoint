# 用自采 bag 微调检测器：提取 → 自动标注 → 审核 → 训练

从 kygi665 录制（Hesai XT32 + RTK 组合导航）到微调后的 Pillar 模型的完整流程。所有命令在仓库根目录执行，设置 `PYTHONPATH=.`。通用 LIO 里程计的写法见 [DOG_PIPELINE.md](DOG_PIPELINE.md) 第 3 节。

```
bag ─① rosbag_to_sequence ─► 序列 ─② run_sequence --save-frames ─► frames/*.npy（模型输入）+ 教师结果
   ─③ auto_label ─► labels.pkl + labels_tracks.csv ─④ review_labels（Foxglove 里看，按轨迹改）─► labels_reviewed.pkl
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

## ② 预处理 + 教师推理 + 保存帧

```bash
python tools/dog/run_sequence.py --seq data/seq/rec_001 --detectors voxel --tta \
    --ckpt-voxel work_dirs/pretrained/nusc_voxel.pth --save-frames --out work_dirs/rec_001
```

- `work_dirs/rec_001/frames/<stamp>.npy` 是预处理后的 5 帧拼接点云（检测系），和狗上线时模型的输入完全一样。训练直接用它，训练和部署走的是同一套预处理。
- 教师用 Voxel + 双翻转 TTA。nuScenes Voxel 权重在仓库里用 Git LFS 存（`work_dirs/pretrained/nusc_voxel.pth`，103 MB），克隆后要 `git lfs install && git lfs pull` 才是真文件，否则只是 134 字节的指针。
- Voxel 必须在 GPU 上跑：CPU 上 spconv 直接报错（`cpu don't support act and bias`）。
- 没有 GPU 时可以先用 Pillar 试跑：`--detectors pillar --ckpt-pillar work_dirs/pretrained/nusc_pp.pth`。

## ③ 自动标注

```bash
python tools/dog/auto_label.py --results work_dirs/rec_001/voxel/results.pkl --out data/labels/rec_001/labels.pkl
```

对整段录制离线跟踪，再逐条轨迹非因果精修（`dog_perception/autolabel.py`）：

- 类别按分数加权投票；尺寸和 z 取加权中位数；
- 中心 RTS 平滑，0.5 s 以内的漏检插值补上；
- **静止 / 运动**：某一帧只有在轨迹速度（±0.5 s 差分）> 0.4 m/s **并且**教师速度头 > 0.15 m/s 时才算运动，否则速度为 0。从头到尾都没动的轨迹，如果 90% 的框离中位数不超过车 1.0 m / 人 0.5 m / 骑行 1.5 m，位置固定为加权中位数；散得更开说明轨迹漂了或跟错了物体，保留平滑后的路径；
- 朝向：运动快（> 1 m/s）时取运动方向；否则取框朝向的环形均值，180° 歧义用运动方向消解，从没动过就用分数最高的那个框；
- 太短（< 5 次观测）或平均分太低（< 0.3）的轨迹丢掉；平均分 < 0.5 的标为待审核。

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

自动标注不能直接拿来训练：漏标的物体会被当成背景，错标的类别和朝向会被模型学进去。验证集要逐框检查；训练集至少过一遍待审核的轨迹。

**看**：生成审核用的 MCAP，在 Foxglove 里打开。

```bash
python tools/dog/review_labels.py mcap --labels data/labels/rec_001/labels.pkl \
    --frames work_dirs/rec_001/frames --results work_dirs/rec_001/voxel/results.pkl \
    --out data/labels/rec_001/review.mcap
```

| topic | 内容 |
|---|---|
| `/lidar` | 当前帧点云（`--all-sweeps` 画 5 帧拼接，文件大很多） |
| `/labels` | 标签框，按类别着色；待审核的轨迹为品红色粗框；插值补出来的框更淡；文字是轨迹号、类别、分数、速度 |
| `/teacher` | 教师原始框（灰色，达到上报阈值的），用来对照 |
| `/tf` | `world → det`。3D 面板显示系选 `world` 时，静止物体应该不动，拖尾（过去 3 s）是一个点 |
| `/log` | 每条待审核轨迹第一次出现的时间 |

**改**：在 `labels_tracks.csv` 的 `action` 列填动作，一行一条轨迹：

| 动作 | 含义 |
|---|---|
| `ok` | 确认无误，清除待审核标记 |
| `drop` | 删掉（误检、重复框、跟错的物体） |
| `vehicle` / `pedestrian` / `cyclist` | 改类别 |
| `flip` | 朝向转 180° |
| `static` | 物体没动：速度置 0，位置和朝向固定为整段的中位数 / 均值 |

- 多个动作用 `+` 连起来，例如 `cyclist+flip`。
- `from` / `to` 列（秒，从录制开始算，对应表里的 `t_first` / `t_last`）把动作限制在一段时间内。例如轨迹在 3.2 s 后跳到了另一个物体上：这一行填 `drop`、`from` 填 3.2；再复制一行，`ok`、`to` 填 3.2。各行按从上到下的顺序执行。
- 除 `drop` 外，任何动作都会清除待审核标记。

**应用**：

```bash
python tools/dog/review_labels.py apply --labels data/labels/rec_001/labels.pkl \
    --edits data/labels/rec_001/labels_tracks.csv --out data/labels/rec_001/labels_reviewed.pkl
```

导出时加 `--skip-review`，没审核过的待审核轨迹就不会进训练集。

按轨迹改不能**补**漏标的物体，也不能逐帧调框。验证集需要逐框标注，见第 10 节的标注工具。

## ⑤ 导出数据集和 GT 采样库

```bash
python tools/dog/export_dataset.py \
    --recording work_dirs/rec_001 data/labels/rec_001/labels_reviewed.pkl \
    --recording work_dirs/rec_002 data/labels/rec_002/labels_reviewed.pkl \
    --val-recordings rec_002 --out data/dog --gt-db --skip-review
```

点云和框都从检测系转到虚拟 nuScenes 雷达系（`ModelFrame`），和预训练模型的 z 先验一致。输出 `data/dog/infos_{train,val}.pkl`、`lidar/` 和 GT 采样库，路径与 `configs/dog/dog_centerpoint_pp_xt32.py` 一致。

## ⑥ head 权重迁移 + 训练

```bash
python tools/dog/convert_nusc_ckpt.py --src work_dirs/pretrained/nusc_pp.pth --dst work_dirs/dog_pp_init.pth
python tools/train.py configs/dog/dog_centerpoint_pp_xt32.py
```

- 迁移：vehicle ← car，pedestrian ← pedestrian，cyclist ← bicycle；其余层原样。配置里 `load_from` 指向迁移后的权重。
- 配置：3 个 task（每类一个），20 个 epoch，one-cycle 最大学习率 5e-4，GT 采样（车 2、人 4、骑行 4），±45° 旋转、±5% 缩放、±2° 俯仰/横滚增强。5k–10k 帧单卡就够。
- 多卡命令（`torch.distributed.launch`）在当前 torch 2.14 上没有验证过。

## ⑦ 评估

```bash
python tools/dist_test.py configs/dog/dog_centerpoint_pp_xt32.py --work_dir work_dirs/dog_pp_eval \
    --checkpoint work_dirs/dog_centerpoint_pp_xt32/latest.pth
```

- 预训练模型和微调模型在**同一批人工检查过的验证录制**上比较：两个都用 `run_sequence.py` 跑，再用 `compare_detectors.py --labels <验证集 labels_reviewed.pkl>` 按 0–10 / 10–20 / 20–40 m 分段比 AP、ATE、AVE。
- 不要用教师自己生成的标签评估教师同架构的模型。
- 看效果：`python tools/dog/run_bag_mcap.py --bag <bag> --ckpt work_dirs/dog_centerpoint_pp_xt32/latest.pth --mcap out.mcap`（配置由权重自动识别）。

## ⑧ 部署到 Orin

`tools/dog/export_pillar_onnx.py` 导出两段 ONNX 并与 PyTorch 逐段校验，再在 Orin 上用 `trtexec --fp16` 建引擎，见 DOG_PIPELINE.md 第 8 节。

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
    --frames work_dirs/rec_001/frames --scene rec_001 --every 5

# 启动标注工具，浏览器打开 http://127.0.0.1:8081
python tools/dog/sustechpoints_convert.py serve
#   等价于上游的 cd third_party/SUSTechPOINTS && python main.py（监听所有网卡）；serve 可用 --host / --port

# 标完导回来，得到和 labels.pkl 一样格式的文件，直接给 export_dataset.py
python tools/dog/sustechpoints_convert.py import --scene rec_001 --out data/labels/rec_001/labels_sustech.pkl
```

导出的场景在 `third_party/SUSTechPOINTS/data/` 下，被 .gitignore 忽略，不会提交到仓库（点云很大）。标好的结果以导回后的 `labels_sustech.pkl` 为准，另行保存。

转换规则（`dog_perception/sustechpoints.py`）：

- **坐标系**：点云和框都在每一帧自己的检测系（重力对齐，地面 z ≈ 0），不需要标定文件，也不写 `ego_pose`（SUSTechPOINTS 的 ego_pose 是 NovAtel 的 UTM 格式，并写死了它自己车上的雷达安装）。
- **点云**：默认导出 5 帧拼接（模型的输入），远处稀疏的物体更容易看清；运动物体会拖影，框要放在拖影最前端（当前帧）。`--current-sweep` 只导当前帧。强度除以 255。
- **帧名**：时间戳的微秒数（16 位数字，按名字排序就是时间顺序）。`dog_meta.json` 保存每帧的时间戳、`T_world_det` 和自动标注的速度、分数，导入时要用，不要删。
- **框**：position 为框中心，scale 为（长、宽、高），rotation.z 为航向；导入时忽略 rotation.x / y。
- **类别**：导出为 Car / Pedestrian / BicycleRider；待审核的轨迹 `obj_attr` 为 `review`，在工具里会显示出来。导入时按 `TYPE_TO_CLASS` 映射回三类：Car、Van、Bus、Truck 等为 vehicle；Pedestrian、RoadWorker、Child 为 pedestrian；BicycleRider、MotorcyleRider、ScooterRider 以及停着的 Bicycle、Motorcycle、Scooter 为 cyclist（和 nuScenes 的映射一致）。其他类型（Cone、TrafficBarrier 等）跳过并计数；`--type-map Trimotorcycle=vehicle` 可以补充。
- **轨迹号**：数字 `obj_id`（自动标注的轨迹）原样保留；工具里新建、不是数字的 ID 分配新号。
- **速度**：框没被挪动（中心偏移 ≤ 5 cm）就保留自动标注的速度，静止物体的处理不受影响；挪动过或新加的框，用同一 `obj_id` 在世界系的中心差分（±0.5 s）重算，低于 0.2 m/s 置 0。所以要改一条轨迹的位置时，用批量模式把整条轨迹一起挪；只挪其中一帧，那几帧的速度会算出一个跳变。

在运动样例 bag 上用真实的 SUSTechPOINTS（2024-06 的最新版本）验证过：导出的 27 帧场景能正常加载（一帧 12 个框、18.7 万个点，网页没有报错，待审核的框显示 `review`）；不做修改直接导回，框、类别、速度与原标签完全一致；在工具里把一辆车挪 1 m 并改成 Van、新加一个行人，再用工具自己的保存功能存盘，导回后车仍是 vehicle、位置偏移 1.00 m，新行人成为一条新轨迹。

## 11. 在样例 bag 上跑通的结果

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
