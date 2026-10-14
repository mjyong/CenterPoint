# 检测模型开发流程：数据 → 标注 → 训练 → 评估对比 → 部署

一轮迭代的完整做法：每一步做什么、用什么命令、产物放在哪、怎样判断可以进入下一步。分步的细节在这些文档里：

| 文档 | 内容 |
|---|---|
| [DOG_PIPELINE.md](DOG_PIPELINE.md) | 四段流水线的设计（预处理、检测、跟踪、预测），Orin 部署方案 |
| [DOG_FINETUNE.md](DOG_FINETUNE.md) | 从 bag 到数据集的每一步的参考说明，标注工具的选择 |
| [LABELING_TUTORIAL.md](LABELING_TUTORIAL.md) | 标注实操：用样例 bag 从转换走到训练数据，含 SUSTechPOINTS 截图 |
| [NUSC_CLIP_EVAL.md](NUSC_CLIP_EVAL.md) | nuScenes 片段上的检测、跟踪、预测评估 |

第 9 节是用两段样例 bag 实际跑通一轮（训练、评估、对比、导出）的记录。

---

## 1. 一轮迭代

```
采集 bag
  ① rosbag_to_sequence        bag → 序列
  ② run_sequence              预处理 + 教师推理（Voxel + TTA），保存模型输入帧
  ③ auto_label                离线跟踪 + 轨迹精修 → 自动标注
  ④ review_labels / SUSTechPOINTS   人工审核：按轨迹（训练集）+ 逐框（验证集）
  ⑤ export_dataset            → DogDataset + GT 采样库
  ⑥ convert_nusc_ckpt + train 迁移 head，微调
  ⑦ dist_test / compare_detectors / nusc_clip_eval   评估与对比
  ⑧ 验收                      和当前线上模型逐类对比，过线才替换
  ⑨ export_pillar_onnx → trtexec   部署到 Orin
  ⑩ 上车验证，新录的 bag 再进 ①
  ⑪ 下一轮：补数据、用微调后的 Voxel 当教师（自训练）
```

| 步骤 | 机器 | 产物 | 进入下一步的条件 |
|---|---|---|---|
| ①–③ | ① ③ 任意；② Voxel 要 GPU | `data/seq/<rec>`、`data/runs/<rec>`、`data/labels/<rec>/labels.pkl` | RTK 固定解；教师结果不为空 |
| ④ | 任意（标注工具在浏览器里） | `labels_reviewed.pkl`、`labels_sustech.pkl` | 训练集的待审核轨迹处理完；验证集逐框看完 |
| ⑤ | 任意 | `data/dog/` | train / val 按录制划分，帧数、各类框数合理 |
| ⑥ | GPU（RTX 4090） | `data/train/<实验名>/epoch_*.pth` | 训练 loss 正常下降 |
| ⑦ ⑧ | GPU | 评估报告、对比表 | 第 5.6 节的验收标准 |
| ⑨ ⑩ | 4090 导出，Orin 建引擎 | `deploy/*.onnx`、`*.engine` | ONNX 校验通过；Orin 上延迟达标 |

## 2. 机器、环境、目录

### 2.1 机器分工

- **GPU 工作站（RTX 4090）**：跑 Voxel 教师（spconv 只能在 GPU 上跑）、正式训练、评估。在你的 4090 上实测，网络单帧 Pillar 约 130 ms，Voxel 约 112 ms，Voxel 双翻转 TTA 约 122 ms（[LABELING_TUTORIAL.md](LABELING_TUTORIAL.md) 第 2 节）。
- **任意 CPU 机器**：①、③、④、⑤ 都能做；Pillar 推理约 0.6 s/帧；训练和评估也能在 CPU 上跑通（第 9 节），但只用来冒烟。
- **Orin**：部署。TensorRT 引擎必须在 Orin 上建。

### 2.2 环境（4090 机器）

```bash
git clone <仓库> && cd CenterPoint && git checkout claude/me/v01
git lfs install && git lfs pull                 # work_dirs/pretrained/nusc_voxel.pth（Voxel 教师）
pip install torch torchvision                   # 选与驱动匹配的 CUDA 版本
pip install -r requirements_dog.txt
pip install spconv-cu120                        # Voxel 教师要 spconv 的 CUDA 版；具体 wheel 名以 spconv 官方 README 为准
python -m pytest tests/dog -q                   # 约 1–2 分钟，应全部通过（1 个跳过）
```

- 不需要运行 `setup.sh`：它编译的 DCN 和旋转 IoU NMS 两个 CUDA 算子，Pillar 和狗的流水线都用不到（检测用 circle NMS）。
- `requirements.txt` 是上游 CenterPoint 的老依赖（有 `Pillow<=6.2.1` 等旧版本限制），以 `requirements_dog.txt` 为准。
- 所有命令在仓库根目录执行，先 `export PYTHONPATH=.`。

### 2.3 目录约定

| 路径 | 内容 | 进 git |
|---|---|---|
| `samples/` | 样例 bag | 是 |
| `work_dirs/pretrained/` | nuScenes 预训练权重（Pillar 直接存，Voxel 用 Git LFS） | 是 |
| `work_dirs/dog_pp_init.pth` | head 迁移后的初始权重（第 4.1 节生成） | 否（`*.pth` 被忽略） |
| `data/seq/<rec>` | 序列 | 否 |
| `data/runs/<rec>` | 模型输入帧 `frames/*.npy`、教师结果 | 否 |
| `data/labels/<rec>` | 自动标注、审核表、审核后的标签 | 否，**要另外备份** |
| `data/dog/` | 训练数据集（det3d 格式） | 否 |
| `data/train/<实验名>/` | 训练输出：`epoch_*.pth`、`latest.pth`、日志 | 否 |
| `third_party/SUSTechPOINTS/data/` | 标注工具里的场景 | 否 |
| `deploy/` | ONNX / TensorRT 引擎 | 否 |

`data/` 整个被 .gitignore 忽略，点云和权重不会被误提交。人工审核的成果（`labels_tracks.csv`、`labels_reviewed.pkl`、`labels_sustech.pkl`）不能丢，要放到共享盘或另外的仓库里备份。

## 3. 数据与标注（①–⑤）

命令和逐步说明见 [DOG_FINETUNE.md](DOG_FINETUNE.md)；第一次做照 [LABELING_TUTORIAL.md](LABELING_TUTORIAL.md) 走一遍样例。要点：

1. **采集**：场景多比帧数多重要，至少 30 段不同场景、5k–10k 帧；专门录 4 m 内的近距离交互；RTK 固定解。
2. **划分**：按整段录制划 train / val，不按帧。验证集要覆盖要关心的情况（近距离行人、骑行者、停车场、路边），并且**逐框人工审核**。
3. **教师**：Voxel + 双翻转 TTA（`run_sequence.py --detectors voxel --tta --device cuda:0`）。
4. **审核**：训练集至少处理待审核轨迹（`review_labels.py`）；验证集在 SUSTechPOINTS 里逐框看完，没标为待审核的也会错（样例里分数 0.50 的"骑行者"是三脚架）。
5. **导出**：`export_dataset.py ... --val-recordings <验证集录制> --out data/dog --gt-db --skip-review`。

**数据集版本**：配置文件里的路径写死为 `data/dog`。每做一版数据集，导出到 `data/dog_vNNN`，再用软链接切换，训练记录里写上版本号：

```bash
python tools/dog/export_dataset.py ... --out data/dog_v002 --gt-db --skip-review
ln -sfn dog_v002 data/dog
```

## 4. 训练（⑥）

### 4.1 head 迁移

```bash
python tools/dog/convert_nusc_ckpt.py --src work_dirs/pretrained/nusc_pp.pth --dst work_dirs/dog_pp_init.pth
```

nuScenes 模型有 6 个 task（10 类）。狗的配置有 5 个 task：车按尺寸分成 car / truck / bus 三个 head，再加行人、骑行者；输出时 car / truck / bus 合并成 vehicle，所以下游仍然只有三类。迁移时 car、truck、bus 分别继承 nuScenes 的 car、truck、bus head，pedestrian 继承 pedestrian，cyclist 继承 bicycle；backbone、neck 和共享卷积原样保留。配置里 `load_from` 指向这个文件。

为什么车要分三个 head、实验结果和代价，见 [DOG_FINETUNE.md](DOG_FINETUNE.md) 第 12 节。旧的单 vehicle head 只能继承 car 的 head，nuScenes 的卡车、公交能力全部丢掉（样例验证集上 vehicle AP@2m 从 0.80 掉到 0.16，第 9 节），新布局把它们原样带了过来。旧布局的配置保留为 `*_3cls.py`，迁移时加 `--config configs/dog/dog_centerpoint_pp_xt32_3cls.py`。

### 4.2 配置要点（`configs/dog/dog_centerpoint_pp_xt32.py`）

| 项 | 值 | 说明 |
|---|---|---|
| `tasks` | car / truck / bus / pedestrian / cyclist 各一个 | 输出合并为 vehicle / pedestrian / cyclist；新加类别要在 `convert_nusc_ckpt.py` 的 `NUSC_HEAD` 里指定从哪个 nuScenes head 继承 |
| `PC_RANGE` | ±40.8 m，z −5–3 m（虚拟 nuScenes 雷达系，地面约 −1.84） | 与线上预处理一致，不要改 z |
| `VOXEL` | 0.2 × 0.2 × 8 m | 输出 102 × 102 |
| `nsweeps` | 5 | 与线上一致 |
| `db_sampler.sample_groups` | car 2、truck 2、bus 2、人 4、骑行 4 | GT 采样，每帧补到这个个数；GT 库里没有的类别（比如还没录到公交）跳过 |
| 数据增强 | ±45° 旋转、±5% 缩放、平移 σ 0.1 m、±2° 俯仰/横滚 | |
| `samples_per_gpu` / `workers_per_gpu` | 4 / 8 | batch 和数据加载进程数 |
| `lr_config` | one-cycle，最大 5e-4 | 微调用的学习率，比从头训练小 |
| `total_epochs` | 20 | |
| `checkpoint_config.interval` | 1 | 每个 epoch 存一次，每个 61 MB |
| `load_from` | `work_dirs/dog_pp_init.pth` | 第 4.1 节的输出 |

改参数时复制一份配置（例如 `configs/dog/dog_pp_bs8.py`）再改，别直接改原文件，实验记录里写配置文件名。

### 4.3 在 4090 上训练

```bash
python tools/train.py configs/dog/dog_centerpoint_pp_xt32.py --work_dir data/train/dog_pp_v001
```

- 单卡直接运行，不用 `torch.distributed.launch`（仓库里的多卡写法在 torch 2.x 上没有验证过）。
- 输出：`data/train/dog_pp_v001/epoch_N.pth`、`latest.pth`（指向最新一个）、`<时间>.log`（文本日志）、`<时间>.log.json`（每行一组 loss）、`det3d/`（源码备份）。
- 中断后接着训：`--resume_from data/train/dog_pp_v001/epoch_N.pth`（会恢复 epoch、迭代数和优化器状态）。
- batch：默认 4。4090 有 24 GB，显存有富余时可以把 `samples_per_gpu` 调到 8，同时把 `lr_max` 按比例放大到 1e-3。这组参数没有在 4090 上实测过，第一次先用默认值，看日志里的 `time` 和 `memory`。
- 没有 GPU 时同一条命令会在 CPU 上跑（日志里有 `no CUDA device: training on the CPU` 的警告），样例上约 2 分钟一个 epoch，只用来检查流程。

### 4.4 看训练日志

`*.log.json` 每行是一次统计，每个字段按 task 给出三个值（vehicle、pedestrian、cyclist）：

| 字段 | 含义 |
|---|---|
| `loss` | 总 loss = `hm_loss` + 0.25 × `loc_loss` |
| `hm_loss` | 中心热力图（focal loss），反映"找不找得到" |
| `loc_loss` | 框回归（L1），反映"框准不准" |
| `loc_loss_elem` | `loc_loss` 拆成 10 项：x、y 偏移，z，log(w)、log(l)、log(h)，vx、vy，sin(yaw)、cos(yaw)；速度两项的权重是 0.2 |
| `num_positive` | 每帧平均正样本数（框数） |
| `lr`、`time` | 学习率、每次迭代耗时（秒） |

正常的样子：前几个 epoch `hm_loss` 快速下降，之后缓慢下降；`loc_loss_elem` 里位置和尺寸几项稳定在 0.05–0.2。速度两项在静止物体为主的数据上接近 0（自动标注把静止物体的速度置为 0）。

训练 loss 只说明模型记住了训练集。**选哪个 epoch、能不能用，只看验证集**（第 5 节）。

### 4.5 选权重

在验证集上评估最后几个 epoch，选验证集 mAP 最高的。数据少时，越往后越可能过拟合：

```bash
for e in 10 15 20; do
  python tools/dist_test.py configs/dog/dog_centerpoint_pp_xt32.py \
      --work_dir data/train/dog_pp_v001/eval_e$e --checkpoint data/train/dog_pp_v001/epoch_$e.pth | grep Evaluation
done
```

## 5. 评估与对比（⑦ ⑧）

### 5.1 原则

1. **同一套验证标签**：所有模型在同一批人工逐框审核过的验证录制上比较。
2. **标签不能来自被评估的模型**：用 Pillar 当教师生成的标签评估 Pillar，结果会偏高。正式流程里教师是 Voxel + TTA，再经过人工审核；评估 Voxel 本身时更要小心。
3. **逐类、分距离看**：只看 mAP 会掩盖某一类的崩溃（第 9 节的微调模型就是车变好、人和骑行者崩了）。
4. **基线写在同一张表里**：至少有"当前线上模型"（nuScenes 预训练 Pillar）和"迁移后未训练"两行。

### 5.2 第一层：模型级（`tools/dist_test.py`）

```bash
python tools/dist_test.py configs/dog/dog_centerpoint_pp_xt32.py \
    --work_dir data/train/dog_pp_v001/eval --checkpoint data/train/dog_pp_v001/latest.pth
```

在 `data/dog/infos_val.pkl` 上跑网络，打印每类一行 `Evaluation <类别>: ...`，预测保存在 `--work_dir/prediction.pkl`。它评的是**网络本身**：分数 ≥ 0.1、task 内 circle NMS 之后的框，不经过线上的地面 / 点数过滤和合并类别 NMS。适合比较不同 epoch、不同训练配置。

| 指标 | 含义 |
|---|---|
| AP@0.5 / 1 / 2 / 4 | 中心距离阈值 0.5 / 1 / 2 / 4 m 下的 AP |
| mAP（每类） | 上面四个 AP 的平均 |
| ATE | 匹配上（2 m 内）的框的平均中心误差（m） |
| ASE | 平均尺寸误差（1 − 对齐后的 3D IoU） |
| AOE | 平均朝向误差（rad） |
| AVE | 平均速度误差（m/s） |
| AP@2.0[a-b m] | 只算距离 a–b m 内的真值和预测；`nan` / `-` 表示这个距离段没有真值 |
| Evaluation mAP | 5 个 head 的 mAP 平均 |
| `merged/...` | 合并成 vehicle / pedestrian / cyclist、做完跨 head 去重（和检测器输出一致）后的同一套指标；和旧的 3 类模型比、和流水线级结果对照，都看这几行 |

### 5.3 第二层：流水线级（`run_sequence.py` + `compare_detectors.py`）

把模型放进线上的完整流水线（预处理、检测、过滤、合并 NMS、跟踪），在验证录制上和人工标签比，**这是决定替换与否的主要依据**：

```bash
# 每个模型在验证录制上跑一遍（配置按权重自动识别：nuScenes 10 类 / 狗 5 个 head / 旧的狗 3 类）
python tools/dog/run_sequence.py --seq data/seq/rec_val --detectors pillar \
    --ckpt-pillar work_dirs/pretrained/nusc_pp.pth --device cuda:0 --out data/eval/pretrained/rec_val
python tools/dog/run_sequence.py --seq data/seq/rec_val --detectors pillar \
    --ckpt-pillar data/train/dog_pp_v001/latest.pth --device cuda:0 --out data/eval/dog_pp_v001/rec_val

# 对比
python tools/dog/compare_detectors.py --labels data/labels/rec_val/labels_sustech.pkl \
    --results pretrained=data/eval/pretrained/rec_val/pillar/results.pkl \
              dog_pp_v001=data/eval/dog_pp_v001/rec_val/pillar/results.pkl
```

输出两张表：

- 每个模型每类一行：AP@0.5/1/2/4、ATE、ASE、AOE、AVE、mAP，以及 0–10 / 10–20 / 20–40 m 三段的 AP@2m。对狗来说近距离（0–10 m）最重要。
- 每个模型一行：参与评估的帧数、mAP、网络耗时（中位数、p95）、预处理耗时、轨迹数、平均轨迹长度。轨迹数比真实物体多很多、平均长度很短，说明跟踪碎了。

验证集有多段录制时，每段各跑一遍，再把各段的 `results.pkl` 分别对比，或者把标签合并后一起算。

### 5.4 第三层：回归（nuScenes）

微调可能让模型忘掉 nuScenes 上学到的东西。用 nuScenes mini_val 检查（配置同样按权重自动识别）：

```bash
python tools/dog/nusc_clip_eval.py --dataroot data/nuscenes_mini --scenes mini_val --no-mcap \
    --ckpt work_dirs/pretrained/nusc_pp.pth --out data/eval/pretrained/nusc
python tools/dog/nusc_clip_eval.py --dataroot data/nuscenes_mini --scenes mini_val --no-mcap \
    --ckpt data/train/dog_pp_v001/latest.pth --out data/eval/dog_pp_v001/nusc
```

对比两个 `report.md` 里的检测 AP 和跟踪 MOTA。nuScenes 是车载 32 线、场景不同，这里只看**掉了多少**，不追求涨。指标含义见 [NUSC_CLIP_EVAL.md](NUSC_CLIP_EVAL.md)。

### 5.5 第四层：人眼看

在没参与训练和验证的新 bag 上生成 Foxglove 文件，和当前线上模型对着看：

```bash
python tools/dog/run_bag_mcap.py --bag <新 bag> --ckpt data/train/dog_pp_v001/latest.pth --mcap data/eval/dog_pp_v001/new.mcap
python tools/dog/run_bag_mcap.py --bag <新 bag> --ckpt work_dirs/pretrained/nusc_pp.pth --mcap data/eval/pretrained/new.mcap
```

终端会打印每帧平均的原始框数、过滤后框数、上报框数和确认的轨迹数，可以先看这几个数有没有明显异常。

### 5.6 验收标准（建议，按需要调整）

在全部验证录制上，和当前线上模型（nuScenes 预训练 Pillar）比：

| 项 | 建议门槛 |
|---|---|
| 每一类的 AP@2m（流水线级） | 不低于线上模型 |
| 0–10 m 的行人、骑行者 AP@2m | 明显提高（这是微调的主要目的） |
| 三类 mAP | 提高 |
| ATE / AOE / AVE | 不变差 |
| 轨迹数 / 平均轨迹长度 | 不碎得更厉害 |
| nuScenes mini_val mAP | 下降不超过约 0.05 |
| Orin 上的延迟（TensorRT FP16） | 在预算内（[DOG_PIPELINE.md](DOG_PIPELINE.md) 第 8 节） |
| 人眼检查 | 新 bag 上没有成片的误检或漏检 |

任何一类变差都不替换：先查原因（数据不够、标注错、过拟合），补数据再训。

### 5.7 实验记录

每次训练在共享的实验记录里加一行，例如：

| 实验 | 日期 | 数据集 | 配置 | 起点 | epoch | val mAP（流水线） | 车 / 人 / 骑行 AP@2m | 0–10 m 人 / 骑行 | nuScenes mAP | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|
| pretrained | — | — | nuScenes | — | — | 0.525 | 0.795 / 0.732 / 0.095 | — | 0.663 | 线上基线（样例数值） |
| dog_pp_v001 | | dog_v001 | dog_centerpoint_pp_xt32 | dog_pp_init | 20 | | | | | |

## 6. 部署（⑨ ⑩）

### 6.1 导出 ONNX 并校验

```bash
python tools/dog/export_pillar_onnx.py --checkpoint data/train/dog_pp_v001/latest.pth \
    --out deploy/dog_pp_v001 --check data/runs/rec_val/frames/<某一帧>.npy
```

配置按权重自动识别。输出 `pfn.onnx`、`rpn_head.onnx`；`--check` 用 onnxruntime 在一帧真实点云上逐段和 PyTorch 对比，并打印在 Orin 上建引擎的 `trtexec` 命令。校验通过的标准：各段最大相对误差在 1e-4 以下，两边的检测框数相同，分数和中心差可以忽略。

### 6.2 Orin

1. 把 `deploy/dog_pp_v001/` 拷到 Orin，运行上一步打印的 `trtexec ... --fp16` 两条命令建引擎（引擎和 GPU、TensorRT 版本绑定，必须在 Orin 上建）。
2. 当前的 ROS 2 节点（`tools/dog/ros2_node.py`）跑的是 PyTorch 模型：`-p config:=dog_pillar -p checkpoint:=<权重>`（`dog_pillar` 是狗模型（5 个 head）的预设配置名，旧的 3 类权重用 `dog_pillar_3cls`）。把 TensorRT 引擎接进节点是 [DOG_PIPELINE.md](DOG_PIPELINE.md) 第 8 节里还没做完的一步。
3. 上车录一段新 bag，回到第 5.5 节对着看，再进下一轮。

## 7. 下一轮（⑪）

- **补数据**：从第 5.3 节的对比里找漏检、误检最多的场景类型（距离段、类别），专门去录。
- **自训练**：数据量上来以后，用微调好的 Voxel 当教师重新标注（第 ② 步换权重），审核工作量会下降。
- **换教师后重做验证集标签吗**：不用。验证集是人工逐框审核过的，和教师无关，保持不变，不同轮次的结果才能比较。

## 8. 每步检查清单

| 步骤 | 检查 |
|---|---|
| ① 转换 | `pos_type` 全为 56（RTK 固定解）；帧数和 bag 时长相符 |
| ② 教师 | 每帧都有结果；在 Foxglove 里抽查几帧 |
| ③ 自动标注 | 待审核轨迹的比例；静止物体速度为 0 |
| ④ 审核 | 训练集待审核轨迹都处理了；验证集每帧看过；`labels_tracks.csv` 的 `note` 写了理由 |
| ⑤ 导出 | train / val 按录制划分；各类框数；`data/dog` 软链接指向正确版本 |
| ⑥ 训练 | 加载了 `dog_pp_init.pth`（日志里有 `load checkpoint`）；loss 下降 |
| ⑦ 评估 | 基线和新模型用同一套验证标签；逐类、分距离看；nuScenes 回归 |
| ⑧ 验收 | 第 5.6 节逐项通过；实验记录已更新 |
| ⑨ 部署 | ONNX 校验通过；Orin 上延迟达标 |

## 9. 样例数据上跑通一轮的记录

数据来自 [LABELING_TUTORIAL.md](LABELING_TUTORIAL.md)：训练集 `rec_static` 28 帧（只做了按轨迹审核），验证集 `rec_moving` 27 帧（逐框审核）。教师是 nuScenes 预训练 Pillar。这台机器只有 4 核 CPU、没有 GPU，所有步骤都在 CPU 上跑。

这一节是车辆拆成 car / truck / bus 三个 head 之前跑的，用的是现在的 `*_3cls.py` 配置（单个 vehicle head）。

**这些数字只用来说明流程能跑通、指标怎么读，不代表微调的效果**：训练集只有一段停车场录制（行人 36 个框、骑行者 28 个框，各来自一两条轨迹），模型必然过拟合；而且验证标签来自 Pillar 教师，对 Pillar 基线有利。

### 9.1 训练

```bash
python tools/dog/convert_nusc_ckpt.py --src work_dirs/pretrained/nusc_pp.pth --dst work_dirs/dog_pp_init.pth
python tools/train.py configs/dog/dog_centerpoint_pp_xt32.py --work_dir data/train/dog_pp_sample
```

迁移输出 `saved work_dirs/dog_pp_init.pth (295 tensors)`。CPU 上每次迭代约 5–6 s，每个 epoch 约 2 分钟；跑到第 15 个 epoch 停下（流程验证够了），再用 `--resume_from .../epoch_15.pth` 续训，日志 `resumed epoch 15, iter 315`，存出 `epoch_16.pth`。

`*.log.json` 里的 loss（vehicle / pedestrian / cyclist）：

| epoch | lr | loss | hm_loss | loc_loss |
|---|---|---|---|---|
| 1 | 7e-5 | 2.38 / 0.52 / 0.60 | 1.89 / 0.26 / 0.29 | 1.96 / 1.02 / 1.25 |
| 2 | 1.5e-4 | 0.83 / 0.35 / 0.33 | 0.45 / 0.14 / 0.08 | 1.51 / 0.87 / 0.98 |
| 5 | 4.6e-4 | 0.29 / 0.23 / 0.15 | 0.09 / 0.06 / 0.02 | 0.79 / 0.65 / 0.53 |
| 10 | 4.1e-4 | 0.23 / 0.20 / 0.13 | 0.08 / 0.06 / 0.02 | 0.62 / 0.57 / 0.44 |
| 15 | 1.5e-4 | 0.18 / 0.17 / 0.11 | 0.06 / 0.04 / 0.02 | 0.49 / 0.49 / 0.34 |

### 9.2 模型级评估（`dist_test.py`，验证集 27 帧）

| 权重 | vehicle mAP | pedestrian mAP | cyclist mAP | 三类 mAP |
|---|---|---|---|---|
| `dog_pp_init.pth`（迁移后未训练） | 0.134 | 0.668 | 0.025 | 0.276 |
| `epoch_15.pth`（微调） | 0.324 | 0.097 | 0.000 | 0.140 |

### 9.3 流水线级对比（`compare_detectors.py`，验证集 27 帧）

| 模型 | vehicle AP@2m | pedestrian AP@2m | cyclist AP@2m | 三类 mAP | 网络耗时（CPU 中位数） | 轨迹数 |
|---|---|---|---|---|---|---|
| nuScenes 预训练 Pillar（线上） | 0.795 | 0.732 | 0.095 | 0.525 | 593 ms | 30 |
| 迁移后未训练 | 0.159 | 0.727 | 0.042 | 0.301 | 396 ms | 18 |
| 微调 15 个 epoch | 0.355 | 0.122 | 0.000 | 0.139 | 445 ms | 2 |

读法：

- 迁移后 vehicle 大幅下降：只继承了 car 的 head。验证集里有 5.5–7.2 m 长的车，抽查的一帧里 6 辆车迁移后的模型只找到 1 辆（4.5 m 长的那辆）。
- 微调后 vehicle 回升一部分，但行人和骑行者崩了：训练集里只有一两条行人、骑行者轨迹，模型记住了那一个场景。按第 5.6 节，这个模型**不能替换**线上模型；要做的是补数据。
- 3 类模型比 10 类模型少 3 个 head，网络更快（CPU 上 396–445 ms 对 593 ms）。

### 9.4 回归（nuScenes）

为了省时间，只取 mini_val 的 scene-0103 前 20 帧（4 个关键帧，`--max-scenes 1 --max-frames 20`）：

| 模型 | vehicle AP@2m | pedestrian AP@2m | 三类 mAP |
|---|---|---|---|
| nuScenes 预训练 Pillar | 0.458 | 0.815 | 0.636 |
| 微调 15 个 epoch | 0.400 | 0.711 | 0.553 |

在 nuScenes 上也掉了。正式评估用完整的 mini_val（两个场景，CPU 上约 13 分钟）。

### 9.5 导出和可视化

- `export_pillar_onnx.py --checkpoint .../latest.pth --check <一帧>`：ONNX 和 PyTorch 的最大相对误差 pfn 1.1e-5、bev 1.1e-5、heads 2.8e-5；两边都是 7 个框，分数差 2e-6，中心差 7e-6 m；用了 9469 / 30000 个柱。`rpn_head.onnx` 21 MB。
- `run_bag_mcap.py --ckpt .../latest.pth`：运动 bag 27 帧，每帧平均原始 6.6 个框、过滤后 5.5、上报 1.4，MCAP 6.8 MB。

### 9.6 跑通过程中修掉的问题

- det3d 的训练和测试原来只能在 GPU 上跑：`model.cuda()`、数据搬运、续训的 `map_location`、日志里的显存统计都写死了 CUDA。现在没有 GPU 时自动用 CPU，有 GPU 时行为不变。
- 3 类模型的 vehicle、cyclist 在线上流水线里被整个丢掉：类别映射的判断写成了"只要有一个类名在 nuScenes 映射表里就用映射表"，而 pedestrian 恰好在表里，于是 vehicle、cyclist 映射不到、被丢弃。现在 3 类模型的类名原样保留。不修的话，微调好的模型部署上去会看不到任何车和骑行者。
- 各工具原来默认用 nuScenes 10 类配置，换成微调权重要手动指定配置。现在 `run_sequence.py`、`run_bag_mcap.py`、`nusc_clip_eval.py`、`export_pillar_onnx.py` 都按权重自动识别（3 个 head task 的是狗的模型）。
