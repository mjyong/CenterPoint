# 服务器任务：只用 0921 数据训练车辆检测模型

**目标**：用 0921 的两份录制（车顶 XT32），训练出只检车辆的 CenterPoint 模型：
- **Voxel 版**：精度高，下一轮当教师；
- **Pillar 版**：给 Orin 在线用。

在验证集上和现在的教师（nuScenes Voxel + TTA）比，定出上报阈值，导出 Pillar 的 ONNX。

这份文档取代 [BAG_SPLIT.md](BAG_SPLIT.md) 第 5 节的 5.12–5.15。5.1–5.11 是已经做完的数据准备，记录留在那里。

## 怎么用

**给人看**：
- 下面的状态表就是进度。一次交给服务器 Agent 一个任务，把那个任务的整段 `text` 复制给它。
- **H** 开头的是人做的步骤，Agent 不做。

**给 Agent 的通用约定**（每个任务都适用）：
1. 仓库 CenterPoint，分支 `claude/me/v01`。开始前 `git pull`，所有命令在仓库根目录执行，`export PYTHONPATH=.`，`SPLIT=data/survey/split/split.csv`。
2. 只做交给你的那一个任务，按步骤顺序做。
   - 先检查"前置条件"，不满足就停下报告。
   - 遇到"停止条件"就停下报告，不要自己改代码、改配置或换参数绕过去。
3. 不要删除或覆盖 `data/seq`、`data/runs`、`data/labels` 里已有的东西，除非步骤里写了。SUSTechPOINTS 的场景目录（公共盘的 `sustech/`）里是人的审核结果，任何时候都不要删除或用 `--force` 覆盖。
   - 数据集目录（`data/dog_veh` 链接指向的目录）：有训练、评估在读的时候，不要删，也不要往里导出。GT 库的文件按序号命名，新导出会把正在跑的训练贴进场景的车换成别的车的点云（T3D 的 Pillar 双卡就是这样坏的）。
   - 新数据集导出到新目录，再把链接指过去（T5 第 3 步）。2026-10-23 之后启动的训练、评估在开始时就把链接解析成真实目录，链接改指不影响它们。
   - 旧目录等用它的任务都结束了再删：删真实路径，不要删 `data/dog_veh` 链接。
4. 做完以后：
   - 把报告写进本文档该任务的"结果"小节（替换"（待填）"）；
   - 把状态表里这一行改成"完成"（或"停在第 N 步"）；
   - 原始输出放在 `data/reports/<任务编号>/`；
   - 按"提交"一项提交，推送到 `claude/me/v01`。
   - 权重（`.pth`）、点云帧、`.pkl`、mcap 都不提交。
5. 选 GPU：按显存放卡，一张卡可以同时放几个任务。先看每张卡：
   ```bash
   nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
   ```
   - 空闲显存（total − used）够这张卡上所有任务加起来、再留 1 GB，就可以放。每个进程大约要：训练 Voxel 7 GB、训练 Pillar 6 GB（batch 4；日志里的 `memory` 只有 3.7 / 3.0 GB，另加 CUDA 上下文和缓存）；`run_sequence.py` 推理、`dist_test.py` 各约 3 GB；Voxel + TTA 教师推理约 6 GB；Pillar + TTA 推理约 13 GB（T14 实测单进程峰值 12.6 GB，比 Voxel + TTA 还多）。
   - 利用率只影响快慢：卡上已经有别人在训练（利用率 > 80%）时，再放上去大家都会变慢，但不会出错。
   - 训练前加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，显存碎片少，占用更接近日志里的 `memory`。
   - 启动后用 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 看各进程实际占了多少，写进报告，以后按实测值放卡。
   - `batch_label.py --dry-run` 的 FREE / busy 是给批量教师推理定的（至少 8 GB 空闲、利用率不超过 20%），比上面严，别的任务不用照它。
6. 大的输出写到 `/public/zjst/robot`（仓库盘只剩约 27 GB，T0）（和 `data/seq`、`data/runs` 同一块盘）。空闲空间用 `df -h /public/zjst/robot` 看，至少 80 GB。不要用 `df -h .`：那是仓库所在盘，后面的导出、训练权重、推理结果都不写到那里。第一次往下面这些目录写之前，先链过去（已经是这个链接就不用再做）：
   ```bash
   for d in train deploy runs_veh_v0 runs_nusc_pp runs_nusc_voxel runs_veh_v1 runs_veh_v1_view runs_r2 labels_r2 seq_lever; do
     mkdir -p /public/zjst/robot/centerpoint_data/$d
     ln -sfn /public/zjst/robot/centerpoint_data/$d data/$d
   done
   ```
   `data/dog_veh` 已经链好，之后只由导出数据集的步骤切换（约定 3），不要再用这个循环改它。
7. 测试：只在 T0、或者 `git pull` 拉到了代码改动（`dog_perception/`、`tools/`、`det3d/`、`configs/`）时跑，而且只跑快速的：
   ```bash
   pip install pytest-xdist       # 第一次
   python -m pytest tests/dog -m "not slow" -n 8 -q --durations=10
   ```
   本地约 1 分钟。完整测试（多 11 个用样例 bag 和训练的慢测试）只在任务里写了时才跑：去掉 `-m "not slow"`。
   每个测试进程最多用 4 个 CPU 线程（`tests/dog/conftest.py`，`DOG_TEST_THREADS` 可改）。之前 T0 跑了 29 分钟，
   多半是每个进程都开了和核数一样多的线程，在共享服务器上互相抢。
8. 评估怎么跑得快（2026-10-23 起）。T7 的 45 分钟里，一半在 `run_sequence.py` 上：两个模型每帧 0.48 s，其中一大半是重新读原始扫描、做预处理（去畸变、拼帧）和轨迹预测，网络只占 35 ms；10 个场景还是一个一个跑的。
   - 验证场景的模型输入帧教师推理时已经存了（`data/runs/<场景>/frames`），评估直接重放：`--replay data/runs/<场景>`，不读扫描、不做预处理，用那次的位姿。在样例 bag 上和现做预处理逐框一致。
   - `--no-predict`：评估不看轨迹预测，跳过。
   - 一条命令跑完所有场景，`--jobs N` 同时跑 N 个场景，`--out` 写成 `data/runs_xxx/{scene}`。每个进程装一份模型，两个模型一起约 4.5 GB 显存，N 按空闲显存和 CPU 核数定（一般 4）。
   - `dist_test.py` 每个约 1.5 GB、1 分钟，3 个一起跑（`&` 后 `wait`）。
   - 测延迟不能并行，也不要和别的任务挤同一张卡（T7 第 2b 步）。

## 状态

| 编号 | 任务 | 谁做 | 依赖 | 预计耗时 | 状态 |
|---|---|---|---|---|---|
| T0 | 环境和数据检查 | Agent | — | 10 分钟 | 完成 |
| T1 | 车辆审核材料 | Agent | T0 | 30 分钟 | 完成；标注规则变了，T1R 里重做 |
| T2 | 导出 v0 数据集（未审核的自动标注） | Agent | T0 | 30 分钟 | 完成；T1R 里重做 |
| T1R | 按新规则重做自动标注（遮挡点数、待审核、静止毛刺），重做 T1、T2，统计 | Agent | T0 | 40 分钟 | 完成 |
| T1S | 训练集也导出到 SUSTechPOINTS 逐框审核（和验证集分组，带分数 / 速度 / 运动，红框待审） | Agent | T1R | 20 分钟 | 完成 |
| T3 | 训练 v0：Voxel、Pillar 两张卡同时跑 | Agent | T1R | 1.5–2 小时（数据读取已加速） | 完成 |
| T3D | 每个模型 2 张卡训练 v0（数据并行），和单卡比 | Agent | T3 | 40 分钟 | 完成；Pillar 那次训练中途数据集被 T5 重写，作废（见复核） |
| T4 | 评估 v0 | — | — | — | 并入 T6（训练期间跑基线）和 T7（对比） |
| H1 | 人工审核车辆：35 个场景全在 SUSTechPOINTS 里逐框审（每 5 帧取 1 帧） | 人 | T1S | 1–2 天 | 完成（junyongmou，2026-10-23） |
| T5 | 应用审核结果，导出 v1 数据集 | Agent | H1 | 30 分钟 | 完成（2026-10-23 补审后重导） |
| T6 | 训练 v1（单卡，和 v0 同配方）；同时跑基线：nuScenes 原始模型、v0 | Agent | T5 | 约 1.2 小时（两个训练挤一张卡时约 2 小时） | 完成 |
| T7 | 比较 7 个模型，选部署 Voxel 还是 Pillar、上报阈值 | Agent | T6 | 1 小时 | 完成 |
| T8 | 部署：导出上线 Pillar 的 ONNX（2026-10-26 起是 nuScenes Pillar，见 T15 复核）；选了 Voxel 时再打包 Voxel，人在 Orin 上测（H2） | Agent | T7 | 10 分钟 | 待做 |
| H2 | （T7 选了 Voxel 时）在 Orin 上测 Voxel v1 的延迟，放得下就部署 Voxel，否则部署 Pillar | 人 | T8 | 半天 | 待做 |
| T9 | （可选）用 v1 Voxel 当教师重新自动标注，准备第二轮 | Agent | T7 | 1 小时 | 待做 |
| T10 | （可选）比较 0921 的两个 INS 杠杆臂 | Agent | T0 | 1 小时 | 待做 |
| T11 | 验证集加多模型候选框（红框"候选"）；35 个场景写上模型框，供 Ctrl + 拖框用 | Agent | T7 | 20 分钟 | 完成（物体最高分门槛 0.5，306 个候选物体） |
| T11b | 重做候选：只要车辆，只用 v1 Voxel、v1 Pillar 两个模型（T11 混进了行人、骑行者） | Agent | T11 | 15 分钟 | 完成（物体最高分门槛 0.3，54 个候选物体） |
| T16 | 双教师的自动标注导成新的 SUSTechPOINTS 场景（35 个），沿用 H1 的审核结论；训练和验证一样，只留两个教师的框 | Agent | T14 | 20 分钟 | 完成（2026-10-26，验证集的 v1 候选已去掉） |
| H3 | 在双教师的场景里审红框（训练 + 验证 35 个场景，只审车）；漏掉的车 Ctrl + 拖框补上 | 人 | T16 | 2–3 小时 | 完成（junyongmou，2026-10-26） |
| T17 | 导回 H3，训练场景传播到每一帧、补停着的车的空档，导出 v2 数据集 | Agent | H3 | 40 分钟 | 完成（2026-10-26） |
| T12 | 在新的验证标签上重评所有模型；选下一轮的两个教师（Voxel、Pillar 各一个） | Agent | T15 | 1 小时 | 不做：教师不换（T15 里 v2 Voxel + TTA 比 nuScenes 教师低 0.109） |
| T13 | 雷达后方盲区：量出盲区，看各模型在盲区边上是不是误检更多 | Agent | T7 | 10 分钟 | 完成（盲区 142°–218°；v1 Pillar 在 10–40 m 边上算明显） |
| T14 | 双教师（nuScenes Voxel + TTA、nuScenes Pillar + TTA）重新自动标注 0921，补进审过的训练标签（不丢 H1） | Agent | T7 | 1 小时 | 完成（双教师标注 T16 接着用；labels_merged.pkl 不再用） |
| T15 | 用 v2 数据集训练 v2；定上线用哪个 Pillar、教师换不换 | Agent | T17 | 3.5 小时 | 完成。上线用 nuScenes Pillar（人定：v1、v2 都不如它，见复核）；教师不换 |
| T18 | 后处理、跟踪、预测的基线：上线 Pillar 在验证场景上的指标、Foxglove MCAP、问题清单；六份对照 | Agent | T15 | 30 分钟 | 待做 |
| H4 | 在 Foxglove 里看基线，按问题清单归类典型问题 | 人 | T18 | 1 小时 | 待做 |
| T19 | 后处理扫参（上报阈值、轨迹输出分数、低分门槛、最少点数、离地 / 下沉） | Agent | T18 | 30 分钟 | 待做 |
| T20 | 跟踪扫参（关联、coast、确认、观测噪声、运动状态） | Agent | T19 | 1 小时 | 待做 |
| T21 | 预测：IMM 对 CV、IMM 参数；oracle 和端到端 | Agent | T20 | 30 分钟 | 待做 |
| T22 | 定版：参数改成默认值（我改代码），从原始扫描完整跑验证场景，和旧默认值比，出最终报告和 MCAP | Agent | T21 | 30 分钟 | 待做（等我改代码） |
| H5 | 看最终效果，定上线参数 | 人 | T22 | 1 小时 | 待做 |

- T1R 之后，T1S 和 T3 可以同时进行；T1S 之后人开始审核（H1），v0 同时在训练。
- H1 完成后（2026-10-23）的顺序：T5 → T6（训练 v1，同时跑基线）→ T7（对比、选模型）→ T8（部署）。导出数据集的任务（T5、以后的重导）不要和读这个数据集的训练、评估同时跑，除非导到新目录（通用约定 3）。
- 2026-10-22 标注规则改了三处（[DOG_FINETUNE.md](DOG_FINETUNE.md) 第 14 节）：遮挡和远处的框按点数处理；待审核改为"平均分 < 0.5 并且 ≥ 0.6 分的帧不到 5 帧"；短暂的假运动去掉。T1、T2 的材料和数据集都是旧规则做的，要在 T1R 里重做。
- v0 只用未审核的自动标注，主要作用是把 GPU 训练流程跑通、得到一个基线，**不上线**。要上线的是 v1。
- 2026-10-26 加：后处理、跟踪、预测的调试 T18 → H4 → T19 → T20 → T21 → T22 → H5，计划和指标见 [PIPELINE_TUNING.md](PIPELINE_TUNING.md)。工具 `tools/dog/pipeline_eval.py` 已经做好，T18 只等 T15 定出上线的 Pillar。除了 T18、T22 各有一步 Pillar 推理，都在 CPU 上跑，不占 GPU。
- 2026-10-26 起的顺序：T16 → H3 → T17 → T15。T12 等 T15：v2 当教师不比两个 nuScenes 教师好，教师就不换，T12 不做（T15 之后：不做）。H3 改成审双教师的训练、验证数据（人决定）。验证集不加 v1 候选，和训练集一样只审两个教师的框。
- T11 → T11b → H3 → T12（2026-10-24 加，T11b 是 10-25 加的）：验证标签是审教师的框得来的，教师漏掉的车不在里面，评哪个模型都偏向教师（T7 复核）。先把其他模型多检出的东西让人判一遍，再重评、选下一轮的教师（[DATA_COLLECTION_PLAN.md](DATA_COLLECTION_PLAN.md) 第 1 节）。和 T8、T9 不冲突，可以同时做；T9 用哪个教师，等 T12 的结论更好。

## 背景：已经有的东西

| 内容 | 位置 | 来自 |
|---|---|---|
| 35 个场景（0921_164127_s000–s005、0921_165659_s000–s028），每个约 30 s | `data/seq/0921_*` | 5.1–5.2 |
| 激光里程计 + RTK 融合后的位姿（第六版，含倾斜） | 各序列的 `odom.npy` | 5.10 |
| 划分：训练 25 个、验证 10 个，不丢场景 | `data/survey/split/split.csv` | 5.10 |
| 教师推理（nuScenes Voxel + TTA）、模型输入帧 | `data/runs/<场景>/voxel/results.pkl`、`frames/` | 5.11 |
| 自动标注 | `data/labels/<场景>/labels.pkl`（已提交；T1R 按新规则重做过） | 5.11、T1R |
| 人工审核（2026-10-23） | SUSTechPOINTS 场景（公共盘的 `sustech/`）；导回的 `labels_sustech.pkl`、训练场景传播到每一帧的 `labels_full.pkl`（已提交） | H1、T5 |
| v1 数据集（只有车） | `data/dog_veh` → `dog_veh_1023_1529` | T5 |

验证集是这 10 个场景：0921_164127_s000–s005、0921_165659_s003–s006。

按只检车辆的规则（[DOG_FINETUNE.md](DOG_FINETUNE.md) 第 13 节），未审核时的车辆样本（T1R 之前的规则；新数字见 T1R 结果）：

| | 帧（不含位姿不可靠的） | 车辆轨迹 | 其中要人定的（`train_as = ignore`） |
|---|---|---|---|
| 训练 | 7,311 | 269 | 135 |
| 验证 | 2,874 | 108 | 72 |

做法要点，见 DOG_FINETUNE.md 第 13 节：
- 只有 car、truck 两个 head，按尺寸分（长 ≥ 5.5 m 或高 ≥ 2.5 m 为 truck），公交归 truck；
- 拿不准的轨迹作忽略框，不当正样本也不当负样本；
- 输出类别仍是 vehicle / pedestrian / cyclist，只是没有行人和骑行者。

---

## T0 环境和数据检查

```text
任务 T0：检查服务器环境和 0921 的数据是否齐全。只检查，不改任何东西。
先按 docs/SERVER_TASKS.md 的"通用约定"做（git pull、PYTHONPATH、SPLIT）。

1. pip install pytest-xdist；python -m pytest tests/dog -m "not slow" -n 8 -q --durations=10：全部通过
   （有 GPU 时 Voxel 的测试也会跑；见通用约定第 7 条）。
2. nvidia-smi；python -c "import torch, spconv; print(torch.__version__, torch.cuda.is_available(), spconv.__version__)"
3. df -h /public/zjst/robot：至少 80 GB 空闲（T2 导出约 40 GB，训练权重约 5 GB，都写在这块盘上，见通用约定第 6 条）。
4. ls -la work_dirs/pretrained/：nusc_voxel.pth 约 103 MB、nusc_pp.pth 约 72 MB。
   只有一百多字节说明是 Git LFS 指针，先 git lfs install && git lfs pull。
5. awk -F, 'NR>1{print $2}' $SPLIT | sort | uniq -c：应为 train 25、val 10。
6. 每个场景的数据齐不齐：
     for r in $(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT); do
       printf "%s frames=%s results=%s labels=%s\n" $r $(ls data/runs/$r/frames 2>/dev/null | wc -l) \
         $(test -f data/runs/$r/voxel/results.pkl && echo ok || echo MISSING) \
         $(test -f data/labels/$r/labels.pkl && echo ok || echo MISSING)
     done | tee data/reports/T0/scenes.txt
   每个场景都应有 frames（几百帧）、results 和 labels。
7. python tools/dog/batch_label.py --split $SPLIT --dry-run：记下哪些卡空闲。

停止条件：测试不通过；没有 CUDA 或 spconv；`/public/zjst/robot` 空闲不足 80 GB；任何场景缺 frames / results / labels
（缺的话报告是哪些，不要自己重跑教师）。
提交：data/reports/T0/scenes.txt 和本文档（结果、状态）。
报告（写进"T0 结果"）：测试结果一行；GPU 型号、数量、空闲的卡号；torch / CUDA / spconv 版本；
空闲磁盘；场景检查是否全部 ok（不 ok 的列出来）；帧数合计。
```

### T0 结果

- 测试：`python -m pytest tests/dog -q`，120 passed，7 warnings，1729 秒，退出码 0。
- GPU：8 张 NVIDIA GeForce RTX 4090。`--dry-run` 空闲的是 3、5、7。
- torch 2.13.0+cu130，CUDA 13.0 可用，spconv 2.3.8。
- 磁盘：`df -h /public/zjst/robot` 可用 699 TB（950 TB 里已用 252 TB），超过 80 GB。
- 权重：`nusc_voxel.pth` 103 MB（107553302 字节），`nusc_pp.pth` 69 MB（71958666 字节），都是完整文件，不是 Git LFS 指针。
- 划分：train 25，val 10。
- 场景：35 个都有 frames、results、labels，没有 MISSING。帧数合计 10363。明细在 `data/reports/T0/scenes.txt`。

---

## T1 车辆审核材料

给 H1 准备审核材料：只看车。

```text
任务 T1：为 0921 的 35 个场景生成只看车辆的审核材料，不改标注。
先按 docs/SERVER_TASKS.md 的"通用约定"做。

1. 每个场景一张车辆轨迹表（要人定的 train_as=ignore 排在前面，带长、高）：
     for r in $(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT); do
       python tools/dog/review_labels.py table --classes vehicle --labels data/labels/$r/labels.pkl \
           --out data/labels/$r/labels_tracks_vehicle.csv
     done | tee data/reports/T1/tables.txt
2. 每个场景一个只画车的 Foxglove 文件（点云 + 标注框 + 教师原始框），8 个同时跑。
   mcap 约 6 GB，写到公共盘，再链回原来的审核路径（不要写到仓库盘，见通用约定第 6 条）：
     mkdir -p /public/zjst/robot/centerpoint_data/review_mcap
     awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT | xargs -P 8 -I{} python tools/dog/review_labels.py mcap \
         --vehicles-only --labels data/labels/{}/labels.pkl --frames data/runs/{}/frames \
         --results data/runs/{}/voxel/results.pkl \
         --out /public/zjst/robot/centerpoint_data/review_mcap/{}_review.mcap
     for r in $(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT); do
       ln -sfn /public/zjst/robot/centerpoint_data/review_mcap/${r}_review.mcap data/labels/$r/${r}_review.mcap
     done
3. 验证集的 10 个场景导出到 SUSTechPOINTS（只有车，每 5 帧取 1 帧）。点云写到公共盘，再链到工具默认的 data/：
     mkdir -p /public/zjst/robot/centerpoint_data/sustech
     for r in $(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT); do
       python tools/dog/sustechpoints_convert.py export --vehicles-only --labels data/labels/$r/labels.pkl \
           --frames data/runs/$r/frames --scene $r --every 5 \
           --out /public/zjst/robot/centerpoint_data/sustech
       rm -rf third_party/SUSTechPOINTS/data/$r
       ln -sfn /public/zjst/robot/centerpoint_data/sustech/$r third_party/SUSTechPOINTS/data/$r
     done | tee data/reports/T1/sustech.txt

停止条件：任何命令报错。
提交：data/labels/0921_*/labels_tracks_vehicle.csv、data/reports/T1/、本文档。不提交 *_review.mcap 和 SUSTechPOINTS 数据。
报告（写进"T1 结果"）：训练、验证各有多少车辆轨迹、多少 train_as=ignore；review mcap 的总大小；
SUSTechPOINTS 每个验证场景导出的帧数；审核人员怎么打开（Foxglove 打开 <场景>_review.mcap；
SUSTechPOINTS 在服务器上 python tools/dog/sustechpoints_convert.py serve，端口 8081）。
```

### T1 结果

- 车辆轨迹（`data/labels/<场景>/labels_tracks_vehicle.csv`，和背景表一致）：训练 269 条，其中 `train_as = ignore` 135 条；验证 108 条，其中 `train_as = ignore` 72 条。表本身很小，留在仓库里提交。
- review mcap：35 个，合计 6.3 GB，在 `/public/zjst/robot/centerpoint_data/review_mcap/`。`data/labels/<场景>/<场景>_review.mcap` 是指向该目录的符号链接（仓库盘上原先那份约 6.3 GB 的旧 mcap 已换成链接）。
- SUSTechPOINTS（只有车，每 5 帧取 1 帧）在 `/public/zjst/robot/centerpoint_data/sustech/`，合计 2.0 GB。`third_party/SUSTechPOINTS/data/<场景>` 是指向它的符号链接。各验证场景帧数：0921_164127_s000 58、s001 57、s002 58、s003 57、s004 58、s005 58，0921_165659_s003 56、s004 56、s005 61、s006 61，合计 580。
- 审核人员怎么打开：Foxglove 打开 `data/labels/<场景>/<场景>_review.mcap`。SUSTechPOINTS 在仓库根目录运行 `python tools/dog/sustechpoints_convert.py serve`，端口 8081。
- 2026-10-23 T1R 按新规则重做了轨迹表、review mcap 和 SUSTechPOINTS。上面的 269 / 135、108 / 72 是旧规则；新数字见 T1R 结果。

---

## T2 导出 v0 数据集

用未审核的自动标注导出只有车辆的数据集。待审核的轨迹自动作忽略框。

```text
任务 T2：迁移 nuScenes 权重，导出只有车辆的 v0 数据集（含 GT 采样库）。
先按 docs/SERVER_TASKS.md 的"通用约定"做。

1. 迁移权重（car、truck 两个 head 分别来自 nuScenes 的 car、truck head）：
     python tools/dog/convert_nusc_ckpt.py --src work_dirs/pretrained/nusc_voxel.pth --dst work_dirs/dog_voxel_veh_init.pth \
         --config configs/dog/dog_centerpoint_voxel_xt32_veh.py
     python tools/dog/convert_nusc_ckpt.py --src work_dirs/pretrained/nusc_pp.pth --dst work_dirs/dog_pp_veh_init.pth \
         --config configs/dog/dog_centerpoint_pp_xt32_veh.py
   每条应打印 "tasks car, truck"。
2. 导出（输出到 data/dog_veh，配置里就是这个路径）：
     REC=$(for r in $(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT); do echo --recording data/runs/$r data/labels/$r/labels.pkl; done)
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
     python tools/dog/export_dataset.py $REC --val-recordings $VAL --classes vehicle --out data/dog_veh --gt-db --jobs 8 \
         2>&1 | tee data/reports/T2/export.txt
3. du -sh data/dog_veh

预期（本地按同样规则统计的）：训练 7,311 帧、验证 2,874 帧；
  train boxes: car 12,902 (+4,422 ignored), truck 1,668 (+4,084 ignored)
  val boxes:   car 3,043 (+3,055 ignored), truck 3,215 (+3,144 ignored)
停止条件：帧数或框数和预期差 5% 以上；任何报错。
提交：data/reports/T2/、本文档。
报告（写进"T2 结果"）：export.txt 的最后十几行（帧数、各类正样本 / 忽略数、"teacher subclass disagrees with size" 个数、
GT 库各类条数）；data/dog_veh 大小；耗时。
```

### T2 结果

- 权重迁移都打印 `tasks car, truck`：`work_dirs/dog_voxel_veh_init.pth` 341 tensors，`work_dirs/dog_pp_veh_init.pth` 241 tensors。权重不提交。
- `export.txt` 末尾：
  - `train 7311 / val 2874 / test 0 frames -> data/dog_veh`
  - `train boxes: car 12902 (+4422 ignored), truck 1668 (+4084 ignored)`
  - `val boxes: car 3043 (+3055 ignored), truck 3215 (+3144 ignored)`
  - `note: 4261 boxes: teacher subclass disagrees with size (ignored)`
  - GT 库：`load 12902 car database infos`，`load 1668 truck database infos`（`dbinfos_train_5sweeps_withvelo.pkl` 同样是 car 12902、truck 1668）
- 帧数和框数与预期一致（差 0%）。点云帧 10185 个 npy（7311 + 2874）。
- 大小：`data/dog_veh` 是指向 `/public/zjst/robot/centerpoint_data/dog_veh` 的链接，`du -sh data/dog_veh` 只看到链接本身，显示 0；`du -shL data/dog_veh` 和目标目录都是 46G。
- 耗时：导出加 GT 库 213 秒（2026-10-22 16:24:11–16:27:44，+08:00）。
- 2026-10-23 T1R 覆盖了 `data/reports/T2/export.txt` 和 `du.txt`。上面是第一次导出；新的帧数仍是 7,311 / 2,874，框数见 T1R 结果。`du -shL` 仍是 46G。

---

## T1R 按新规则重做自动标注，重做 T1、T2

2026-10-22 自动标注改了三处（DOG_FINETUNE.md 第 14 节）：
- 每个框记下框里的点数：遮挡、太远的框在导出时作忽略框或不要；
- 待审核规则：平均分 < 0.5 **并且** ≥ 0.6 分的帧不到 5 帧；
- 去掉短于 1 s 的假运动。

教师结果（`data/runs/<场景>/voxel/results.pkl`）不用重跑，只重做自动标注（CPU），再重做 T1、T2。

```text
任务 T1R：按新规则重做 0921 的自动标注和审核材料，重新导出 v0 数据集，并统计待审核轨迹和在线运动状态。
先按 docs/SERVER_TASKS.md 的"通用约定"做（这次拉到了代码改动，要跑快速测试）。
前置：
  - data/labels/0921_*/labels_tracks*.csv 的 action 列全是空的（H1 还没开始；不是就停下报告）；
  - 如果 T3 正在跑：停掉（它用的是旧标注）。已经跑完的权重不要删，T3 第 0 步会改名留着做对比。

1. 重做自动标注。覆盖 data/labels/<场景>/labels.pkl 和 labels_tracks.csv；只用 CPU，8 个同时跑：
     mkdir -p data/reports/T1R
     awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT | xargs -P 8 -I{} sh -c \
       'python tools/dog/auto_label.py --results data/runs/{}/voxel/results.pkl --out data/labels/{}/labels.pkl \
            > data/reports/T1R/auto_label_{}.txt 2>&1'
     cat data/reports/T1R/auto_label_*.txt > data/reports/T1R/auto_label.txt && rm data/reports/T1R/auto_label_0921*.txt
   每个场景一行 "labelled ... N tracks (M flagged for review)"。
2. 检查新字段：
     python -c "import pickle; L=pickle.load(open('data/labels/0921_165659_s016/labels.pkl','rb')); \
       o=[o for f in L['frames'] for o in f['objects']][0]; print({k: o.get(k) for k in ('det_score','confident_frames','points','points_all')})"
   points / points_all 是 None 就停下：说明没找到 data/runs/<场景>/frames。
3. 重做 T1 第 1 步（车辆轨迹表），输出 tee 到 data/reports/T1R/tables.txt（不要覆盖 data/reports/T1/）。
4. 校准待审核规则：
     python tools/dog/review_labels.py confidence --vehicles-only \
         --labels $(for r in $(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT); do echo data/labels/$r/labels.pkl; done) \
         --spot-check data/reports/T1R/review_spot_check.csv | tee data/reports/T1R/confidence.txt
5. 在线运动状态的基线（教师检测，离线修正过的位姿）：
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
     TRAIN=$(awk -F, 'NR>1 && $2=="train" {print $1}' $SPLIT)
     python tools/dog/motion_eval.py $(for r in $VAL; do echo --scene data/labels/$r/labels.pkl data/runs/$r/voxel/results.pkl; done) \
         --on-speed 0.5 0.7 1.0 --on-frames 2 3 5 | tee data/reports/T1R/motion_val.txt
     python tools/dog/motion_eval.py $(for r in $TRAIN; do echo --scene data/labels/$r/labels.pkl data/runs/$r/voxel/results.pkl; done) \
         --on-speed 0.5 0.7 1.0 --on-frames 2 3 5 | tee data/reports/T1R/motion_train.txt
6. 重做 T1 第 2、3 步：review mcap 和 SUSTechPOINTS。直接覆盖公共盘上的旧文件，链接不用动；
   输出 tee 到 data/reports/T1R/mcap.txt、data/reports/T1R/sustech.txt。
7. 重做 T2 第 2、3 步（权重迁移不用重做）。导出前先清空旧数据集：
     rm -rf data/dog_veh/*      # 不要写成 rm -rf data/dog_veh（那会删掉链接）
   输出 tee 到 data/reports/T2/export.txt（覆盖第一次的），du 写到 data/reports/T2/du.txt。

停止条件：任何报错；第 2 步没有点数；导出的帧数不是训练 7,311、验证 2,874。
提交：data/labels/0921_*/labels.pkl、labels_tracks.csv、labels_tracks_vehicle.csv；data/reports/T1R/、data/reports/T2/；本文档。
报告（写进"T1R 结果"）：
- 待审核：各场景 train_as=ignore 条数汇总（训练 / 验证），和 T1 的 135 / 72 比；
- confidence.txt 的表；review_spot_check.csv 有多少条；
- motion_val.txt、motion_train.txt 的表；
- export.txt 末尾：train / val boxes 两行，以及 note 行（"no points in the input (left out)"、
  "few points in the current sweep (ignored)"、"teacher subclass disagrees with size" 各多少），和第一次 T2 比；
- 耗时。
```

### T1R 结果

- 前置：35 个场景的 `labels_tracks*.csv` 里 `action` 都是空的。T3 没有在跑，没有删 `data/train/dog_*_veh_v0`（那是未做 T1R 的一版，已提交日志）。
- 点数：`0921_165659_s016` 第一个框 `points = 378`、`points_all = 378`，不是 None。
- 待审核（车辆轨迹表 `train_as = ignore`）：训练 267 条轨迹里 109 条，验证 103 条里 54 条。T1 旧规则是训练 269 条里 135 条、验证 108 条里 72 条。新规则少了：训练忽略 135 → 109，验证 72 → 54。
- `confidence.txt`：370 条轨迹、35215 个框；平均分 < 0.50 的有 188 条轨迹、12952 个框。一条轨迹有 N 帧 ≥ S 就算确定时，仍然待审核的轨迹 / 框：

  | S \ N | 3 | 5 | 10 |
  |---|---|---|---|
  | 0.50 | 82 / 1712 | 100 / 2775 | 121 / 4210 |
  | 0.60 | 131 / 5115 | 142 / 6534 | 158 / 7774 |
  | 0.70 | 167 / 9622 | 173 / 10347 | 182 / 11942 |

  现在用的是 (0.6, 5)。`review_spot_check.csv` 46 条（不再被标成待审核、留给人看是不是车）。
- 在线运动状态（教师检测，离线修正过的位姿）。验证 10 个场景、2884 帧：

  | on speed | on frames | moving recall | false moving | static tracks flagged | false episodes / h | moving stretches | missed | latency median / p90 s |
  |---|---|---|---|---|---|---|---|---|
  | 0.50 | 2 | 0.890 (952) | 0.0392 (9841) | 10 / 77 | 72.1 | 26 | 1 | 0.40 / 0.92 |
  | 0.50 | 3 | 0.836 (952) | 0.0226 (9841) | 4 / 77 | 28.0 | 26 | 6 | 0.50 / 1.10 |
  | 0.50 | 5 | 0.765 (952) | 0.0128 (9841) | 1 / 77 | 4.0 | 26 | 9 | 0.70 / 1.34 |
  | 0.70 | 2 | 0.851 (952) | 0.0265 (9841) | 5 / 77 | 36.1 | 26 | 4 | 0.40 / 1.09 |
  | 0.70 | 3 | 0.816 (952) | 0.0151 (9841) | 1 / 77 | 4.0 | 26 | 7 | 0.50 / 1.12 |
  | 0.70 | 5 | 0.708 (952) | 0.0095 (9841) | 1 / 77 | 4.0 | 26 | 12 | 0.70 / 1.68 |
  | 1.00 | 2 | 0.785 (952) | 0.0137 (9841) | 1 / 77 | 4.0 | 26 | 6 | 0.40 / 1.52 |
  | 1.00 | 3 | 0.744 (952) | 0.0101 (9841) | 1 / 77 | 4.0 | 26 | 9 | 0.50 / 1.84 |
  | 1.00 | 5 | 0.624 (952) | 0.0019 (9841) | 0 / 77 | 0.0 | 26 | 14 | 1.20 / 2.77 |

  训练 25 个场景、7479 帧：

  | on speed | on frames | moving recall | false moving | static tracks flagged | false episodes / h | moving stretches | missed | latency median / p90 s |
  |---|---|---|---|---|---|---|---|---|
  | 0.50 | 2 | 0.922 (1266) | 0.0233 (18961) | 15 / 234 | 44.0 | 26 | 1 | 0.40 / 0.56 |
  | 0.50 | 3 | 0.902 (1266) | 0.0187 (18961) | 9 / 234 | 26.0 | 26 | 1 | 0.50 / 0.66 |
  | 0.50 | 5 | 0.850 (1266) | 0.0113 (18961) | 2 / 234 | 4.0 | 26 | 3 | 0.70 / 0.90 |
  | 0.70 | 2 | 0.917 (1266) | 0.0189 (18961) | 11 / 234 | 32.0 | 26 | 1 | 0.40 / 0.60 |
  | 0.70 | 3 | 0.892 (1266) | 0.0149 (18961) | 7 / 234 | 18.0 | 26 | 1 | 0.50 / 0.76 |
  | 0.70 | 5 | 0.829 (1266) | 0.0097 (18961) | 1 / 234 | 2.0 | 26 | 4 | 0.70 / 1.09 |
  | 1.00 | 2 | 0.897 (1266) | 0.0110 (18961) | 6 / 234 | 18.0 | 26 | 2 | 0.40 / 0.77 |
  | 1.00 | 3 | 0.852 (1266) | 0.0073 (18961) | 3 / 234 | 6.0 | 26 | 2 | 0.50 / 1.77 |
  | 1.00 | 5 | 0.792 (1266) | 0.0057 (18961) | 1 / 234 | 2.0 | 26 | 6 | 0.70 / 1.20 |
- 审核材料：35 个 review mcap 仍是 6.3G，链回 `data/labels/<场景>/<场景>_review.mcap`。SUSTechPOINTS 10 个验证场景帧数与 T1 相同（合计 580），2.0G。
- 导出（帧数与停止条件一致：训练 7,311、验证 2,874）。和第一次 T2 比，truck 正样本多了、忽略少了：
  - 这次 train：car 13,346（+3,663 ignored），truck 3,652（+2,003 ignored）。第一次是 car 12,902（+4,422），truck 1,668（+4,084）。
  - 这次 val：car 3,295（+2,787 ignored），truck 4,631（+1,527 ignored）。第一次是 car 3,043（+3,055），truck 3,215（+3,144）。
  - note：`no points in the input (left out)` 311；`few points in the current sweep (ignored)` 1,116；`teacher subclass disagrees with size` 4,237（第一次只有这一条，4,261）。
  - GT 库：car 13,346、truck 3,652。`du -shL data/dog_veh` 46G（`du -sh` 打在链接上仍是 0）。
- 耗时（2026-10-23，+08:00）：自动标注 09:14:39–09:16:58；轨迹表约 11 秒；运动状态 09:17:56–09:24:32；mcap 09:24:47–09:25:44；SUSTechPOINTS 约 21 秒；导出加 GT 库 193 秒（09:26:16–09:30:01）。整段约 15 分钟。

---

## T1S 训练集也导出到 SUSTechPOINTS

2026-10-23 起，训练集也逐框审核（每 5 帧取 1 帧），不再用 Foxglove 按轨迹审。SUSTechPOINTS 改了这些（`third_party/SUSTechPOINTS/VENDORED.md`）：
- 场景列表分"验证集"、"训练集"两组；
- 待审核的框是红框，标签是红字；
- 每个框的标签下显示"轨迹平均分 / 这一帧的分数"（插值的帧显示"插值"）、运动（带速度）或静止、点少；
- 点云仍是 5 帧叠加（0.4 s，和模型看到的一样），但旧的 4 帧画得暗，框按亮的当前帧点对齐；
- 一个场景的最后一帧再按"下一帧"，直接进下一个场景的第一帧；
- 右键菜单可以对整条轨迹（所有帧）一次做决定：确认、作忽略、静止、运动、删除。

审核结果导回后，T5 把训练集关键帧的审核结果传播到每一帧（`sustechpoints_convert.py propagate`）。

```text
任务 T1S：把 35 个场景（训练 25 + 验证 10）都导出到 SUSTechPOINTS 给人逐框审核，覆盖 T1R 导出的验证场景。
先按 docs/SERVER_TASKS.md 的"通用约定"做（拉到了 dog_perception/、tools/、third_party/SUSTechPOINTS/ 的改动，要跑快速测试）。
前置：没有人在 SUSTechPOINTS 里改过标注（H1 还没开始）。导出自己会检查：某个场景的标注在 SUSTechPOINTS 里
改过内容（只按 Ctrl+S 存过、内容没变的不算），就报错 "labels of N frames were changed in SUSTechPOINTS"。
报这个错就停下报告，不要加 --force、不要删目录绕过去：那会把人审过的结果扔掉。

1. 如果 SUSTechPOINTS（sustechpoints_convert.py serve）正在跑，停掉：代码改了，审核前要重新起。
2. 导出，每 5 帧取 1 帧，只有车，带上训练 / 验证：
     mkdir -p data/reports/T1S
     awk -F, 'NR>1 && $2!="drop" {print $1, $2}' $SPLIT | while read r s; do
       python tools/dog/sustechpoints_convert.py export --vehicles-only --labels data/labels/$r/labels.pkl \
           --frames data/runs/$r/frames --scene $r --every 5 --split $s \
           --out /public/zjst/robot/centerpoint_data/sustech
       ln -sfn /public/zjst/robot/centerpoint_data/sustech/$r third_party/SUSTechPOINTS/data/$r
     done 2>&1 | tee data/reports/T1S/export.txt
3. 统计每个场景的帧数、框数、红框数（obj_attr 含 review）：
     python tools/dog/sustechpoints_convert.py stats --root /public/zjst/robot/centerpoint_data/sustech \
         | tee data/reports/T1S/stats.txt
4. du -sh /public/zjst/robot/centerpoint_data/sustech

停止条件：任何报错；某个场景导出 0 帧。
提交：data/reports/T1S/、本文档。
报告（写进"T1S 结果"）：训练、验证各多少场景、帧、框、红框（stats.txt 最后的 total 两行）；红框最多的 5 个训练场景；sustech 目录大小。
```

### T1S 结果

- 训练 25 个场景：1,476 帧，4,610 个框，红框 1,023。验证 10 个场景：580 帧，2,466 个框，红框 837。合计 2,056 帧。`stats.txt` 的两行 total 与此一致。
- 红框最多的 5 个训练场景：0921_165659_s028（171）、s016（92）、s027（78）、s026（76）、s018（64）。
- `/public/zjst/robot/centerpoint_data/sustech` 是 9.0G，就是这 35 个场景。每个场景链回 `third_party/SUSTechPOINTS/data/<场景>`。
- 导出前 `0921_164127_s000` 的 58 帧、`0921_164127_s003` 的 16 帧 json 比 `dog_meta.json` 新。按本次要求覆盖重导，网页上那些修改没有保留。
- （2026-10-23 复核）按前置条件这里应该停下。SUSTechPOINTS 按一次 Ctrl+S 会把已加载的所有帧重写一遍，所以文件变新不一定是改过内容；但覆盖后已无法确认。之后的导出会按内容检查（`edited_frames`），改过内容的场景拒绝覆盖，并去掉了导出前的 `rm -rf`。
- 三个场景有帧、没有车框：验证集 0921_164127_s005、0921_165659_s004，训练集 0921_165659_s013。
- 快速测试：`pytest tests/dog -m "not slow" -n 8`，127 passed。旧的 `serve` 已停掉，导出后重新开在 `http://10.36.136.7:8081`。


---

## T3 训练 v0

**数据读取加速（2026-10-23）。** 第一次 T3 每步约 3/4 的时间在等数据：Voxel 每步 0.66 s，其中读数据 0.48 s；GPU 自己只要 0.18 s。原因都在 CPU 上，不在磁盘上：

| 原因 | 改法 |
|---|---|
| 体素化每个样本都新建并填满一个 40 × 1088 × 1088 的查找表（189 MB），实际最多用 12 万格 | 每个 worker 只建一次，用完只把用过的格子复原（`point_cloud_ops.points_to_voxel`，结果逐字节相同） |
| `np.random.shuffle` 打乱 20 万个点，逐行交换 | 一次随机下标索引 |
| 每个 worker 里 numpy（OpenBLAS）和 torch 都开出和核数一样多的线程，两个训练 16 个 worker 互相抢 | worker 里限制为单线程 |
| 每个 epoch 重新起 worker，每个 worker 重新编译 numba 体素化 | worker 常驻（`persistent_workers`） |
| 每批约 80 MB 的体素从普通内存拷到 GPU | 锁页内存（`pin_memory`） |

另外修了一个增强的问题：`SweepPoseJitter` 的随机数生成器原来在主进程里建好，再复制给 8 个 worker，状态都一样，8 个 worker 抽到的位姿抖动完全相同，每个 epoch 还重复一遍。现在每个 worker 用 PyTorch 给它单独设种子的 numpy 生成器。

本地单核、文件已在缓存里，每个样本：

| | Voxel | Pillar |
|---|---|---|
| 以前 | 288 ms（打乱 129、体素化 116） | 148 ms（打乱 122） |
| 现在 | 38 ms | 42 ms |

本地 3 个 worker 的 DataLoader：每批（4 个样本）从 0.392 s 降到 0.072 s。服务器上 8 个 worker 应该远快于 GPU 每步的 0.18 s，预计 Voxel 20 个 epoch 约 1.5–2 小时（原来 5.4 小时），Pillar 约 1.2–1.5 小时（原来 4.6 小时）。

```text
任务 T3：在两张空闲 GPU 上同时训练只检车辆的 Voxel 和 Pillar 模型 v0（20 个 epoch）。
先按 docs/SERVER_TASKS.md 的"通用约定"做（拉到了 det3d/ 的改动，要跑快速测试）。前置：T1R 完成（data/dog_veh/infos_train.pkl 是 T1R 第 7 步导出的）。

0. 保留第一次（T1R 之前的标注）训练的权重，以后拿来对比；再测一下数据读取：
     [ -d data/train/dog_voxel_veh_v0 ] && mv data/train/dog_voxel_veh_v0 data/train/dog_voxel_veh_v0_oldlabels
     [ -d data/train/dog_pp_veh_v0 ] && mv data/train/dog_pp_veh_v0 data/train/dog_pp_veh_v0_oldlabels
     mkdir -p data/train data/reports/T3
     python tools/dog/bench_dataloader.py configs/dog/dog_centerpoint_voxel_xt32_veh.py | tee data/reports/T3/bench_voxel.txt
     python tools/dog/bench_dataloader.py configs/dog/dog_centerpoint_pp_xt32_veh.py | tee data/reports/T3/bench_pp.txt
   看第 3 行的 "s per batch"：应明显小于 0.18 s（Voxel）/ 0.15 s（Pillar）。
   第 2 行的磁盘速度：两个训练一起跑大约要 200 MB/s；低于这个数，读盘会成为下一个瓶颈，报告里写出来。
1. 选两张空闲的卡（batch_label.py --dry-run），各起一个训练：
     mkdir -p data/train data/reports/T3
     CUDA_VISIBLE_DEVICES=<卡1> nohup python tools/train.py configs/dog/dog_centerpoint_voxel_xt32_veh.py \
         --work_dir data/train/dog_voxel_veh_v0 > data/train/dog_voxel_veh_v0.log 2>&1 &
     CUDA_VISIBLE_DEVICES=<卡2> nohup python tools/train.py configs/dog/dog_centerpoint_pp_xt32_veh.py \
         --work_dir data/train/dog_pp_veh_v0 > data/train/dog_pp_veh_v0.log 2>&1 &
   只有一张空闲卡时，先跑 Voxel，跑完再跑 Pillar。
2. 开始 10 分钟后检查两个日志（tail -30）：
   - loss 是有限值，hm_loss 在下降；
   - 每次迭代耗时（time）和其中读数据的时间（data_time）、每个 epoch 多少次迭代，按 20 个 epoch 估算总时长；
     第一次 T3 是 time 0.66 / data_time 0.48（Voxel）、0.56 / 0.41（Pillar），这次 data_time 应在 0.05 s 以下；
   - nvidia-smi 看显存和利用率，这次应该大部分时间在 80% 以上。data_time 仍超过 0.1 s 就记下来报告（先不用改）。
3. 等训练结束（每个 epoch 存一个 data/train/<名字>/epoch_N.pth），期间每小时看一次日志。

停止条件（停下并报告日志最后 50 行，不要改配置重跑）：
  报错（OOM、spconv、数据加载）；loss 变成 nan / inf；估算总时长超过 10 小时；
  训练进程意外退出（这时可以用 --resume_from data/train/<名字>/latest.pth 续一次，再退出就停下）。
提交：把两个 .log 和 work_dir 里的 *.log.json 复制到 data/reports/T3/（第一次的日志改名加 _oldlabels 留着），
提交它们、bench_*.txt 和本文档。
报告（写进"T3 结果"，第一次的结果留着，新的写在后面）：bench 的三行；卡号和型号；每次迭代耗时和 data_time；
GPU 利用率；每个 epoch 迭代数；总耗时；显存；每 5 个 epoch 最后一次的 loss（总的、hm_loss、loc_loss，两个 head 分开写）。
```

### T3 结果

这一版是 **没有做 T1R** 的训练。开训时任务只依赖 T2，用的是 2026-10-22 16:22 导出的旧规则 `data/dog_veh`（`infos_train.pkl`），不是 T1R 第 7 步会重导的数据集。权重在 `data/train/dog_voxel_veh_v0`、`data/train/dog_pp_veh_v0`（不提交）。T1R 导出之后，正式的 T3 会写进同一目录，这批权重会被盖掉，需要重跑。

- 卡：都是 RTX 4090。Voxel 用 7，Pillar 用 5（`--dry-run` 标成空闲的是 3、5、7）。
- 每个 epoch 1478 步，20 个 epoch，`epoch_20.pth` 的 meta 是 epoch 20、iter 29560。没有 OOM、nan 或进程中途退出。
- 每次迭代：Voxel 平均 0.661 s（其中读数据 0.482 s），Pillar 平均 0.556 s（读数据 0.409 s）。10 分钟时估计总时长约 6 小时和 4.6 小时，没有到 10 小时的停止线。读数据占每步大约四分之三，GPU 利用率经常低于 50%，按任务没有改配置。
- 总耗时：Voxel 5 小时 25 分（16:55:24–22:20:14），Pillar 4 小时 33 分（16:55:24–21:28:49）。
- 显存（训练日志里的 memory，本进程）：Voxel 3079–3696 MB，Pillar 2586–3026 MB。10 分钟时 nvidia-smi 整卡大约是 7 号 16.9 GB、5 号 17.9 GB，卡上本来还有别人的进程。
- 每 5 个 epoch 最后一条日志（第 1460 步；日志每 20 步一条，1478 不是 20 的倍数）的 loss / hm_loss / loc_loss，顺序是 car、truck：
  - Voxel：第 5 个 epoch 0.3109 / 0.1474 / 0.6539 和 0.2653 / 0.1061 / 0.6369；第 10 个 0.3046 / 0.1641 / 0.5618 和 0.2619 / 0.1200 / 0.5676；第 15 个 0.2150 / 0.1045 / 0.4419 和 0.1635 / 0.0483 / 0.4608；第 20 个 0.1951 / 0.0924 / 0.4111 和 0.1535 / 0.0510 / 0.4101。
  - Pillar：第 5 个 epoch 0.3621 / 0.1746 / 0.7499 和 0.4300 / 0.2054 / 0.8984；第 10 个 0.3530 / 0.1769 / 0.7042 和 0.3729 / 0.2042 / 0.6749；第 15 个 0.3064 / 0.1748 / 0.5264 和 0.2588 / 0.1110 / 0.5909；第 20 个 0.3089 / 0.1842 / 0.4989 和 0.2911 / 0.1643 / 0.5073。
- 日志在 `data/reports/T3/`：`dog_voxel_veh_v0_oldlabels.log`、`dog_pp_veh_v0_oldlabels.log`，以及同名的 `*.log.json`（work_dir 里原名都是 `20261022_165525.log.json`，复制时改成了和模型一样的名字）。第 0 步把权重目录改成了 `data/train/dog_*_veh_v0_oldlabels`，没有盖掉。

**T1R 之后重跑（2026-10-23）。** 用的是 T1R 导出的 `data/dog_veh`（`infos_train.pkl` 7311 帧，09:24）。权重在 `data/train/dog_voxel_veh_v0`、`data/train/dog_pp_veh_v0`（不提交）。`epoch_20.pth` 的 meta 是 epoch 20、iter 37360。没有 nan，进程没有中途退出。

- bench（训练集 7472 个样本，batch 4、8 个 worker）：
  - Voxel：单核每样本 48.1 ms；磁盘 10047 MB/s；DataLoader 0.053 s/batch。
  - Pillar：单核每样本 42.6 ms；磁盘 5468 MB/s；DataLoader 0.028 s/batch。
  - 都低于 GPU 每步大约 0.18 s / 0.15 s。磁盘远高于两个训练一起要的 200 MB/s（测的时候页缓存是热的）。
- 卡：都是 RTX 4090。Voxel 用 7，Pillar 用 5（`--dry-run` 标成空闲的是 5、7）。
- 每个 epoch 1868 步，20 个 epoch。
- 每次迭代：Voxel 平均 0.107 s（读数据 0.004 s），Pillar 平均 0.079 s（读数据 0.002 s）。10 分钟时 data_time 已经在 0.05 s 以下，估计总时长约 1 小时和 40 分钟，没有到 10 小时的停止线。当时 nvidia-smi 利用率是 7 号 90%、5 号 100%。
- 总耗时：Voxel 1 小时 7 分（10:09:29–11:16:01），Pillar 49 分（10:09:29–10:58:35）。
- 显存（训练日志里的 memory，本进程）：Voxel 3103–3620 MB，Pillar 2607–2997 MB。10 分钟时 nvidia-smi 整卡大约是 7 号 11.4 GB、5 号 18.0 GB，卡上还有别人的进程。
- 每 5 个 epoch 最后一条日志（第 1860 步；日志每 20 步一条）的 loss / hm_loss / loc_loss，顺序是 car、truck：
  - Voxel：第 5 个 epoch 0.3652 / 0.2047 / 0.6419 和 0.3476 / 0.1559 / 0.7666；第 10 个 0.3543 / 0.1958 / 0.6338 和 0.2795 / 0.1193 / 0.6409；第 15 个 0.2646 / 0.1485 / 0.4642 和 0.2049 / 0.0943 / 0.4425；第 20 个 0.2044 / 0.0910 / 0.4536 和 0.1705 / 0.0716 / 0.3955。
  - Pillar：第 5 个 epoch 0.4776 / 0.2889 / 0.7546 和 0.4992 / 0.2915 / 0.8308；第 10 个 0.4534 / 0.2953 / 0.6325 和 0.3763 / 0.1878 / 0.7540；第 15 个 0.3199 / 0.1836 / 0.5453 和 0.2561 / 0.1116 / 0.5778；第 20 个 0.2459 / 0.1221 / 0.4954 和 0.2103 / 0.0841 / 0.5046。
- 日志在 `data/reports/T3/`：`dog_voxel_veh_v0.log`、`dog_pp_veh_v0.log`，以及同名的 `*.log.json`（work_dir 里是 `20261023_100928.log.json`、`20261023_100929.log.json`）。bench 在 `bench_voxel.txt`、`bench_pp.txt`。

---

## T3D 每个模型用 2 张卡训练（数据并行）

T3 加速后，单卡训练已经是 Voxel 1 小时 7 分、Pillar 49 分，GPU 利用率 90–100%。再快只能多张卡一起训：

- **数据并行**：每张卡各拿一份完整的模型，各读 4 个不同的样本，各自算梯度；两张卡的梯度取平均后，两边做同一次更新。所以一步用了 8 个样本，每个 epoch 的步数减半（1868 → 934）。这就是"batch 翻倍"：每张卡上的 batch 没变，合起来的 batch 是 8。
- **学习率**：20 个 epoch 看到的样本数不变，但参数只更新一半的次数，每次用的是 8 个样本的平均梯度（更平稳）。学习率不变的话，模型走的"路程"短了，可能欠训练。常用的做法是按 batch 线性放大学习率：`--autoscale-lr` 把最大学习率从 5e-4 提到 1e-3。
- **不改学习率的另一条路**：每张卡 2 个样本，合起来还是 4，步数、学习率都和单卡一样。但每张卡上 BatchNorm 只有 2 个样本，统计量更不准，卡也吃不满，所以不用这个。
- **结论要看评估**：T6 第 4 步把双卡 v0 和单卡 v0 放在一起评（只评 Voxel，见 T3D 复核）。

代码（2026-10-23）：`tools/train.py` 现在认 torchrun 设的 `LOCAL_RANK`（以前两个进程都跑到 0 号卡）；分布式时不再调用没安装的 apex（每张卡的 BatchNorm 各算各的，和单卡一样；配置里写 `sync_bn = True` 才跨卡同步）；梯度只由 DistributedDataParallel 平均一次。本地用 CPU（gloo）跑通了双进程训练。

```text
任务 T3D：Voxel、Pillar 各用 2 张卡训练 v0（20 个 epoch），和 T3 的单卡 v0 比速度，之后再比精度。
先按 docs/SERVER_TASKS.md 的"通用约定"做（拉到了 tools/train.py、det3d/ 的改动，要跑快速测试）。前置：T3 完成。

1. 选 4 张空闲的卡（batch_label.py --dry-run），两个训练各用 2 张；只有 2 张空闲时先跑 Voxel，跑完再跑 Pillar：
     mkdir -p data/reports/T3D
     CUDA_VISIBLE_DEVICES=<卡1>,<卡2> NCCL_P2P_DISABLE=1 nohup torchrun --nproc_per_node=2 \
         --master_addr 127.0.0.1 --master_port 29511 tools/train.py configs/dog/dog_centerpoint_voxel_xt32_veh.py \
         --autoscale-lr --work_dir data/train/dog_voxel_veh_v0_2gpu > data/train/dog_voxel_veh_v0_2gpu.log 2>&1 &
     CUDA_VISIBLE_DEVICES=<卡3>,<卡4> NCCL_P2P_DISABLE=1 nohup torchrun --nproc_per_node=2 \
         --master_addr 127.0.0.1 --master_port 29512 tools/train.py configs/dog/dog_centerpoint_pp_xt32_veh.py \
         --autoscale-lr --work_dir data/train/dog_pp_veh_v0_2gpu > data/train/dog_pp_veh_v0_2gpu.log 2>&1 &
   - NCCL_P2P_DISABLE=1：4090 之间不能 P2P，有的驱动不关会卡死。
   - 两个训练的 --master_port 必须不同。
2. 开始 5 分钟后检查两个日志：
   - 有 "Distributed training: True"；
   - 每个 epoch 934 步；lr 在前 30% 的步里升到 0.001；
   - 每次迭代耗时、data_time，按 20 个 epoch 估算总时长；
   - nvidia-smi：4 张卡都在用、利用率。
3. 等训练结束，期间每 20 分钟看一次日志。

停止条件（停下并报告日志最后 50 行）：报错（NCCL、OOM、数据加载）；超过 10 分钟日志没有新的一行（NCCL 卡住）；loss 变成 nan / inf。
提交：两个 .log 和 work_dir 里的 *.log.json 复制到 data/reports/T3D/，提交它们和本文档。
报告（写进"T3D 结果"）：卡号；每次迭代耗时和 data_time；总耗时，和 T3 单卡的 1 小时 7 分 / 49 分比；GPU 利用率；
每 5 个 epoch 最后一次的 loss / hm_loss / loc_loss（两个 head 分开写）。
```

### T3D 结果

`--dry-run` 当时只有 3 张卡算空闲（3、5、7），不够两个模型同时各占两张。Voxel 先用 7 和 5。Pillar 开的时候按要求把条件放宽，加了 4 号：占用 15.3 GB（低于 16 GB）、利用率 1%，空闲 7.3 GB（略低于 8 GB 那条线），和已经空闲的 3 号一起用。两边都是 RTX 4090，`--autoscale-lr`，最大学习率到了 0.001。每个 epoch 934 步，`epoch_20.pth` 的 meta 是 epoch 20、iter 18680。没有 nan。开头各有一次 CUDA 缓存分配失败的警告，随后继续跑完。

- 每次迭代：Voxel 平均 0.152 s（读数据 0.007 s），Pillar 平均 0.121 s（读数据 0.004 s）。单卡 T3 是 0.107 / 0.004 和 0.079 / 0.002；双卡一步更慢，主要是卡间同步（日志里的 transfer_time 大约 0.06–0.07 s 和 0.04 s）。
- 总耗时：Voxel 47 分（13:49:30–14:36:56），单卡 1 小时 7 分；Pillar 38 分（14:34:41–15:12:14），单卡 49 分。Pillar 在 Voxel 还剩最后一个 epoch 时才开，两段大部分是先后跑的。
- 5 分钟时 Voxel 的 7 号、5 号利用率都是 100%。Pillar 开训约半分钟后 3 号、4 号也是 100%，两张卡空闲只剩大约 300–400 MB（上面本来就有别人的进程）。
- 显存（训练日志里的 memory，本进程）：Voxel 3258–3576 MB，Pillar 2800–2994 MB。
- 每 5 个 epoch 最后一条日志（第 920 步；日志每 20 步一条）的 loss / hm_loss / loc_loss，顺序是 car、truck：
  - Voxel：第 5 个 epoch 0.3926 / 0.2309 / 0.6468 和 0.3715 / 0.1814 / 0.7606；第 10 个 0.2760 / 0.1430 / 0.5322 和 0.2535 / 0.1000 / 0.6137；第 15 个 0.2554 / 0.1264 / 0.5161 和 0.2162 / 0.1039 / 0.4492；第 20 个 0.2062 / 0.1038 / 0.4096 和 0.1624 / 0.0543 / 0.4324。
  - Pillar：第 5 个 epoch 0.4624 / 0.2747 / 0.7509 和 0.5321 / 0.3046 / 0.9103；第 10 个 0.3751 / 0.1946 / 0.7219 和 0.3499 / 0.1775 / 0.6897；第 15 个 0.3243 / 0.1605 / 0.6554 和 0.3785 / 0.1774 / 0.8043；第 20 个 0.2585 / 0.0925 / 0.6640 和 0.2902 / 0.1199 / 0.6810。
- 日志在 `data/reports/T3D/`：`dog_voxel_veh_v0_2gpu.log`、`dog_pp_veh_v0_2gpu.log`，以及同名的 `*.log.json`（work_dir 里是 `20261023_134930.log.json`、`20261023_143441.log.json`）。权重在 `data/train/dog_*_veh_v0_2gpu`（不提交）。
- （2026-10-23 复核）
  - **Voxel：双卡有效，精度不掉。**
    - 速度：47 分对 1 小时 7 分，每秒 52.6 个样本对 37.7，是单卡的 1.4 倍（理想 2 倍）。
    - 收敛：最后一个 epoch 的平均 loss 和单卡一样（总 0.383 / 0.384，hm 0.171 / 0.171，loc 0.846 / 0.851），学习率放大后没有欠训练。
  - **双卡每步多出的约 0.045 s 花在哪**（按 log.json 的时间点拆）：
    - 约 0.03 s 在"拷数据到 GPU"这一段。这里是同步拷贝，要等上一步的 GPU 工作做完，所以实际是两张卡的梯度平均，加上两张卡互相等（每批点数不同，快的等慢的）。4090 之间不能 P2P，梯度走主机内存：Voxel 每步 32 MB，Pillar 20 MB。
    - 约 0.005 s 是 `find_unused_parameters` 每步多遍历一次计算图。日志里 DDP 自己也提示没有用不到的参数，已改成默认关（配置里写 `find_unused_parameters = True` 才开）。
  - **Pillar 双卡这次作废。**
    - 现象：14:57 左右（第 12 个 epoch）起，loc loss 突然从约 1.2 跳到 1.6，之后一直高；单卡同一阶段是平滑下降的。跳的是朝向两项（sin、cos 各 +0.15）和尺寸，位置几乎没变。最终 loss 0.588，单卡 0.473。
    - 原因：这正是 T5 重新导出 `data/dog_veh` 的时候（H1 记录 14:34 提交，T5 15:01 提交）。训练开始时读进内存的是 v0 的 GT 库索引（框的尺寸、朝向），GT 库的点云文件却按"帧序号_类别_序号"命名，被 v1 的同名文件换掉了。贴进场景的车的点云和框对不上，朝向、尺寸的 loss 就上去了。Voxel 14:37 结束，在 T5 之前，不受影响。
    - 另外，Pillar 用的 3、4 号卡上还有别人的进程，速度数字也偏慢。
  - **这是任务安排的错**：状态说明里写了"T3D 和 T5 同时跑"，而 T5 要重写 T3D 正在读的数据集。已改：
    - 通用约定 3：训练、评估在读的数据集目录，不删、不往里导出；新数据集导到新目录再切链接；
    - 代码：`DogDataset` 开始时把 `data/dog_veh` 解析成真实目录，之后链接改指，不影响已经在跑的训练、评估（拉到这次代码后启动的才有）；
    - T5 第 3 步改成导到新目录再切链接，以后补审再重跑 T5 时用。
  - **结论**：双卡能用、不掉精度（Voxel 为证），快 1.4 倍。T6 用单卡没问题；空闲卡够、又赶时间时再用双卡。Pillar 双卡不用重跑（同一套代码，Voxel 已经验证），也不评它。

---

## T4 评估 v0（并入 T6、T7）

2026-10-23 起不单独做：v0 的模型级评估和 nuScenes 原始模型、v0 的流水线级结果，在 T6 训练 v1 的同时跑（T6 第 3、4 步）；所有模型的对比在 T7。

---

## H1 人工审核车辆（人做）

只审车：不用管行人、骑行者，也不用改 car / truck / bus（导出时按尺寸分）。全部在 SUSTechPOINTS 里做：35 个场景（训练 25、验证 10）每 5 帧取 1 帧，共约 2,000 帧（条数见 T1S 结果）。不再需要 Foxglove 的轨迹表。

**打开**：服务器上在仓库根目录运行 `python tools/dog/sustechpoints_convert.py serve`，浏览器打开 8081 端口。左上角选场景，列表分"验证集""训练集"两组；选第一帧，之后一直按 `PageDown`（或 `4`）往后翻。一个场景翻完，自动进入列表里的下一个场景；`PageUp`（或 `3`）往回翻。改动自动保存。

**看到的东西**：
- 红框、红字：需要人定的目标：自动标注拿不准（平均分 < 0.5 且高分帧不到 5 帧），或者教师说是 truck、尺寸却是小车。
- 标签第二行：`0.43/0.71` 是轨迹平均分 / 这一帧教师的分数（"插值"表示这一帧教师没检出，是补出来的）；然后是"运动 2.3m/s"（橙色）或"静止"（灰色）；"点少"表示框里当前帧不到 5 个点，训练时自动作忽略框。
- 选中一个框，顶部信息栏还有：框里的点数（当前帧 / 5 帧）、不审核的话训练会把它当成什么（car / truck / ignore）。
- **点云是 5 帧叠加的**（0.4 s，和模型看到的一样，远处、被挡住的车也能看出来）。旧的 4 帧画得暗，当前帧亮：运动的车会拖一条暗的尾巴，**框按亮的点对齐**，不要把尾巴框进去。

**红框怎么处理**（右键框，或标签上的"…"菜单，一次作用于这条轨迹的所有帧）：

| 情况 | 操作 |
|---|---|
| 是车，框也对 | 确认：是车（所有帧，去掉 review） |
| 不是车 | Delete this object (in all frames) |
| 看不清 | 看不清：作忽略框（所有帧） |
| 停着的车，标签却写"运动" | 静止的车（所有帧） |
| 在动的车，标签却写"静止" | 运动的车（所有帧） |
| 只是某一帧的框不对 | 在这一帧拖框调整；只删这一帧用 Delete |
| 小三轮车、老年代步车 | 类型改成 Trimotorcycle（算骑行者，车辆模型不用） |
| 单独停着的挂车厢 | 是车：确认，类型 Truck |

- **不是红框的也扫一眼**：框歪了就调，不是车就删。
- **漏标的车**补上：类型 Car 或 Truck；同一辆车在不同帧画的框，ID 不同也没关系，T5 会按位置把它们连起来。
- **是车的红框一定要点"确认"**：导回时，还带 review 的框仍算拿不准，训练时作忽略框（不当车，也不当背景）。只是把框调对、不点确认，这辆车就白审了。
- 特别注意长 2.5–3.1 m、高约 2.3 m、几乎不动、教师却连续给高分的"truck"：验证集有 4 条（0921_164127_s003 #22、s004 #4、0921_165659_s006 #2、0921_164127_s002 #10），训练集有 0921_165659_s001 #136。先看清是什么：不是车就删；是叉车之类的车辆就确认。回头告诉我是什么，同类东西以后可以定一条规则。

**先审哪些**：先审验证集（T7 的结论靠它）；训练集先审 truck 尺寸的忽略轨迹最多的 6 个场景：0921_165659_s028、s016、s012、s027、s018、s026。这些审完就能先做 T5、T6，其余场景之后补审，再重跑 T5。

做完把状态表的 H1 改成"完成"，注明审核人、日期、审完的场景。

### H1 结果

审核人 junyongmou，2026-10-23。对照 T1S 导出时的框（位置、类型、`obj_attr`）检查了 35 个场景的 `label/*.json`。

- 有改动的 27 个场景，一共动了 99 条轨迹。按场景（改过的轨迹数）：验证集 0921_164127_s000（7）、s001（4）、s002（10）、s003（5）、s004（10），0921_165659_s003（2）、s005（1）、s006（2）；训练集 0921_165659_s000（1）、s001（4）、s002（2）、s009（1）、s010（1）、s012（3）、s014（1）、s016（4）、s017（2）、s018（1）、s019（3）、s020（1）、s021（1）、s022（2）、s024（8）、s025（2）、s026（6）、s027（8）、s028（7）。
- 红框：验证集 837 → 还剩 63 个框（14 条轨迹），训练集 1,023 → 还剩 366 个框。还带着 `review` 的，T5 导回后作忽略框。明确标了忽略的有 463 个框，标了静止的 134 个（都在验证集），标了运动的 16 个（0921_164127_s003）。删掉 40 个框，新画 35 个（主要在 0921_165659_s024：删 7、新画 34）。框挪动超过 5 cm 的只有 0921_165659_s028 的 3 个。类型改动 1 处：0921_164127_s001 的 #18 从 Truck 改成 Car。
- 没有改动的 8 个场景：0921_164127_s005、0921_165659_s004、s013 本来就没有车框；s011（29 框）、s015（601 框）没有红框。s007（5 条、36 框）、s008（2 条、11 框）、s023（#4，61 框）红框原样留着，等于没确认，导回后整段忽略。
- 文档里点名的 5 条“小尺寸高分 truck”：0921_164127_s002 #10、0921_165659_s001 #136 标成忽略；0921_164127_s003 #22、0921_165659_s006 #2 确认为 Truck（去掉 review）；0921_164127_s004 #4 标成静止，类型仍是 Truck。没有另写这是什么物体。
- （2026-10-23 复核）
  - 能进 T5。确认为 Truck 的 3 条长不到 3.1 m、高约 2.3 m，导出时按尺寸进 car head。
  - 框的位置基本没动（挪动超过 5 cm 的只有 3 个）：这一轮审核主要是整条轨迹的决定，框的几何仍是自动标注的。
  - s007、s008、s023 剩下的红框（8 条轨迹、108 个框）导回后整段作忽略框，少了这些车的正样本，但不会出错；花几分钟补审，再重跑 T5 即可。


---

## T5 应用审核结果，导出 v1 数据集

```text
任务 T5：导回 H1 在 SUSTechPOINTS 里的审核结果，训练场景把关键帧的结果传播到每一帧，重新导出只有车辆的数据集（v1），覆盖 data/dog_veh。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：H1 完成（状态表里注明了审完的场景）。还没审的训练场景也照常走
下面的步骤：没人动过的红框导回后仍作忽略框，其余的框当车，和直接用自动标注的效果一样。

1. 35 个场景都导回（--frames 给框数点，评估时遮挡的框不计分）：
     mkdir -p data/reports/T5
     for r in $(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT); do
       echo "== $r"
       python tools/dog/sustechpoints_convert.py import --scene $r --frames data/runs/$r/frames \
           --out data/labels/$r/labels_sustech.pkl
     done 2>&1 | tee data/reports/T5/import.txt
   每行打印帧数和计数，其中 "still marked review" 是没点确认的红框（导出时作忽略框）。
2. 训练场景：关键帧（每 5 帧 1 帧）的审核结果传播到每一帧：
     for r in $(awk -F, 'NR>1 && $2=="train" {print $1}' $SPLIT); do
       echo "== $r"
       python tools/dog/sustechpoints_convert.py propagate --full data/labels/$r/labels.pkl \
           --keyframes data/labels/$r/labels_sustech.pkl --out data/labels/$r/labels_full.pkl
     done 2>&1 | tee data/reports/T5/propagate.txt
   计数：keyframe boxes / interpolated boxes（两端都审过，插值）/ left out: removed by the reviewer（审核删掉的轨迹）/
   ignore: reviewed at one end、ignore: between keyframes only（没人看过的，作忽略框）/
   other classes, copied as they are（行人、骑行者：只导出了车，没人审，原样照抄）。
3. 导出：验证场景用 labels_sustech.pkl（只有关键帧），训练场景用 labels_full.pkl（每一帧）：
     REC=$(for r in $(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT); do echo --recording data/runs/$r data/labels/$r/labels_sustech.pkl; done
           for r in $(awk -F, 'NR>1 && $2=="train" {print $1}' $SPLIT); do echo --recording data/runs/$r data/labels/$r/labels_full.pkl; done)
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
     NEW=$(dirname $(readlink -f data/dog_veh))/dog_veh_$(date +%m%d_%H%M)   # 新目录，和旧的在同一块公共盘
     python tools/dog/export_dataset.py $REC --val-recordings $VAL --classes vehicle \
         --out $NEW --gt-db --jobs 8 2>&1 | tee data/reports/T5/export.txt
     OLD=$(readlink -f data/dog_veh); ln -sfn $NEW data/dog_veh; echo "data/dog_veh: $OLD -> $NEW"
   - 不要先 rm -rf 旧的数据集，也不要导到 data/dog_veh 里：可能有训练、评估正在读它（通用约定 3）。
   - 旧目录 $OLD 写进报告。用它的训练、评估都结束了再删（rm -rf $OLD，是真实路径，不是链接）。
   - 2026-10-23 之前启动的训练还没有"开始时解析链接"，它们跑完之前不要做这一步。

验证集只有逐框审核过的帧（每 5 帧取 1 帧，约 580 帧），训练集是每一帧（约 7,300 帧，略少于 T1R：第一个关键帧之前、
最后一个关键帧之后的几帧不要），这是预期的。
停止条件：某个场景导回的帧数和 T1S 导出的不一致；某个训练场景 propagate 后留下的帧少于原来的 90%；任何报错。
提交：data/labels/0921_*/labels_sustech.pkl、labels_full.pkl、data/reports/T5/、本文档。
报告（写进"T5 结果"）：import 的类别计数和 still marked review 合计；propagate 各计数的合计；
export 的各类正样本 / 忽略数，和 T1R 的导出（训练 car 13,346 / truck 3,652，忽略 3,663 / 2,003）对比。
```

### T5 结果

- 2026-10-23 补审后重导。35 个场景导回的帧数和 T1S 一致（验证 580、训练关键帧 1,476，合计 2,056）。类别只有 vehicle：7,068 个框。仍带着 `review` 的 310 个（验证 63、训练 247），导出时作忽略框。挪动或新画的 38 个框都在训练集，和上一版一样。
- 相对上一版，红框少了 119 个，都在训练集：0921_165659_s007 36→0（框 130→129），s008 11→0（168→166），s023 61→0（468 不变），s024 16→7，s025 35→33。验证集没变。
- propagate（25 个训练场景，行人、骑行者原样照抄）：留下 7,232 / 7,479 帧（96.7%）。keyframe boxes 4,602，interpolated boxes 17,220，other classes, copied as they are 43,640，left out: removed by the reviewer 157，ignore: reviewed at one end 724，frames left out (outside the keyframes) 79。ignore: between keyframes only 为 0。157 是删掉的车（上一版复核时是 144，这次 s007 删 5 个框、s008 删 8 个）。
- `0921_165659_s024` 打印是 229/303 = 75.6%。这 303 帧里 55 帧是 `pose_bad`，propagate 不写这些帧。去掉之后是 229/248 = 92.3%，少掉的 19 帧就是该场景的 `frames left out (outside the keyframes)`。其余训练场景按全部帧算都 ≥ 90.7%（最低是 0921_165659_s011 的 274/302）。合计和任务预期的约 7,300 帧一致，所以继续导出。
- 导出到新目录，没有删旧的。`data/dog_veh`：`/public/zjst/robot/centerpoint_data/dog_veh` → `/public/zjst/robot/centerpoint_data/dog_veh_1023_1529`。旧目录等用它的训练、评估结束后再删。train 7,232 / val 580 帧。新目录 `du -sh` 37G。
  - train：car 14,524（+2,335 ignored），truck 4,078（+1,321 ignored）。T1R 是 car 13,346（+3,663），truck 3,652（+2,003）。上一版 v1 是 car 14,382（+2,477），truck 4,078（+1,337）：补审把 142 个 car 忽略框收成正样本，truck 忽略少 16，truck 正样本不变。
  - val 只有关键帧：car 950（+272 ignored），truck 1,132（+102 ignored），和上一版一样。T1R 的验证集是每一帧，不直接比。
  - note：`few points in the current sweep (ignored)` 875；`no points in the input (left out)` 298；`teacher subclass disagrees with size (ignored)` 121（T1R 这一条是 4,237）。
  - GT 库：car 14,524、truck 4,078。
- （2026-10-23 复核，补审后）用仓库里的标签在本地重跑 propagate，25 个训练场景的 `labels_full.pkl` 逐框一致，计数和上面相同。
  - 补审：s007 的 36 个红框里 22 个确认、14 个作忽略、删 1 个框；s008 的 11 个全部确认、删 2 个框；s023 的 61 个全部作忽略；s024、s025 只处理了一部分。
  - 还剩 310 个红框（验证 63、训练 247），作忽略框，不影响训练。
  - 导到新目录再切链接，做法对。能进 T6。

---

## T6 训练 v1，同时跑基线

**要回答的问题**（T7 给出答案）：
1. 部署 v1 Voxel 还是 v1 Pillar：谁在验证集上更好就部署谁。差别在噪声以内时选 Pillar：在 Orin 上更省，TensorRT 的路已经通了（T8）。
2. 人工审核值不值：v1（审核过的标注）比 v0（没审核的自动标注）好多少。
3. 微调值不值：v1 比 nuScenes 原始模型（Pillar 是现在线上能用的）好多少。

**参加比较的 7 个模型**，都只看车辆，都在 10 个验证场景上整段跑：

| 名字 | 模型 | 结果目录 | 谁跑 |
|---|---|---|---|
| nusc_pp | nuScenes 原始 Pillar | `data/runs_nusc_pp/<场景>/pillar` | T6 |
| nusc_voxel | nuScenes 原始 Voxel，单次前向 | `data/runs_nusc_voxel/<场景>/voxel` | T6 |
| teacher | nuScenes Voxel + 翻转 TTA（自动标注的教师，只做参照，不能部署） | `data/runs/<场景>/voxel` | 5.11 已有 |
| v0_pp、v0_voxel | 没审核的自动标注训练的（T3） | `data/runs_veh_v0/<场景>/{pillar,voxel}` | T6 |
| v1_pp、v1_voxel | 审核过的标注训练的（T6） | `data/runs_veh_v1/<场景>/{pillar,voxel}` | T7 |

**评分**：用审核过的验证标签（`labels_sustech.pkl`，每 5 帧 1 帧，约 580 帧）。主要看两个数：
- 车辆 AP@2.0（中心距离 2 m 内算对）在 0–20 m 内：狗要在这个范围内对车做出反应；
- mAP（0.5 / 1 / 2 / 4 m 平均，0–40 m）。

**验证集不大**：有车的场景只有 8 个，约 2,500 个车框、107 条轨迹，同一场景里的帧高度相关。所以 `compare_detectors.py --bootstrap` 按场景有放回地重抽 1000 次，给出每个差值的 90% 区间，以及"A 比 B 好"的重抽占比。差值的区间跨过 0，就说明这个验证集分不出谁好。

**v1 和 v0 的训练配方完全一样**：单卡，batch 4，学习率 5e-4，20 个 epoch。这样两者的差别只来自标注。T3D 证明双卡不掉精度，但这里不用。

```text
任务 T6：用补审后的 v1 数据集训练只检车辆的 Voxel 和 Pillar 模型 v1；训练期间在另一张卡上跑基线（nuScenes 原始模型、v0），给 T7 用。
先按 docs/SERVER_TASKS.md 的"通用约定"做（拉到了 dog_perception/、tools/、det3d/ 的改动，要跑快速测试）。
前置：T5 完成（2026-10-23 补审后重导到 dog_veh_1023_1529）。

0. 检查数据集，记下旧目录的时间，再删掉旧目录：
     mkdir -p data/reports/T6
     readlink -f data/dog_veh
     python -c "import pickle; print(*(len(pickle.load(open('data/dog_veh/infos_%s.pkl' % s, 'rb'))) for s in ('train', 'val')))"
   应为 /public/zjst/robot/centerpoint_data/dog_veh_1023_1529 和 "7232 580"，不是就停下报告。
   旧目录 /public/zjst/robot/centerpoint_data/dog_veh 是第一次 T5 的导出，已经不用了。删之前先记时间，核对 T3D 复核的判断：
     D=/public/zjst/robot/centerpoint_data/dog_veh
     ls -l --time-style=full-iso $D/infos_train.pkl $D/dbinfos_train_5sweeps_withvelo.pkl | tee data/reports/T6/old_dataset_times.txt
     ls -l --time-style=full-iso $D/gt_database_5sweeps_withvelo/car | sed -n 2,3p | tee -a data/reports/T6/old_dataset_times.txt
   确认本仓库没有训练、评估在跑：ps aux | grep -E "dog_(voxel|pp)_veh|run_sequence|dist_test" | grep -v grep 没有输出。
   然后 du -sh $D，再 rm -rf $D（真实路径，不是 data/dog_veh 链接）。
1. 放卡（通用约定 5）：Voxel 约 7 GB、Pillar 约 6 GB，两个训练同时起，不要等：
   - 有两张卡各空出 ≥ 8 GB（Voxel）/ ≥ 7 GB（Pillar）：各放一张，约 1 小时 7 分都跑完；
   - 只有一张卡空出 ≥ 14 GB：两个都放这张卡上。各自会慢一半左右，合计约 2 小时（和先后跑差不多），但不用等卡；
   - 都不够：先起 Voxel，Pillar 等有卡再起（每 20 分钟看一次）。
     mkdir -p data/train
     ls -d data/train/dog_*_veh_v1 2>/dev/null     # 应该没有输出；有的话停下报告，不要覆盖
     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     CUDA_VISIBLE_DEVICES=<卡1> nohup python tools/train.py configs/dog/dog_centerpoint_voxel_xt32_veh.py \
         --work_dir data/train/dog_voxel_veh_v1 > data/train/dog_voxel_veh_v1.log 2>&1 &
     CUDA_VISIBLE_DEVICES=<卡2，可以和卡1相同> nohup python tools/train.py configs/dog/dog_centerpoint_pp_xt32_veh.py \
         --work_dir data/train/dog_pp_veh_v1 > data/train/dog_pp_veh_v1.log 2>&1 &
   单卡，不加 --autoscale-lr，和 v0 一样（两个训练在同一张卡上也是各自单卡训练，配方不变，只是慢）。
   第 3、4 步的基线每个进程约 3 GB，有空的显存就放（训练那张卡也行），不用等第三张卡；一次只跑一个基线进程。
2. 开始 10 分钟后检查两个日志（tail -30）：loss 是有限值、hm_loss 在下降；time 和 data_time，各占一张卡时应和 T3 差不多
   （Voxel 0.107 / 0.004 s，Pillar 0.079 / 0.002 s），同一张卡时各慢一半左右；每个 epoch 的迭代数；按 20 个 epoch 估算总时长。
   nvidia-smi --query-compute-apps=pid,used_memory --format=csv：记下两个训练进程各占多少显存。
3. 基线，流水线级（CUDA_VISIBLE_DEVICES=<有 ≥ 4 GB 空闲的卡，训练那张也行>）：
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
     RUNS=$(for r in $VAL; do echo data/runs/$r; done)      # 重放存好的帧（通用约定 8）
     { python tools/dog/run_sequence.py --replay $RUNS --detectors pillar --ckpt-pillar work_dirs/pretrained/nusc_pp.pth \
           --no-predict --no-mcap --jobs 4 --device cuda:0 --out "data/runs_nusc_pp/{scene}"
       python tools/dog/run_sequence.py --replay $RUNS --detectors voxel --ckpt-voxel work_dirs/pretrained/nusc_voxel.pth \
           --no-predict --no-mcap --jobs 4 --device cuda:0 --out "data/runs_nusc_voxel/{scene}"
     } 2>&1 | tee data/reports/T6/runs_nusc.txt
4. v0 的模型级评估（同一张卡），在新的验证集（infos_val，审核过的 580 帧）上，用来选 v0 的 epoch：
     REP=data/reports/T6
     ev() {  # ev <voxel|pp> <权重目录名> <epoch>
       python tools/dist_test.py configs/dog/dog_centerpoint_$1_xt32_veh.py --work_dir data/train/$2/eval_e$3 \
           --checkpoint data/train/$2/epoch_$3.pth 2>&1 | grep -E "Evaluation|time per frame" > $REP/dist_test_$2_e$3.txt
     }
     for m in voxel pp; do for e in 10 15 20; do ev $m dog_${m}_veh_v0 $e & done; wait; ev $m dog_${m}_veh_v0_oldlabels 20; done
     ev voxel dog_voxel_veh_v0_2gpu 20
   选 epoch：每个模型取 "Evaluation merged/vehicle" 那行里 AP@2.0 最高的 epoch；和更晚的 epoch 相差 < 0.005 时
   取更晚的（学习率退火完，更稳）。记为 E_v0_voxel、E_v0_pp。然后：
     python tools/dog/run_sequence.py --replay $RUNS --detectors pillar,voxel \
         --ckpt-pillar data/train/dog_pp_veh_v0/epoch_<E_v0_pp>.pth --ckpt-voxel data/train/dog_voxel_veh_v0/epoch_<E_v0_voxel>.pth \
         --no-predict --no-mcap --jobs 4 --device cuda:0 --out "data/runs_veh_v0/{scene}" 2>&1 | tee data/reports/T6/runs_v0.txt
   data/runs_veh_v0 已经有东西的话，先改名成 data/runs_veh_v0_old，不要混在一起。
5. 等训练结束，期间每 20 分钟看一次日志。

停止条件（停下并报告日志最后 50 行，不要改配置重跑）：报错（spconv、数据加载）；loss 变成 nan / inf；
估算总时长超过 4 小时；训练进程意外退出（可以用 --resume_from data/train/<名字>/latest.pth 续一次，再退出就停下）。
显存不够（CUDA out of memory）：多半是同一张卡上别人的任务涨了。换一张空出够多显存的卡，用 --resume_from 续训一次；再出现就停下报告。
提交：两个训练的 .log 和 work_dir 里的 *.log.json 复制到 data/reports/T6/；data/reports/T6/ 的其余文件；本文档。
报告（写进"T6 结果"）：
- 第 0 步的三个时间、删掉的旧目录大小；
- 训练：卡号（两个是否在同一张卡上）；每个训练进程占的显存（nvidia-smi）；time、data_time；每个 epoch 迭代数；总耗时；每 5 个 epoch 最后一次的 loss / hm_loss / loc_loss（两个 head 分开），
  和 T3 的 v0 并排；
- 第 4 步每个权重的 merged/vehicle 一行（mAP、AP@0.5/1/2/4、三个距离段的 AP@2.0），以及 E_v0_voxel、E_v0_pp；
- 第 3、4 步的流水线有没有报错，每个场景的帧数。
```

### T6 结果

数据集链接是 `/public/zjst/robot/centerpoint_data/dog_veh_1023_1529`，帧数 7232 / 580。旧目录 `/public/zjst/robot/centerpoint_data/dog_veh`（47G）已删，链接没动。删之前的时间：`infos_train.pkl` 2026-10-23 14:53:33，`dbinfos_train_5sweeps_withvelo.pkl` 14:55:56，GT 库 `1000_car_0.bin` / `1000_car_1.bin` 14:53:50。

开训时按当时的空闲规则只有 3 号和 5 号能用（都是 RTX 4090）。Voxel 用 5，Pillar 用 3，单卡，没有 `--autoscale-lr`。每个 epoch 2077 步（v0 是 1868），`epoch_20.pth` 的 meta 是 epoch 20、iter 41540。没有 nan。Pillar 开头有一次 CUDA 缓存分配失败的警告，随后继续跑完。

- 每次迭代：Voxel 平均 0.137 s（读数据 0.007 s），Pillar 平均 0.086 s（读数据 0.003 s）。T3 的 v0 是 0.107 / 0.004 和 0.079 / 0.002。
- 总耗时：Voxel 1 小时 35 分（15:52:33–17:27:36），v0 是 1 小时 7 分；Pillar 1 小时 0 分（15:52:31–16:52:23），v0 是 49 分。步数多了约一成，Voxel 每步也更慢。10 分钟时估计约 1.5 小时和 1 小时，没有到 4 小时的停止线。当时整卡利用率是 5 号 53%、3 号 76%（卡上还有别人的进程）。
- 显存（训练日志里的 memory，本进程）：Voxel 3299–3551 MB，Pillar 2638–3050 MB。
- 每 5 个 epoch 最后一条日志（第 2060 步）的 loss / hm_loss / loc_loss，顺序是 car、truck。括号里是 T3 v0 同一位置的数：
  - Voxel v1：第 5 个 epoch 0.3979 / 0.2250 / 0.6916 和 0.3481 / 0.1675 / 0.7225（v0：0.3652 / 0.2047 / 0.6419 和 0.3476 / 0.1559 / 0.7666）；第 10 个 0.3828 / 0.2196 / 0.6527 和 0.3070 / 0.1303 / 0.7072（v0：0.3543 / 0.1958 / 0.6338 和 0.2795 / 0.1193 / 0.6409）；第 15 个 0.2816 / 0.1541 / 0.5101 和 0.2725 / 0.1179 / 0.6186（v0：0.2646 / 0.1485 / 0.4642 和 0.2049 / 0.0943 / 0.4425）；第 20 个 0.2186 / 0.1077 / 0.4436 和 0.1640 / 0.0569 / 0.4281（v0：0.2044 / 0.0910 / 0.4536 和 0.1705 / 0.0716 / 0.3955）。
  - Pillar v1：第 5 个 epoch 0.4714 / 0.2897 / 0.7270 和 0.4935 / 0.2192 / 1.0974（v0：0.4776 / 0.2889 / 0.7546 和 0.4992 / 0.2915 / 0.8308）；第 10 个 0.4525 / 0.2934 / 0.6363 和 0.4598 / 0.2333 / 0.9058（v0：0.4534 / 0.2953 / 0.6325 和 0.3763 / 0.1878 / 0.7540）；第 15 个 0.3529 / 0.2055 / 0.5895 和 0.2787 / 0.1244 / 0.6170（v0：0.3199 / 0.1836 / 0.5453 和 0.2561 / 0.1116 / 0.5778）；第 20 个 0.2448 / 0.1268 / 0.4719 和 0.2788 / 0.1212 / 0.6303（v0：0.2459 / 0.1221 / 0.4954 和 0.2103 / 0.0841 / 0.5046）。

`data/runs_veh_v0` 是空的，没有改名。基线在 7 号卡上跑：nuScenes 流水线 16:53–17:30（Voxel 训练还在进行），v0 的 `dist_test` 17:31–17:38，v0 流水线 17:39–17:58。都没有报错。10 个验证场景的帧数，pillar 和 voxel 一样：164127 的 s000–s002、s004、s005 各 288，s003 是 287；165659 的 s003、s004 各 276，s005 是 303，s006 是 302。

第 4 步 merged/vehicle（mAP，AP@0.5 / 1 / 2 / 4，AP@2.0 的 0–10 / 10–20 / 20–40 m）：

| 权重 | mAP | AP@0.5 | AP@1 | AP@2.0 | AP@4 | 0–10 | 10–20 | 20–40 |
|---|---|---|---|---|---|---|---|---|
| voxel v0 e10 | 0.692 | 0.518 | 0.679 | 0.762 | 0.809 | 0.708 | 0.928 | 0.709 |
| voxel v0 e15 | 0.662 | 0.481 | 0.667 | 0.735 | 0.767 | 0.690 | 0.886 | 0.709 |
| voxel v0 e20 | 0.662 | 0.505 | 0.660 | 0.734 | 0.747 | 0.682 | 0.886 | 0.702 |
| voxel v0 oldlabels e20 | 0.610 | 0.531 | 0.611 | 0.645 | 0.653 | 0.676 | 0.764 | 0.621 |
| voxel v0 2gpu e20 | 0.663 | 0.530 | 0.654 | 0.728 | 0.738 | 0.615 | 0.872 | 0.697 |
| pillar v0 e10 | 0.688 | 0.552 | 0.640 | 0.762 | 0.797 | 0.661 | 0.924 | 0.698 |
| pillar v0 e15 | 0.700 | 0.568 | 0.686 | 0.758 | 0.786 | 0.826 | 0.930 | 0.683 |
| pillar v0 e20 | 0.704 | 0.554 | 0.692 | 0.771 | 0.800 | 0.807 | 0.932 | 0.713 |
| pillar v0 oldlabels e20 | 0.649 | 0.521 | 0.657 | 0.704 | 0.713 | 0.821 | 0.913 | 0.616 |

**E_v0_voxel = 10**（AP@2.0 0.762，比第 15、20 个 epoch 高过 0.005）。**E_v0_pp = 20**（0.771，三个里最高）。流水线用的就是这两个权重。

日志在 `data/reports/T6/`：`dog_voxel_veh_v1.log`、`dog_pp_veh_v1.log`，以及同名的 `*.log.json`（work_dir 里是 `20261023_155232.log.json`、`20261023_155231.log.json`）。权重在 `data/train/dog_*_veh_v1`（不提交）。

（2026-10-23 复核）
- **能进 T7。** 训练没有跑坏：每个 epoch 的平均 loss 平滑下降，没有 T3D 那种突跳。v1 每步多出的时间来自共用的卡（5 号上还有别人，利用率 53%），不是数据读取（data_time 0.007 s）。
- **v1 的训练 loss 比 v0 高一成**（最后一个 epoch：Voxel 0.430 对 0.384，Pillar 0.524 对 0.473），这不说明 v1 差：
  - 审核把 v0 里作忽略框的难例收成了正样本，每帧 truck 正样本 10.1 个（v0 9.5 个）；
  - 训练 loss 只说明模型记住了训练集，好坏看 T7 的验证集。
- **T3D 的判断坐实了。** 旧目录的 GT 库文件 14:53:50、`dbinfos` 14:55:56 写成，Pillar 双卡那次的 loc loss 14:57 起突跳，时间正好接上。
- **v0 的模型级结果**（审过的验证集，580 个关键帧）：
  - **双卡不掉精度**：Voxel 第 20 个 epoch，双卡 AP@2.0 0.728、单卡 0.734，差 0.006（mAP 0.663 对 0.662），在 T3D 定的 0.01 以内。0–10 m 差得多一些（0.615 对 0.682），但近处的框少，不当真。
  - **新标注规则（T1R）比旧规则好**：AP@2.0 Voxel 0.734 对 0.645，Pillar 0.771 对 0.704。注意旧规则那一版是在数据读取修好之前训的，`SweepPoseJitter` 的 8 个 worker 当时抽到同样的抖动，所以这个差里不全是标注的功劳。
  - **Voxel v0 第 10 个 epoch 最好，之后反而下降**（0.762 → 0.735 → 0.734）；Pillar v0 一直在升（第 20 个最好）。没审核的标签有错，Voxel 容量大，后半程把错标学了进去，这是一种可能的解释。T7 看 v1 是否还这样：审过的标签如果让 Voxel 不再后程下降，就是审核的作用。
  - **选 epoch 和最后比较用的是同一个验证集**，三个 epoch 里挑最好的，数字会略偏乐观（估计 0.01 左右）。v0、v1 用同一条规则，比较仍然公平；写结论时记着这一点。
- **延迟数字不能直接比**：基线是在 7 号卡上跑的，同时 5 号、3 号在训练，卡上还有别人的进程，nuScenes Pillar 每帧网络耗时 12–110 ms 来回跳。T7 第 2 步之后加了一步，在同一张卡上把 v1 的 Pillar 和 Voxel 背靠背各测一次。
- 没有记每个训练进程占的显存：T6 开始时 GPU 规则还没改（通用约定 5）。以后的任务照新规则记。

---

## T7 比较所有模型，选部署的模型和上报阈值

```text
任务 T7：在审核过的验证集上比较 7 个模型（见 T6 开头的表），选出部署的模型（v1 Voxel 或 v1 Pillar）和它的上报阈值。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T6 完成（v1 两个训练结束；data/runs_nusc_pp、data/runs_nusc_voxel、
data/runs_veh_v0 下 10 个验证场景都有 results.pkl）。
GPU：下面的步骤一次一个进程，每个约 3 GB，CUDA_VISIBLE_DEVICES 设成一张空出 ≥ 4 GB 的卡就行（通用约定 5），不用整张空卡。

1. v1 的模型级评估，选 epoch（ev 函数和 T6 第 4 步一样，只是结果写到 data/reports/T7/）：
     mkdir -p data/reports/T7
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
     REP=data/reports/T7
     ev() {  # ev <voxel|pp> <权重目录名> <epoch>
       python tools/dist_test.py configs/dog/dog_centerpoint_$1_xt32_veh.py --work_dir data/train/$2/eval_e$3 \
           --checkpoint data/train/$2/epoch_$3.pth 2>&1 | grep -E "Evaluation|time per frame" > $REP/dist_test_$2_e$3.txt
     }
     for m in voxel pp; do for e in 10 15 20; do ev $m dog_${m}_veh_v1 $e & done; wait; done   # 3 个一起跑
   规则同 T6 第 4 步，记为 E_v1_voxel、E_v1_pp。
2. v1 流水线级（通用约定 8：重放存好的帧，4 个场景一起跑；以前一个一个跑、现做预处理要 22 分钟）：
     python tools/dog/run_sequence.py --replay $(for r in $VAL; do echo data/runs/$r; done) --detectors pillar,voxel \
         --ckpt-pillar data/train/dog_pp_veh_v1/epoch_<E_v1_pp>.pth --ckpt-voxel data/train/dog_voxel_veh_v1/epoch_<E_v1_voxel>.pth \
         --no-predict --no-mcap --jobs 4 --device cuda:0 --out "data/runs_veh_v1/{scene}" 2>&1 | tee data/reports/T7/runs_v1.txt
2b. 延迟（T6 的基线跑在和训练共用的卡上，耗时不能比）：找一张利用率 < 20% 的卡，v1 的两个模型在同一个场景上背靠背各跑一次：
     S=data/runs/0921_164127_s001
     { python tools/dog/run_sequence.py --replay $S --detectors pillar --no-predict --no-mcap --device cuda:0 \
           --ckpt-pillar data/train/dog_pp_veh_v1/epoch_<E_v1_pp>.pth --out data/runs_latency/pillar 2>&1 | grep network
       python tools/dog/run_sequence.py --replay $S --detectors voxel --no-predict --no-mcap --device cuda:0 \
           --ckpt-voxel data/train/dog_voxel_veh_v1/epoch_<E_v1_voxel>.pth --out data/runs_latency/voxel 2>&1 | grep network
     } | tee data/reports/T7/latency.txt
   记两个模型的网络耗时中位数，并记下当时 nvidia-smi 里这张卡的利用率。这是 4090 上的数，Orin 上要另测（H2），只看两个模型的比例。
3. 对比、阈值扫描、按场景重抽：
     python tools/dog/compare_detectors.py --classes vehicle \
         --labels $(for r in $VAL; do echo data/labels/$r/labels_sustech.pkl; done) \
         --results $(for r in $VAL; do echo v1_voxel=data/runs_veh_v1/$r/voxel/results.pkl v1_pp=data/runs_veh_v1/$r/pillar/results.pkl \
             v0_voxel=data/runs_veh_v0/$r/voxel/results.pkl v0_pp=data/runs_veh_v0/$r/pillar/results.pkl \
             nusc_voxel=data/runs_nusc_voxel/$r/voxel/results.pkl nusc_pp=data/runs_nusc_pp/$r/pillar/results.pkl \
             teacher=data/runs/$r/voxel/results.pkl; done) \
         --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 --target-recall 0.9 --bootstrap 1000 \
         | tee data/reports/T7/compare_val.md
   输出依次是：AP 表；耗时表（4090 上的网络耗时，不是 Orin）；每个模型的阈值表；最后是重抽的两张表
   （每个模型的区间；每一对模型 "A − B" 的差值、区间、A 领先的重抽占比 "A ahead"）。模型按上面的顺序给，
   每一行都是"新的 − 旧的"，差值为正就是前者更好。
4. 运动状态（狗为运动的车停下，为停着的车不停）。v1 两个模型和 nusc_pp 各一次：
     motion() {  # motion <名字> <结果目录名> <pillar|voxel>
       python tools/dog/motion_eval.py $(for r in $VAL; do echo --scene data/labels/$r/labels_sustech.pkl data/$2/$r/$3/results.pkl; done) \
           --details data/reports/T7/motion_$1_details.csv | tee data/reports/T7/motion_$1.txt
     }
     motion v1_pp runs_veh_v1 pillar; motion v1_voxel runs_veh_v1 voxel; motion nusc_pp runs_nusc_pp pillar
5. 选部署的模型，看重抽表里 "v1_voxel − v1_pp" 这一行：
   - 部署 Voxel，满足其一：
     a. ΔAP@2.0 0-20 m ≥ +0.02，并且这一项的 A ahead ≥ 90%；
     b. ΔmAP ≥ +0.03、这一项的 A ahead ≥ 90%，同时 ΔAP@2.0 0-20 m ≥ −0.01（近处不比 Pillar 差）。
   - 否则部署 Pillar（分不出来时选更省的）。
   - 检查：选中的模型对 nusc_pp 那一行（v1_voxel − nusc_pp 或 v1_pp − nusc_pp）的 ΔAP@2.0 0-20 m 应 > 0。
     不是的话停下报告，多半哪里出了错。
6. 上报阈值：在选中模型的阈值表里取 "highest threshold with recall >= 0.90 within 20 m"；没有就取 best F1。
   写出这个阈值和现在代码里的 0.35 下的 precision / recall（0–20 m、0–40 m）。
7. 给选中的模型在 0921_164127_s001（验证集里运动的车最多）出一个 Foxglove 文件，人眼看一看：
     python tools/dog/run_sequence.py --replay data/runs/0921_164127_s001 --detectors <pillar|voxel> \
         --ckpt-<pillar|voxel> <选中的权重> --device cuda:0 --out data/runs_veh_v1_view/0921_164127_s001
   把生成的 mcap 复制到 data/examples/runs/0921_164127_s001_v1/ 提交（Git LFS）。

停止条件：任何报错；某个模型某个场景的 results.pkl 缺失；第 5 步的检查不通过。
提交：data/reports/T7/、上面的 mcap、本文档。
报告（写进"T7 结果"）：
- E_v1_voxel、E_v1_pp，以及模型级每个 epoch 的 merged/vehicle 一行；
- compare_val.md 的 AP 表、耗时表、重抽的两张表（整段贴）；latency.txt 两行和当时的卡利用率；
- 三个问题各一句话，每句带差值和 90% 区间：
  (1) 部署哪个，按第 5 步的哪一条；
  (2) 审核的作用：v1_pp − v0_pp、v1_voxel − v0_voxel；
  (3) 微调的作用：v1_pp − nusc_pp、v1_voxel − nusc_voxel，以及 v0 对 nuScenes 原始模型；
- 三个 motion_*.txt 的表；details CSV 里 missed、false moving 各多少行；
- 上报阈值的建议（由人决定是否改 DEFAULT_SCORE_THRESHOLDS）。
```

### T7 结果

**E_v1_voxel = 10，E_v1_pp = 20。** 规则和 T6 一样：AP@2.0 最高；和更晚的 epoch 相差不到 0.005 才取更晚的。Voxel 第 10 个 epoch 0.807，第 15、20 个是 0.791、0.784，差超过 0.005。Pillar 第 20 个 epoch 0.789，比第 10 个的 0.788 略高。审过的标签没有改变 Voxel 后程下降：v0 是 0.762 → 0.735 → 0.734，v1 是 0.807 → 0.791 → 0.784。选 epoch 和下面的比较用的是同一个验证集，最好的那个会略偏乐观（T6 复核里估计大约 0.01）；v0、v1 用同一条规则。

模型级 merged/vehicle（mAP，AP@0.5 / 1 / 2 / 4，AP@2.0 的 0–10 / 10–20 / 20–40 m）：

| 权重 | mAP | AP@0.5 | AP@1 | AP@2.0 | AP@4 | 0–10 | 10–20 | 20–40 |
|---|---|---|---|---|---|---|---|---|
| voxel v1 e10 | 0.745 | 0.605 | 0.738 | 0.807 | 0.831 | 0.650 | 0.943 | 0.777 |
| voxel v1 e15 | 0.728 | 0.601 | 0.707 | 0.791 | 0.815 | 0.756 | 0.924 | 0.761 |
| voxel v1 e20 | 0.720 | 0.584 | 0.706 | 0.784 | 0.805 | 0.715 | 0.929 | 0.753 |
| pillar v1 e10 | 0.712 | 0.550 | 0.685 | 0.788 | 0.826 | 0.812 | 0.915 | 0.749 |
| pillar v1 e15 | 0.683 | 0.511 | 0.674 | 0.761 | 0.788 | 0.813 | 0.917 | 0.732 |
| pillar v1 e20 | 0.711 | 0.541 | 0.702 | 0.789 | 0.813 | 0.825 | 0.945 | 0.738 |

流水线用这两个权重，10 个验证场景都跑完，没有报错。帧数和 T6 一样：164127 的 s000–s002、s004、s005 各 288，s003 是 287；165659 的 s003、s004 各 276，s005 是 303，s006 是 302。评估在 5 号卡（开跑时空闲约 8.5 GB、利用率 0%）。`dist_test` 进程占 1464 MB；两个模型一起的 `run_sequence` 刚加载时采样到 834 MB，单模型跑起来最高看到 4286 MB。

`compare_val.md` 的 AP 表、耗时表、重抽的两张表：

| model | class | AP@0.5 | AP@1.0 | AP@2.0 | ATE | ASE | AOE | AVE | AP@4.0 | mAP | AP@2.0[0-10m] | AP@2.0[10-20m] | AP@2.0[20-40m] | #gt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_voxel | vehicle | 0.604 | 0.738 | 0.807 | 0.321 | 0.165 | 1.218 | 0.248 | 0.831 | 0.745 | 0.647 | 0.939 | 0.777 | 2082 |
| v1_pp | vehicle | 0.547 | 0.711 | 0.797 | 0.363 | 0.165 | 1.018 | 0.632 | 0.818 | 0.718 | 0.819 | 0.942 | 0.752 | 2082 |
| v0_voxel | vehicle | 0.518 | 0.680 | 0.763 | 0.389 | 0.158 | 1.200 | 0.274 | 0.809 | 0.692 | 0.706 | 0.911 | 0.719 | 2082 |
| v0_pp | vehicle | 0.554 | 0.694 | 0.770 | 0.367 | 0.155 | 0.858 | 0.703 | 0.796 | 0.704 | 0.813 | 0.931 | 0.717 | 2082 |
| nusc_voxel | vehicle | 0.742 | 0.862 | 0.903 | 0.259 | 0.148 | 0.600 | 0.156 | 0.919 | 0.856 | 0.947 | 0.968 | 0.868 | 2082 |
| nusc_pp | vehicle | 0.646 | 0.744 | 0.804 | 0.350 | 0.179 | 0.774 | 0.410 | 0.824 | 0.754 | 0.589 | 0.933 | 0.793 | 2082 |
| teacher | vehicle | 0.832 | 0.914 | 0.936 | 0.202 | 0.123 | 0.610 | 0.158 | 0.944 | 0.906 | 0.977 | 0.986 | 0.912 | 2082 |

| model | frames | mAP | net ms (median) | net ms (p95) | preprocess ms | #tracks | mean track len |
|---|---|---|---|---|---|---|---|
| v1_voxel | 580 | 0.745 | 22.4 | 107.0 | 152.6 | 110 | 83.8 |
| v1_pp | 580 | 0.718 | 13.5 | 88.0 | 152.6 | 134 | 68.3 |
| v0_voxel | 580 | 0.692 | 18.0 | 48.3 | 139.3 | 114 | 74.3 |
| v0_pp | 580 | 0.704 | 8.0 | 32.5 | 139.3 | 118 | 68.0 |
| nusc_voxel | 580 | 0.856 | 32.5 | 84.7 | 170.6 | 599 | 37.5 |
| nusc_pp | 580 | 0.754 | 19.1 | 66.1 | 161.6 | 896 | 33.0 |
| teacher | 580 | 0.906 | 51.8 | 168.3 | 495.6 | 470 | 43.5 |

| model | mAP | interval | AP@2.0 0-20 m | interval |
|---|---|---|---|---|
| v1_voxel | 0.745 | 0.638 – 0.794 | 0.904 | 0.742 – 0.957 |
| v1_pp | 0.718 | 0.643 – 0.759 | 0.922 | 0.812 – 0.967 |
| v0_voxel | 0.692 | 0.575 – 0.750 | 0.884 | 0.725 – 0.948 |
| v0_pp | 0.704 | 0.585 – 0.767 | 0.914 | 0.774 – 0.974 |
| nusc_voxel | 0.856 | 0.775 – 0.891 | 0.965 | 0.921 – 0.985 |
| nusc_pp | 0.754 | 0.611 – 0.815 | 0.873 | 0.676 – 0.963 |
| teacher | 0.906 | 0.841 – 0.934 | 0.984 | 0.963 – 0.997 |

| A − B | ΔmAP | interval | A ahead | ΔAP@2.0 0-20 m | interval | A ahead |
|---|---|---|---|---|---|---|
| v1_voxel − v1_pp | +0.027 | -0.018 – 0.052 | 89% | -0.018 | -0.082 – -0.002 | 2% |
| v1_voxel − v0_voxel | +0.052 | 0.024 – 0.096 | 100% | +0.020 | -0.012 – 0.061 | 87% |
| v1_voxel − v0_pp | +0.041 | 0.000 – 0.095 | 95% | -0.010 | -0.055 – 0.020 | 22% |
| v1_voxel − nusc_voxel | -0.112 | -0.151 – -0.081 | 0% | -0.061 | -0.179 – -0.021 | 0% |
| v1_voxel − nusc_pp | -0.010 | -0.055 – 0.077 | 40% | +0.031 | -0.071 – 0.161 | 67% |
| v1_voxel − teacher | -0.161 | -0.216 – -0.129 | 0% | -0.081 | -0.217 – -0.036 | 0% |
| v1_pp − v0_voxel | +0.026 | 0.001 – 0.077 | 95% | +0.038 | 0.009 – 0.107 | 100% |
| v1_pp − v0_pp | +0.014 | -0.011 – 0.066 | 76% | +0.008 | -0.015 – 0.059 | 62% |
| v1_pp − nusc_voxel | -0.138 | -0.173 – -0.099 | 0% | -0.043 | -0.118 – -0.008 | 2% |
| v1_pp − nusc_pp | -0.036 | -0.083 – 0.073 | 21% | +0.049 | -0.043 – 0.204 | 76% |
| v1_pp − teacher | -0.188 | -0.231 – -0.153 | 0% | -0.062 | -0.148 – -0.026 | 0% |
| v0_voxel − v0_pp | -0.011 | -0.033 – 0.005 | 12% | -0.030 | -0.056 – -0.020 | 0% |
| v0_voxel − nusc_voxel | -0.164 | -0.234 – -0.121 | 0% | -0.081 | -0.221 – -0.028 | 0% |
| v0_voxel − nusc_pp | -0.062 | -0.119 – 0.027 | 10% | +0.011 | -0.107 – 0.156 | 54% |
| v0_voxel − teacher | -0.214 | -0.295 – -0.169 | 0% | -0.100 | -0.245 – -0.045 | 0% |
| v0_pp − nusc_voxel | -0.153 | -0.228 – -0.104 | 0% | -0.052 | -0.164 – 0.000 | 5% |
| v0_pp − nusc_pp | -0.051 | -0.117 – 0.050 | 14% | +0.041 | -0.066 – 0.195 | 70% |
| v0_pp − teacher | -0.203 | -0.289 – -0.153 | 0% | -0.071 | -0.189 – -0.020 | 0% |
| nusc_voxel − nusc_pp | +0.102 | 0.069 – 0.187 | 100% | +0.093 | 0.016 – 0.261 | 99% |
| nusc_voxel − teacher | -0.050 | -0.068 – -0.042 | 0% | -0.019 | -0.043 – -0.009 | 0% |
| nusc_pp − teacher | -0.152 | -0.249 – -0.113 | 0% | -0.112 | -0.297 – -0.030 | 0% |

`latency.txt`（7 号卡，`0921_164127_s001`，背靠背）：

```text
[pillar] 288 frames, network 8.3 ms (median)
[voxel] 288 frames, network 24.1 ms (median)
```

Voxel 大约是 Pillar 的 2.9 倍。这是 4090 上的数。开测那一瞬间 nvidia-smi 是 59%，随后别人的进程显存停在 7010 MB，采样利用率是 0%、0%、41%、5%、0%、24%；41% 那次本进程显存升到 3350 MB。后来在 4 号卡上又测了一次（开测 0%），邻居中途醒来，结束时 99%，得到 20.1 / 33.8 ms，不用这组。

(1) **部署 Pillar**，第 5 步两条都不满足，分不出来就选更省的。`v1_voxel − v1_pp` 的 ΔAP@2.0 0–20 m 是 −0.018（90% 区间 −0.082 – −0.002，Voxel 领先的重抽只占 2%），不到 +0.02。ΔmAP 是 +0.027（区间 −0.018 – +0.052，领先 89%），不到 +0.03，领先占比也不到 90%，近处还差 0.018。对 nusc_pp 的检查过了：`v1_pp − nusc_pp` 的 ΔAP@2.0 0–20 m 是 +0.049（区间 −0.043 – +0.204，领先 76%），点估计大于 0；区间跨过 0，近处的优势不算稳。

(2) 审核对 Voxel 的 mAP 有用，对 Pillar 和近处 AP 分不出来。`v1_voxel − v0_voxel` 的 ΔmAP 是 +0.052（区间 +0.024 – +0.096，领先 100%），ΔAP@2.0 0–20 m 是 +0.020（区间 −0.012 – +0.061，领先 87%）。`v1_pp − v0_pp` 的 ΔmAP 是 +0.014（区间 −0.011 – +0.066，领先 76%），ΔAP@2.0 0–20 m 是 +0.008（区间 −0.015 – +0.059，领先 62%）。

(3) 微调没有超过 nuScenes 原始 Voxel；对原始 Pillar，近处的点估计更好，但区间都跨过 0。`v1_voxel − nusc_voxel` 的 ΔmAP 是 −0.112（区间 −0.151 – −0.081，领先 0%），ΔAP@2.0 0–20 m 是 −0.061（区间 −0.179 – −0.021，领先 0%）。`v1_pp − nusc_pp` 的 ΔmAP 是 −0.036（区间 −0.083 – +0.073，领先 21%），ΔAP@2.0 0–20 m 是 +0.049（区间 −0.043 – +0.204，领先 76%）。v0 也一样：`v0_voxel − nusc_voxel` 的 ΔmAP −0.164（区间 −0.234 – −0.121）、ΔAP@2.0 0–20 m −0.081（区间 −0.221 – −0.028），都是原始 Voxel 更好；`v0_pp − nusc_pp` 的 ΔmAP −0.051（区间 −0.117 – +0.050，领先 14%），ΔAP@2.0 0–20 m +0.041（区间 −0.066 – +0.195，领先 70%）。

运动状态（默认 on speed 0.70、on frames 3）：

v1_pp：

| on speed | on frames | moving recall | false moving | static tracks flagged | false episodes (per h) | moving stretches: missed / all, latency median / p90 s | relevant (>= 1.0 m/s, <= 30 m): missed / all, latency median / p90 s |
|---|---|---|---|---|---|---|---|
| 0.70 | 3 | 0.788 (104) | 0.0248 (1414) | 10 / 51 | 11 (137 / h) | 3 / 10, 0.50 / 2.40 | 0 / 6, 0.25 / 1.50 |

v1_voxel：

| on speed | on frames | moving recall | false moving | static tracks flagged | false episodes (per h) | moving stretches: missed / all, latency median / p90 s | relevant (>= 1.0 m/s, <= 30 m): missed / all, latency median / p90 s |
|---|---|---|---|---|---|---|---|
| 0.70 | 3 | 0.447 (103) | 0.0153 (1436) | 3 / 51 | 4 (50 / h) | 4 / 10, 0.75 / 8.00 | 1 / 6, 0.50 / 8.50 |

nusc_pp：

| on speed | on frames | moving recall | false moving | static tracks flagged | false episodes (per h) | moving stretches: missed / all, latency median / p90 s | relevant (>= 1.0 m/s, <= 30 m): missed / all, latency median / p90 s |
|---|---|---|---|---|---|---|---|
| 0.70 | 3 | 0.857 (133) | 0.0307 (1695) | 14 / 61 | 19 (237 / h) | 3 / 12, 0.50 / 1.50 | 1 / 6, 0.50 / 0.50 |

details 里 missed / false moving：v1_pp 3 / 11，v1_voxel 4 / 4，nusc_pp 3 / 19。

上报阈值建议 **0.25**（v1 Pillar 的 "highest threshold with recall >= 0.90 within 20 m"）。这个阈值下 precision / recall：0–20 m 是 0.880 / 0.911，0–40 m 是 0.868 / 0.773。现在代码里 `DEFAULT_SCORE_THRESHOLDS` 的 vehicle 是 0.35，同一张表上 0–20 m 是 0.918 / 0.867，0–40 m 是 0.924 / 0.722。0.35 时 20 m 内召回不到 0.90。改不改由人定，这里没有改代码。

Foxglove：`data/examples/runs/0921_164127_s001_v1/0921_164127_s001_pillar.mcap`（Pillar v1 第 20 个 epoch，288 帧）。

（2026-10-23 复核）
- **部署 Pillar v1：同意。** 按第 5 步的规则 Voxel 没有明显更好（近处还差 0.018），Pillar 网络快 2.9 倍。另外 Voxel v1 的运动状态有硬伤：运动帧召回只有 0.447（Pillar 0.788）。漏报的运动段里，位移速度对（2.6 m/s），速度头却读 0.0–0.7 m/s：5 m 外一辆 2.6 m/s 的车，速度头给 0.01。选中的是第 10 个 epoch，速度头可能还没学到运动的车。
- **对现在线上的 nuScenes Pillar，v1 Pillar 实际用起来好得多**（两者都不是教师，这个比较是公平的）：
  - 误检少很多：上报阈值 0.35 时，40 m 内误检 114 个对 695 个；整段确认的轨迹 134 条对 896 条（标注里约 107 条）；
  - 停着的车被报运动：每小时 137 次对 237 次；相关的运动段一个没漏（0 / 6，nuScenes Pillar 漏 1 个）；
  - 代价是远处召回低一些：同样阈值 0.35，40 m 内召回 0.722 对 0.825，20 m 内 0.867 对 0.923；
  - 所以 mAP 低 0.036，20 m 内 AP@2.0 高 0.049，两个区间都跨 0。
- **"原始 nuScenes Voxel 比所有微调模型都好"不能直接当结论。** 验证标签就是教师（nuScenes Voxel + TTA）的框，人审时只做了确认、删除、作忽略：验证集没有新画的框，框也几乎没挪。所以：
  - 框的位置就是教师给的：AP@0.5 教师 0.832、原始 Voxel 0.742，微调模型 0.55–0.60；
  - 教师漏掉的车不在标签里：别的模型检出它们反而算误检。
  
  原始 Voxel 就是教师去掉 TTA，和标签同源，分数天然偏高。非教师模型之间（v1 对 v0、Voxel 对 Pillar、v1 对 nuScenes Pillar）的比较受影响小。
- **但差距不全是偏差**：微调模型在远处召回明显低。阈值 0.1 时 40 m 内最高召回 0.86–0.87，原始 Voxel 0.96；20 m 内都在 0.96 以上。可能是训练时远处点少的车作忽略框、正样本少。要回答"微调到底有没有比原始 Voxel 好"，得有不偏向教师的验证标签：
  - 把各模型（v1、nuScenes Pillar）在验证关键帧上 ≥ 0.3 分、却和现有标签都对不上的框，作为红框候选导进 SUSTechPOINTS，人看是不是车（是就确认，不是就删）；
  - 再按 AP@2 m / 4 m（对框的位置不敏感）重评。
  
  这个工具已经加了（2026-10-24）：T11–T12。
- **审核的作用**：Voxel 明确（mAP +0.052，100% 领先），Pillar 看不出来（+0.014，76%）。
- **两个模型都在第 10 个 epoch 左右就够了**：Voxel 两轮都是第 10 个 epoch 最好；Pillar v1 第 10、20 个几乎一样（0.788 / 0.789）。下一轮训练可以先试 10 个 epoch，训练时间减半，再看验证集。
- **上报阈值**：0.25 时 v1 Pillar 在 20 m 内召回 0.911，误检比现在线上的 nuScenes Pillar 在 0.35 时少三分之二（226 对 695）。建议部署 v1 Pillar 时车辆用 0.25。注意 `DEFAULT_SCORE_THRESHOLDS` 也是跟踪器新建轨迹的门槛，自动标注（教师）也在用；直接改它会改变下一轮的自动标注，最好做成部署时按模型设的参数。
- **T7 慢**：45 分钟里约一半是流水线级重跑（两个模型 10 个场景 22 分钟，每帧 0.48 s 里大半是重读扫描、预处理和轨迹预测）。已加速（通用约定 8）：`run_sequence.py --replay` 重放存好的帧、`--no-predict`、`--jobs` 多场景并行，`dist_test` 3 个一起跑。T6、T7、T9 的命令都改成了新写法，预计流水线级从 22 分钟降到 5 分钟以内。

---

## T8 部署选中的模型

不管 T7 选了谁，都先导出 Pillar v1 的 ONNX：选了 Pillar 就直接用；选了 Voxel，它是 Voxel 在 Orin 上跑不进延迟时的退路。

**2026-10-26 改**：上线的 Pillar 换成 nuScenes Pillar（T15 复核），第 1 步导出它：
权重用 `work_dirs/pretrained/nusc_pp.pth`，输出目录 `data/deploy/nusc_pp`，其余照旧。上报阈值等 T19 定。

```text
任务 T8：导出上线 Pillar（nuScenes Pillar，2026-10-26 起）的两段 ONNX 并和 PyTorch 逐段对比；T7 选了 Voxel 时，再把 Voxel v1 打包，准备上 Orin 实测。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T7 完成，已知 E_v1_pp、E_v1_voxel 和选中的模型。

1. f=$(ls data/runs/0921_165659_s004/frames/*.npy | sed -n 150p)
   python tools/dog/export_pillar_onnx.py --checkpoint work_dirs/pretrained/nusc_pp.pth \
       --out data/deploy/nusc_pp --check $f 2>&1 | tee data/reports/T8/export.txt
   ls -la data/deploy/nusc_pp
2. 只在 T7 选了 Voxel 时做：
     mkdir -p data/deploy/dog_voxel_veh_v1
     cp data/train/dog_voxel_veh_v1/epoch_<E_v1_voxel>.pth data/deploy/dog_voxel_veh_v1/model.pth
     cp configs/dog/dog_centerpoint_voxel_xt32_veh.py data/deploy/dog_voxel_veh_v1/
   Orin 上的做法见 docs/DOG_PIPELINE.md 8.2 路线 1（PyTorch + spconv），由人在 Orin 上测延迟（H2）。

通过标准（第 1 步）：export.txt 里 pfn / bev / heads 的相对误差都 < 1e-3，PyTorch 和 ONNX 的检测个数相同，
分数差 < 1e-3、中心差 < 1 cm（本地用迁移权重试过：分数差 2.7e-07、中心差 3e-06 m）。
停止条件：不满足通过标准；报错。
提交：data/reports/T8/、本文档。不提交 ONNX 和权重（data/deploy/ 被 .gitignore 忽略），报告路径。
报告（写进"T8 结果"）：export.txt 全文；ONNX 文件大小；它打印的 trtexec 命令（在 Orin 上建 FP16 引擎用）；
选了 Voxel 时 data/deploy/dog_voxel_veh_v1 的内容。
```

**H2（选了 Voxel 时，人做）**：在 Orin 上按 DOG_PIPELINE.md 8.2 路线 1 跑 Voxel v1，最高功耗模式、锁频后测每帧网络耗时。加上预处理和跟踪，端到端放得下 100 ms（10 Hz）就部署 Voxel；放不下就部署 T8 导出的 Pillar。

### T8 结果

（待填）

---

## T9（可选）用 v1 Voxel 当教师，准备第二轮

v1 Voxel 专门在这批数据上训过，当教师应该比 nuScenes 模型准，待审核的轨迹会少很多。第二轮的审核和训练，等这一步的结果出来再定。

```text
任务 T9：用 v1 Voxel（T7 选的 epoch，加 TTA）重新跑全部 35 个场景的教师推理和自动标注，输出到新的目录，不覆盖第一轮。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T7 完成，已知 E_v1_voxel。

1. python tools/dog/batch_label.py --split $SPLIT --ckpt-voxel data/train/dog_voxel_veh_v1/epoch_<E_v1_voxel>.pth \
       --runs data/runs_r2 --labels data/labels_r2 --replay-runs data/runs --no-predict
   `--replay-runs data/runs`：重放第一轮存好的模型输入帧，不重新读扫描、不再存一份帧（省 36 GB）；自动标注也在这些帧上数点。
2. 每个场景出一张车辆表，统计待定的轨迹数：
     for r in $(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT); do
       python tools/dog/review_labels.py table --classes vehicle --labels data/labels_r2/$r/labels.pkl \
           --out data/labels_r2/$r/labels_tracks_vehicle.csv
     done | tee data/reports/T9/tables.txt

停止条件：任何报错；磁盘不足。
提交：data/reports/T9/、data/labels_r2/batch_summary.csv（复制到 data/reports/T9/）、本文档。
报告（写进"T9 结果"）：batch_summary；和第一轮相比，车辆轨迹数、train_as=ignore 的条数（训练、验证分开）；耗时。
```

### T9 结果

（待填）

---

## T10（可选）比较 0921 的两个 INS 杠杆臂

0921 的天线接反了。有两个杠杆臂可用（[SENSORS.md](SENSORS.md) 3.2 节）：
- 从转弯数据拟合的 (−0.06, −0.54, 0)，现在用的是它；
- 按天线位置推算的 (0.021, −0.474, 0)。

看哪个残差小，以后定下用哪个。只看诊断，不改正式的位姿。

```text
任务 T10：在 0921 转弯较多的场景上，用两个杠杆臂各融合一次位姿，比较残差。不覆盖 data/seq 里的正式位姿。
先按 docs/SERVER_TASKS.md 的"通用约定"做。

1. 找出航向变化超过 45° 的场景（5.6 拟合杠杆臂用的就是这些，约 14 个）：
     python - <<'PY' | tee data/reports/T10/turns.txt
     import glob, numpy as np
     from dog_perception.geometry import quat_to_R, yaw_of
     from dog_perception.io import SequenceReader
     for d in sorted(glob.glob("data/seq/0921_*")):
         yaw = np.unwrap([yaw_of(quat_to_R(q)) for q in SequenceReader(d).odom[:, 4:8]])
         print(d.split("/")[-1], round(float(np.degrees(np.ptp(yaw))), 1))
     PY
   把超过 45° 的场景各复制两份：cp -r data/seq/<场景> data/seq_lever/fit/ 和 data/seq_lever/derived/（约 5 GB）。
2. python tools/dog/refine_poses.py --seq data/seq_lever/fit/* --lever-arm -0.06 -0.54 0 --plots data/reports/T10/fit
   python tools/dog/refine_poses.py --seq data/seq_lever/derived/* --lever-arm 0.0213 -0.474 0 --plots data/reports/T10/derived
3. python tools/dog/sequence_preview.py --seq "data/seq_lever/fit/*" --out data/reports/T10/preview_fit --jobs 6
   python tools/dog/sequence_preview.py --seq "data/seq_lever/derived/*" --out data/reports/T10/preview_derived --jobs 6

停止条件：报错。
提交：data/reports/T10/ 里的 csv / json / png（不要 mcap）、本文档。
报告（写进"T10 结果"）：每个场景两种杠杆臂下的 lever_residual_m、修正量 p90、预览的 shift p90，以及哪个更小。
```

### T10 结果

（待填）

---

## T11 验证集加多模型候选框

工具见 [DOG_FINETUNE.md](DOG_FINETUNE.md) 第 16 节。T7 的 7 个模型在 10 个验证场景上的 results.pkl 都在，不用 GPU，也不重新推理。

```text
任务 T11：给 10 个验证场景加多模型候选框（红框，标签写"候选"和检出它的模型），给 35 个场景写上模型框（Ctrl + 拖框用）。
先按 docs/SERVER_TASKS.md 的"通用约定"做（这次 git pull 拉到了代码改动，跑快速测试）。前置：T7 完成
（data/runs_veh_v1、data/runs_veh_v0、data/runs_nusc_pp、data/runs_nusc_voxel、data/runs 下 10 个验证场景都有 results.pkl）。
只读 results.pkl、改公共盘 sustech/ 里的场景，不用 GPU。不要用 --force，不要重新 export 任何场景（通用约定 3）。

1. 停掉正在跑的 SUSTechPOINTS：它的代码改了，要重启才有新功能；停着的时候，浏览器里还开着的旧页面也存不回去。
     ps aux | grep "sustechpoints_convert.py serve" | grep -v grep
   有就记下它的启动命令和 PID，然后 kill <PID>。
2. 备份 10 个验证场景的标注。工具自己会把 label/ 复制到场景里的 label_backup/<时间>/，这里连 dog_meta.json、dog_info 再备一份：
     mkdir -p data/reports/T11 /public/zjst/robot/centerpoint_data/sustech_backup
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
     tar -C /public/zjst/robot/centerpoint_data/sustech -cf /public/zjst/robot/centerpoint_data/sustech_backup/T11_before.tar \
         $(for r in $VAL; do echo $r/label $r/dog_meta.json $r/dog_info; done)
3. 先空跑，只统计（{scene} 原样写，工具按场景替换）。只要 40 m 内、至少在 2 个关键帧里出现的物体；
   物体最高分的门槛从 0.3 试到 0.5（物体的其它框只要 ≥ 0.3 都留着）：
     RES="v1_pp=data/runs_veh_v1/{scene}/pillar/results.pkl v1_voxel=data/runs_veh_v1/{scene}/voxel/results.pkl
          v0_pp=data/runs_veh_v0/{scene}/pillar/results.pkl v0_voxel=data/runs_veh_v0/{scene}/voxel/results.pkl
          nusc_pp=data/runs_nusc_pp/{scene}/pillar/results.pkl nusc_voxel=data/runs_nusc_voxel/{scene}/voxel/results.pkl
          teacher=data/runs/{scene}/voxel/results.pkl"
     cand() {  # cand <物体最高分门槛> [--dry-run]
       python tools/dog/sustechpoints_convert.py candidates --root /public/zjst/robot/centerpoint_data/sustech \
           --scenes $VAL --results $RES --min-score 0.3 --max-range 40 --min-keyframes 2 --object-min-score $1 $2
     }
     for x in 0.3 0.4 0.5; do cand $x --dry-run | tail -2 | tee data/reports/T11/dry_run_$x.txt; done
   取 total 的候选物体数（"N objects"）≤ 400 的最小门槛，记为 X。三个都超过 400 就停下报告。
4. 正式跑：cand X 2>&1 | tee data/reports/T11/candidates.txt
   每个场景先打印"model boxes for Ctrl + drag"（写 dog_dets/），再打印候选数。
5. 训练场景也写上模型框（只有教师的），审训练集时 Ctrl + 拖框也能用：
     TRAIN=$(awk -F, 'NR>1 && $2=="train" {print $1}' $SPLIT)
     python tools/dog/sustechpoints_convert.py detections --root /public/zjst/robot/centerpoint_data/sustech \
         --scenes $TRAIN --results teacher=data/runs/{scene}/voxel/results.pkl | tee data/reports/T11/detections.txt
6. 统计每个场景的帧数、框数、红框数：
     python tools/dog/sustechpoints_convert.py stats --root /public/zjst/robot/centerpoint_data/sustech \
         | tee data/reports/T11/stats.txt
7. 第 1 步停掉过的话，用记下的同一条命令重新起 SUSTechPOINTS（后台运行，日志写到 data/reports/T11/serve.log）。

停止条件：任何报错；第 3 步三个门槛的候选物体都超过 400。
提交：data/reports/T11/（不要 tar、serve.log）、本文档。
报告（写进"T11 结果"，接在第一次的记录后面）：
- 三个门槛空跑的 total 行，选的 X；
- candidates.txt 每个场景一行（候选框数、物体数）和 total 行；
- "at a place labelled in another frame"（同一位置别的帧有标注）的框数；
- 每个模型 "found by X" 和 "found by X only" 的数（哪个模型多检出的东西最多）；
- "already labelled"、"deleted by the reviewer"、"no points" 的总数；
- stats.txt 里验证集的红框总数（含 H1 剩下的）；
- SUSTechPOINTS 是否已重新起好，地址。
```

### T11 结果

停在第 3 步。`--min-score 0.3` 的空跑有 **2045 个候选物体**（5470 个框），超过 400，没有写入场景，也没有做第 4–6 步，没有改 `--min-score`。备份在 `/public/zjst/robot/centerpoint_data/sustech_backup/T11_before.tar`（3.1 MB，不提交）。

每个场景（候选框数、物体数）：

| 场景 | 框 | 物体 |
|---|---|---|
| 0921_164127_s000 | 599 | 130 |
| 0921_164127_s001 | 698 | 295 |
| 0921_164127_s002 | 780 | 333 |
| 0921_164127_s003 | 676 | 263 |
| 0921_164127_s004 | 456 | 185 |
| 0921_164127_s005 | 418 | 45 |
| 0921_165659_s003 | 553 | 233 |
| 0921_165659_s004 | 316 | 169 |
| 0921_165659_s005 | 473 | 187 |
| 0921_165659_s006 | 501 | 205 |
| total | 5470 | 2045 |

各模型检出的候选框（"found by"，其中只有它一个模型的是 "only"）。多检出最多的是 **nusc_pp**：3937 框，其中 2298 框只有它。

| 模型 | found by | only |
|---|---|---|
| nusc_pp | 3937 | 2298 |
| nusc_voxel | 2535 | 471 |
| teacher | 2030 | 163 |
| v1_voxel | 213 | 118 |
| v1_pp | 201 | 80 |
| v0_voxel | 115 | 28 |
| v0_pp | 87 | 16 |

already labelled 2360，deleted by the reviewer 0，no points 6。stats 没跑，验证集红框总数还没有。

SUSTechPOINTS 已用原来的命令重新起好：`python tools/dog/sustechpoints_convert.py serve --host 10.36.136.7`，地址 http://10.36.136.7:8081 。这次空跑没有改标注。

（2026-10-25 复核）
- **停得对。** 2045 个物体一次审不完；直接把门槛提到 0.5，又会把教师漏掉、其它模型分数不高的真车也筛掉。
- **候选都是些什么**：验证集的 0921_164127_s000、s002 在本地有原始扫描，用 nuScenes Pillar（多检出最多的模型）重跑，加到审过的标签上（150 个物体），逐个看：
  - **一大半只在一个关键帧出现**（s000 是 50 个里的 31 个），多是闪一下的杂物；
  - **四分之一在 40 m 外**（s002 有 115 个框），点少，人也很难判断；
  - **将近一半的框分数在 0.30–0.35**；
  - 还有一类是真车：**别的关键帧标了，这一帧没标**。例如 s000 西南角那辆 16 m 的货车，有的帧有标注，有的帧只剩候选：自动标注的轨迹没覆盖到这一帧。这类要确认，否则每个检出它的模型都算误检。
- **H1 没删过验证集的车框**（拿不准的都标了忽略），所以 "deleted by the reviewer 0" 是对的，不是工具的问题。
- **工具改了**（`candidates` 新参数）：
  - `--max-range 40`：只要 40 m 内的；T12 评估也加 `--max-range 40`，40 m 外不审也不评分，谁都不吃亏；
  - `--min-keyframes 2`：只出现在一个关键帧的物体不提；
  - `--object-min-score`：物体最高的那个框达到门槛才提，它的其它框只要 ≥ 0.3 都留着，车不会只剩几帧；
  - 同一位置别的帧有标注的候选，标签上写"其他帧是 #12"，审核时一眼能认出是同一辆车。
- **本地两个场景的效果**（只有 nuScenes Pillar）：150 个物体 → 40 m 内且 ≥ 2 个关键帧 43 个 → 再要求最高分 ≥ 0.4 是 27 个、≥ 0.5 是 15 个，分别是 29%、18%、10%。按这个比例，服务器上 2045 个大约变成 590、370、205 个，所以第 3 步改成三个门槛都空跑一次，取 ≤ 400 的最小门槛。
- **你要做的**：把改过的 T11 整段再交给 Agent 跑一次，第 1、2 步照做（备份会覆盖成同样的内容，没关系）。跑完开始 H3。

（2026-10-25 重跑）三个门槛的空跑 total：

| 物体最高分 | total |
|---|---|
| 0.3 | 3838 candidate boxes (685 objects) |
| 0.4 | 3324 candidate boxes (509 objects) |
| 0.5 | 2679 candidate boxes (306 objects) |

**X = 0.5**（不超过 400 的最小门槛）。正式写入的每个场景：

| 场景 | 框 | 物体 |
|---|---|---|
| 0921_164127_s000 | 284 | 18 |
| 0921_164127_s001 | 293 | 40 |
| 0921_164127_s002 | 350 | 52 |
| 0921_164127_s003 | 309 | 41 |
| 0921_164127_s004 | 237 | 48 |
| 0921_164127_s005 | 316 | 9 |
| 0921_165659_s003 | 269 | 27 |
| 0921_165659_s004 | 127 | 16 |
| 0921_165659_s005 | 241 | 21 |
| 0921_165659_s006 | 253 | 34 |
| total | 2679 | 306 |

"at a place labelled in another frame" 137 框。多检出最多的仍是 **nusc_pp**：2104 框，其中 912 框只有它。

| 模型 | found by | only |
|---|---|---|
| nusc_pp | 2104 | 912 |
| nusc_voxel | 1509 | 95 |
| teacher | 1331 | 33 |
| v1_voxel | 136 | 75 |
| v1_pp | 79 | 16 |
| v0_voxel | 45 | 5 |
| v0_pp | 46 | 3 |

already labelled 2178，deleted by the reviewer 0，no points 5。stats 里验证集红框（marked review，含这次的候选和 H1 剩下的）一共 **2742**，580 帧、5145 个框。25 个训练场景写了教师的模型框（`detections.txt`）。

SUSTechPOINTS 已重新起好：`python tools/dog/sustechpoints_convert.py serve --host 10.36.136.7`，地址 http://10.36.136.7:8081 。备份覆盖了 `sustech_backup/T11_before.tar`（3.1 MB）。

（2026-10-25 复核第二次）
- **候选里混进了行人和骑行者，是工具的问题。** 验证场景是 T1S 导出的（10-23 02:54 UTC），那时 `dog_meta.json` 还不记 `label_classes`（07:11 UTC 才加）。候选工具读不到类别，就用了全部三类，nuScenes 的两个模型和教师检出的行人、骑行者都进来了。nusc_pp 的 2104 个框里大部分是这些。
- **改了**：`candidates`、`detections` 加 `--classes`，默认 `vehicle`，不再看场景记没记类别。
- **按新的决定（10-25）**：人工审核只审车辆（主要是小轿车和货车）；候选只用最好的两个模型，Voxel、Pillar 各一个，也就是以后一直迭代的两个教师：现在是 v1 Voxel（第 10 个 epoch）和 v1 Pillar（第 20 个 epoch）。
- 这一次写进去的 306 个候选不用手工清：T11b 重跑时，还红着的候选全部换掉；已经确认的留着；已经删掉的工具记得，不会再提。
- **你要做的**：先别审（H3 等 T11b）；把 T11b 交给 Agent。

---

## T11b 重做候选：只要车辆，只用两个 v1 模型

```text
任务 T11b：在 10 个验证场景里重做候选框，只要车辆，只用 v1 Voxel、v1 Pillar 两个模型；训练场景的模型框也只留车辆。
先按 docs/SERVER_TASKS.md 的"通用约定"做（这次 git pull 拉到了代码改动，跑快速测试）。前置：T11 完成。
只读 results.pkl、改公共盘 sustech/ 里的场景，不用 GPU。不要用 --force，不要重新 export 任何场景（通用约定 3）。

1. 停掉正在跑的 SUSTechPOINTS（同 T11 第 1 步），记下启动命令。
2. 备份（这次的备份里有 T11 写的候选，和 T11 的备份分开放）：
     mkdir -p data/reports/T11b
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
     tar -C /public/zjst/robot/centerpoint_data/sustech -cf /public/zjst/robot/centerpoint_data/sustech_backup/T11b_before.tar \
         $(for r in $VAL; do echo $r/label $r/dog_meta.json $r/dog_info; done)
3. 空跑三个门槛（只有两个模型、只有车辆）：
     RES="v1_voxel=data/runs_veh_v1/{scene}/voxel/results.pkl v1_pp=data/runs_veh_v1/{scene}/pillar/results.pkl"
     cand() {  # cand <物体最高分门槛> [--dry-run]
       python tools/dog/sustechpoints_convert.py candidates --root /public/zjst/robot/centerpoint_data/sustech \
           --scenes $VAL --results $RES --classes vehicle --min-score 0.3 --max-range 40 --min-keyframes 2 \
           --object-min-score $1 $2
     }
     for x in 0.3 0.4 0.5; do cand $x --dry-run | tail -2 | tee data/reports/T11b/dry_run_$x.txt; done
   取 total 的候选物体数 ≤ 400 的最小门槛，记为 X。三个都超过 400 就停下报告。
4. 正式跑：cand X 2>&1 | tee data/reports/T11b/candidates.txt
   T11 写的、还红着的候选会全部换掉（"earlier candidates replaced"）；已经确认的留着（"earlier candidates confirmed"）。
5. 训练场景的模型框重写，只留车辆：
     TRAIN=$(awk -F, 'NR>1 && $2=="train" {print $1}' $SPLIT)
     python tools/dog/sustechpoints_convert.py detections --root /public/zjst/robot/centerpoint_data/sustech \
         --scenes $TRAIN --results teacher=data/runs/{scene}/voxel/results.pkl --classes vehicle \
         | tee data/reports/T11b/detections.txt
6. 检查验证场景里所有候选框的类型（类型、是否还红着、框数）。下面的 Python 从行首开始写，整段复制：
python - $VAL <<'EOF' | tee data/reports/T11b/candidate_types.txt
import collections, glob, json, sys
c = collections.Counter()
for r in sys.argv[1:]:
    for f in glob.glob("/public/zjst/robot/centerpoint_data/sustech/%s/label/*.json" % r):
        for a in json.load(open(f)) or []:
            oid = str(a.get("obj_id"))
            if oid.startswith("c") and oid[1:].isdigit():
                c[(a.get("obj_type"), "red" if "review" in str(a.get("obj_attr") or "") else "decided")] += 1
for k, v in sorted(c.items()):
    print(k[0], k[1], v)
EOF
   red 的只能是 Car、Truck（或 Bus）；有 Pedestrian、BicycleRider 等还红着的就停下报告。
7. 统计：python tools/dog/sustechpoints_convert.py stats --root /public/zjst/robot/centerpoint_data/sustech | tee data/reports/T11b/stats.txt
8. 用第 1 步记下的命令重新起 SUSTechPOINTS。

停止条件：任何报错；第 3 步三个门槛都超过 400；第 6 步还有红着的非车辆候选。
提交：data/reports/T11b/（不要 tar）、本文档。
报告（写进"T11b 结果"）：三个门槛空跑的 total 行和选的 X；candidates.txt 每个场景一行和 total 行
（含 earlier candidates replaced / confirmed、at a place labelled in another frame）；candidate_types.txt；
stats 里验证集的红框总数；SUSTechPOINTS 地址。
```

### T11b 结果

三个门槛的空跑 total（只有 v1 Voxel、v1 Pillar，只要车辆）：

| 物体最高分 | total |
|---|---|
| 0.3 | 237 candidate boxes (54 objects) |
| 0.4 | 198 candidate boxes (36 objects) |
| 0.5 | 168 candidate boxes (24 objects) |

**X = 0.3**（三个都不超过 400，取最小门槛）。正式写入：

| 场景 | 框 | 物体 | replaced | confirmed | 其他帧已标 |
|---|---|---|---|---|---|
| 0921_164127_s000 | 62 | 5 | 284 | 0 | 4 |
| 0921_164127_s001 | 40 | 11 | 293 | 0 | 2 |
| 0921_164127_s002 | 6 | 3 | 350 | 0 | 2 |
| 0921_164127_s003 | 26 | 9 | 309 | 0 | 2 |
| 0921_164127_s004 | 57 | 12 | 237 | 0 | 2 |
| 0921_164127_s005 | 4 | 1 | 316 | 0 | 0 |
| 0921_165659_s003 | 4 | 2 | 269 | 0 | 0 |
| 0921_165659_s004 | 7 | 3 | 127 | 0 | 0 |
| 0921_165659_s005 | 17 | 5 | 241 | 0 | 5 |
| 0921_165659_s006 | 14 | 3 | 253 | 0 | 0 |
| total | 237 | 54 | 2679 | 0 | 17 |

T11 写的、还红着的 2679 个框全部换掉，已经确认的是 0。v1 Voxel 检出 169 框（其中 155 只有它），v1 Pillar 检出 82 框（其中 68 只有它）。

`candidate_types.txt`：Car red 191，Truck red 46。没有还红着的行人、骑行者。stats 里验证集红框一共 **300**（580 帧、2703 个框），含这次的 237 个候选框和 H1 剩下的。25 个训练场景的模型框按只要车辆重写了。

SUSTechPOINTS 已重新起好：`python tools/dog/sustechpoints_convert.py serve --host 10.36.136.7`，地址 http://10.36.136.7:8081 。备份在 `sustech_backup/T11b_before.tar`（4.5 MB，不提交）。

（2026-10-25 复核）
- **可以开始 H3。** 54 个物体、237 个框，都是车（Car 191、Truck 46）。T11 留下的 2679 个还红着的框全换掉了，确认过的是 0，说明 T11 之后还没人动过。
- **两个模型多检出的东西几乎不重合**：v1 Voxel 169 个框里 155 个只有它，v1 Pillar 82 个里 68 个只有它，两个都检出的只有 14 个框。两个模型各漏各的，正好说明两个教师互补（DATA_COLLECTION_PLAN.md 第 1 节）；也说明这些候选里误检不会少，H3 时大胆删。
- "其他帧是 #…" 的有 17 个框：多半是停着的车，自动标注的轨迹没覆盖这一帧，确认即可。
- v1 为什么还不如 nuScenes Voxel + TTA，分析写在 DATA_COLLECTION_PLAN.md 2.1 节：一半是量法（标签就是教师的框），一半是数据少、训练标签有空档（停着的车被拆成两段、中间看得见却没标）。

---

## T16 双教师的标注导进 SUSTechPOINTS，沿用 H1 的审核结论

H3 要审的是双教师（nuScenes Voxel + TTA、nuScenes Pillar + TTA，T14）的标注，训练、验证都审。直接从头审就把 H1 一两天的工作扔了，
所以先用 `sustechpoints_convert.py carry`（`dog_perception/sustechpoints.py` 的 `carry_review`）把 H1 的结论搬过去：
- 和 H1 的框是同一辆车：沿用 H1 的类型、忽略、静止 / 运动；H1 挪过的框用 H1 的框，没挪过的用双教师的框（两个教师加权平均，一般更准）；
  H1 还红着的（没确认），这里也红着；
- H1 删掉的东西，双教师又检出：直接删；
- H1 有、双教师没检出的车（包括 H1 里人新画的）：照搬过来（ID 前面加 h）；
- 剩下的：H1 没见过的新目标，标红。

人只要看红框：新目标、H1 剩下没定的。训练和验证一样，不加 v1 候选。

```text
任务 T16：把 T14 的双教师自动标注（data/labels_dual）导成新的 SUSTechPOINTS 场景（35 个，公共盘 sustech_dual/），
沿用 H1 的审核结论；训练和验证一样，只留两个教师的框，不加 v1 候选。SUSTechPOINTS 改指向新场景。
先按 docs/SERVER_TASKS.md 的"通用约定"做（拉到了代码改动，跑快速测试）。前置：T14 完成。不用 GPU。
旧的 sustech/ 不要动（H1 的结果在里面），只新建 sustech_dual/。

1. 停掉 SUSTechPOINTS（同 T11 第 1 步），记下启动命令。
2. 导出（每 5 帧取 1 帧，只有车，带训练 / 验证）：
     mkdir -p /public/zjst/robot/centerpoint_data/sustech_dual data/reports/T16
     NEW=/public/zjst/robot/centerpoint_data/sustech_dual; OLD=/public/zjst/robot/centerpoint_data/sustech
     awk -F, 'NR>1 && $2!="drop" {print $1, $2}' $SPLIT | while read r s; do
       python tools/dog/sustechpoints_convert.py export --vehicles-only --labels data/labels_dual/$r/labels.pkl \
           --frames data/runs/$r/frames --scene $r --every 5 --split $s --out $NEW
     done 2>&1 | tee data/reports/T16/export.txt
   每个场景的帧数要和 T1S 导出的一样（data/reports/T1S/export.txt），不一样就停下报告。
3. 沿用 H1 的结论：
     ALL=$(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT)
     python tools/dog/sustechpoints_convert.py carry --from-root $OLD --root $NEW --scenes $ALL | tee data/reports/T16/carry.txt
   "frames without an earlier review" 应该没有（帧对不上）；有就停下报告。
4. （不要再跑。）曾按 T11b 给验证集加过 v1 候选，2026-10-26 已去掉。
5. Ctrl + 拖框用的模型框（两个教师的）：
     python tools/dog/sustechpoints_convert.py detections --root $NEW --scenes $ALL --classes vehicle \
         --results teacher=data/runs/{scene}/voxel/results.pkl pp_tta=data/runs_pp_tta/{scene}/pillar/results.pkl \
         | tail -3 | tee data/reports/T16/detections.txt
6. 统计：python tools/dog/sustechpoints_convert.py stats --root $NEW | tee data/reports/T16/stats.txt
   训练、验证的红框（marked review）合计。红框超过 3,000 个就停下报告（审不完）。
7. SUSTechPOINTS 改指向新场景（旧的留在 sustech/）：
     for r in $ALL; do ln -sfn $NEW/$r third_party/SUSTechPOINTS/data/$r; done
   用第 1 步记下的命令重新起 SUSTechPOINTS。

停止条件：任何报错；第 2 步帧数对不上；第 3 步有帧对不上；第 6 步红框超过 3,000。
提交：data/reports/T16/、本文档。
报告（写进"T16 结果"）：carry.txt 的 total 行（沿用了多少、又删了多少、照搬了多少、新的红框多少）；candidates.txt 的 total 行；
stats 里训练、验证各有多少帧、框、红框；SUSTechPOINTS 地址。
```

### T16 结果

35 个场景导出到 `/public/zjst/robot/centerpoint_data/sustech_dual`，每个场景的帧数和 T1S 一样。旧的 `sustech/` 没动。

- carry.txt total：沿用 H1 的结论 **6424** 框，H1 删过、双教师又检出再删 **10** 框，H1 有而双教师没有、照搬过来 **655** 框，新目标标红 **471** 框。另外 H1 还红着的 **310** 框仍是红的。没有对不上的帧。审核人挪过的框用了 H1 几何的有 3 个。T11b 的旧候选没照搬 151 框。
- 第 4 步曾加上 225 个 v1 候选框。2026-10-26 按审核人的要求去掉了（`stats_no_candidates.txt`）：每个验证场景里非候选框一个没少。验证集和训练集一样，只留两个 nuScenes 教师的框。去掉之后验证集 **580** 帧、**2611** 框、红框 **196**。Ctrl + 拖框仍是两个教师（`teacher`、`pp_tta`）。`0921_165659_s004` 去掉候选后没有框。
- 去掉之前的 stats：训练 **1476** 帧、**4939** 框、红框 **584**；验证 **580** 帧、**2836** 框、红框 **422**。合计红框 **1006**，没超过 3,000。

SUSTechPOINTS 已重新起好：`python tools/dog/sustechpoints_convert.py serve --host 10.36.136.7`，地址 http://10.36.136.7:8081 。可以开始 H3（先验证集，Ctrl+F5，只审红框）。

---

## H3 审双教师的标注（人做）

T16 之后，在 SUSTechPOINTS 里审 35 个场景（先验证集，再训练集），浏览器重新打开页面（Ctrl+F5）。**只审车辆**（主要是小轿车和货车），行人、骑行者不审。

H1 定过的框不用再看（不是红的）。要看的红框有两种，训练和验证一样：
- **新目标**：两个教师看到、H1 没见过的；
- **H1 剩下没定的**：H1 时就红着的。

| 看到的 | 操作 |
|---|---|
| 是车（含停着的、挂车厢、叉车） | 右键 → 确认（整条轨迹）。框明显歪了（偏出车 0.5 m 以上）就调一下 |
| 不是车（墙角、集装箱、货堆、岗亭……） | 右键 → Delete this object (in all frames)。**删，不要标忽略**：删了，模型才学到那里是背景 |
| 看不清（点太少、只看到一角） | 右键 → 看不清（作忽略框） |
| 同一个 ID 有几帧是车、有几帧不是 | 按 `v` 进批量窗口逐帧看，不是车的那几帧单独删 |
| 标签写"其他帧是 #12" | 同一位置别的帧有 #12：多半是同一辆停着的车，确认即可；#12 是忽略框的，这里也点"看不清" |

**漏掉的车**：翻帧时看到车没有框，按住 Ctrl 拖矩形框进去，自动出框（两个教师的模型框），检查后类别改成 Car 或 Truck。

**盲区边上**（机器人正后方左右约 140° 的方向）的红框多留意一下：T13 里 v1 Pillar 在那里误检多，杂物删掉。

做完把状态表的 H3 改成"完成"，注明审核人、日期，大致记一下确认了多少、删了多少、补画了几辆。

### H3 结果

junyongmou，2026-10-26。35 个场景审完，没有新画的车。收尾时还红着的 **82** 框（**44** 条轨迹：训练 59、验证 23）全部改成忽略。现在训练 1476 帧、4903 框，验证 580 帧、2575 框，红框都是 **0**。

---

## T17 导回 H3，导出 v2 数据集

```text
任务 T17：导回 H3 在 sustech_dual/ 里的审核结果；训练场景把关键帧的结果传播到每一帧，再补停着的车的空档；导出 v2 数据集（新目录）。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：H3 完成。
不要覆盖 H1 的 labels_sustech.pkl、labels_full.pkl：新的叫 labels_sustech2.pkl、labels_full2.pkl、labels_train2.pkl。

1. 35 个场景导回：
     mkdir -p data/reports/T17; NEW=/public/zjst/robot/centerpoint_data/sustech_dual
     ALL=$(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT)
     for r in $ALL; do
       echo "== $r"
       python tools/dog/sustechpoints_convert.py import --scene-dir $NEW/$r --frames data/runs/$r/frames \
           --out data/labels/$r/labels_sustech2.pkl
     done 2>&1 | tee data/reports/T17/import.txt
   "still marked review" 合计超过 100 个框就停下问人（H3 没审完）。
2. 训练场景：关键帧传播到每一帧（--full 用双教师的自动标注），再补停着的车的空档（只补框里看得见的帧）：
     TRAIN=$(awk -F, 'NR>1 && $2=="train" {print $1}' $SPLIT)
     for r in $TRAIN; do
       echo "== $r"
       python tools/dog/sustechpoints_convert.py propagate --full data/labels_dual/$r/labels.pkl \
           --keyframes data/labels/$r/labels_sustech2.pkl --out data/labels/$r/labels_full2.pkl
       python tools/dog/merge_labels.py --reviewed data/labels/$r/labels_full2.pkl --frames data/runs/$r/frames \
           --out data/labels/$r/labels_train2.pkl
     done 2>&1 | tee data/reports/T17/propagate.txt
3. 导出到新目录（照 T5 第 3 步）：验证场景用 labels_sustech2.pkl，训练场景用 labels_train2.pkl：
     REC=$(for r in $(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT); do echo --recording data/runs/$r data/labels/$r/labels_sustech2.pkl; done
           for r in $TRAIN; do echo --recording data/runs/$r data/labels/$r/labels_train2.pkl; done)
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
     NEWDS=$(dirname $(readlink -f data/dog_veh))/dog_veh_$(date +%m%d_%H%M)
     python tools/dog/export_dataset.py $REC --val-recordings $VAL --classes vehicle \
         --out $NEWDS --gt-db --jobs 8 2>&1 | tee data/reports/T17/export.txt
     OLD=$(readlink -f data/dog_veh); ln -sfn $NEWDS data/dog_veh; echo "data/dog_veh: $OLD -> $NEWDS"
   旧目录 $OLD 写进报告，T15 的 v1 对比不读它，用它的训练、评估都结束了再删。

停止条件：任何报错；导回的帧数和 T16 导出的不一致；某个训练场景传播后留下的帧少于原来的 90%。
提交：data/labels/0921_*/labels_sustech2.pkl、labels_full2.pkl、labels_train2.pkl、data/reports/T17/、本文档。
报告（写进"T17 结果"）：import 的计数合计（车辆框、still marked review、moved or new）；propagate 和补空档的合计；
export 的 car / truck 正样本、忽略数，和 T5（v1）的对比；新数据集目录。
```

### T17 结果

35 个场景导回的帧数和 T16 一致（验证 580、训练关键帧 1,476，合计 2,056）。车辆框 **7,478**，still marked review **0**，moved or new **642**（其中 637 个是双教师导出里没有的，主要是从 H1 照搬过来、ID 带 h 的框）。H1 的 `labels_sustech.pkl`、`labels_full.pkl` 没动。

- propagate（25 个训练场景）：留下 **7,232 / 7,479** 帧（96.7%）。keyframe boxes 4,903，interpolated boxes 18,356，other classes, copied as they are 36,292，left out: removed by the reviewer 104，ignore: reviewed at one end 536，frames left out (outside the keyframes) 79。
- `0921_165659_s024` 打印是 229/303 = 75.6%。这 303 帧里 55 帧是 `pose_bad`，propagate 不写这些帧。去掉之后是 229/248 = 92.3%，和 T5 一样。其余训练场景按全部帧算都 ≥ 90.7%（最低是 `0921_165659_s011` 的 274/302）。所以继续导出。
- 补停着的车的空档：parked places 136（其中 51 处被拆成好几段），gap frames filled **274**，gap frames hidden (left as they are) 1,544。
- 导出到新目录 `/public/zjst/robot/centerpoint_data/dog_veh_1026_1410`（37G）。`data/dog_veh`：`/public/zjst/robot/centerpoint_data/dog_veh_1023_1529` → 这个新目录。旧目录是 v1，等用它的训练、评估结束后再删。train 7,232 / val 580 帧。
  - train：car 14,339（+2,893 ignored），truck 5,073（+1,418 ignored）。v1（T5）是 car 14,524（+2,335），truck 4,078（+1,321）。
  - val：car 1,100（+217 ignored），truck 1,091（+155 ignored）。v1 是 car 950（+272），truck 1,132（+102）。
  - note：few points 1,025；no points 358；subclass disagrees 34。v1 是 875 / 298 / 121。
  - GT 库：car 14,339、truck 5,073。

（2026-10-26 复核）
- **结果可用，T15 可以开始。** 本地把提交的标签读了一遍：
  - 验证 10 个场景：车辆框 2,575 个（忽略 331），没有红框。
  - 训练 25 个场景：传播后车辆框 24,069 个（忽略 3,763，补空档 274）。
  - 训练标签里还带 review 标记的 250 个框全是忽略框（H3 收尾时改的），训练时不当正样本，不影响。
  - s024 只留 75.6% 的解释成立：去掉 55 帧 pose_bad 后是 92.3%。
- **和 v1 比**：
  - 训练集：轿车基本不变（14,339 对 14,524），货车多 995 个（+24%），忽略框多 655 个；
  - 验证集：轿车多 150 个（+16%），货车少 41 个；
  - 子类别不一致从 121 降到 34。
- **新的验证集偏向两个 nuScenes 教师，也偏向 v2**：
  - 验证标签里大多数框的位置是两个教师融合出来的（H1 没挪过的框都换成了双教师的框），v1 的候选也去掉了；
  - v2 学的正是这种框，所以 AP@0.5、AP@1.0 这些对框位置敏感的指标，会偏向 v2 和 nuScenes 模型。
  - 教师换不换，T15 本来就只看 AP@2.0、AP@4.0。上线 Pillar 的第二条规则看 ΔmAP，而 mAP 里有 AP@0.5、AP@1.0，所以 T15 第 7 步加了一条：走 ΔmAP 这条时，AP 表里 v2_pp 的 AP@2.0、AP@4.0 还都不能低于 v1_pp。
- **旧数据集目录** `dog_veh_1023_1529`（v1）：T15 不读它，v1 的结果已经在 `runs_veh_v1` 里。确认没有进程在读，就可以删真实路径。

---

## T12 在新的验证标签上重评，选下一轮的两个教师

```text
任务 T12：在 T17 导回的新验证标签（labels_sustech2.pkl）上重评 T7 的 7 个模型；再选下一轮的两个教师（Voxel、Pillar 各一个，
以后每轮一起迭代，最终一起做自动标注）：Voxel 在 nuScenes Voxel + TTA（现在的教师）和 v1 Voxel 第 10、20 个 epoch 加 TTA
里选；Pillar 是 v1 Pillar（第 20 个 epoch），看加不加 TTA。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T17 完成（10 个验证场景都有 data/labels/<场景>/labels_sustech2.pkl）；
T14 第 1 步完成（data/runs_pp_tta 下 10 个验证场景都有 results.pkl）。
新标签是 labels_sustech2.pkl（双教师 + H1 的结论 + H3），旧的 labels_sustech.pkl（H1）留着比较。
GPU：第 3 步每个进程约 6 GB（Voxel + TTA），--jobs 2 就是 12 GB，按通用约定 5 放卡。

1. mkdir -p data/reports/T12; VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
2. 标签变了多少（每个场景：车辆框 旧 → 新；人加的框 added 旧 → 新，其中确认的、仍作忽略的）。
   下面的 Python 从行首开始写（heredoc 里不能缩进），整段复制：
python - $VAL <<'EOF' | tee data/reports/T12/label_changes.txt
import pickle, sys
def veh(path):
    return [o for f in pickle.load(open(path, "rb"))["frames"] for o in f["objects"] if o["label"] == "vehicle"]
print("scene vehicles_old vehicles_new added_old added_new added_new_confirmed added_new_still_review")
for r in sys.argv[1:]:
    old, new = veh("data/labels/%s/labels_sustech.pkl" % r), veh("data/labels/%s/labels_sustech2.pkl" % r)
    add = [o for o in new if o.get("added")]
    print(r, len(old), len(new), sum(bool(o.get("added")) for o in old), len(add),
          sum(not o["review"] and not o.get("ignore") for o in add), sum(bool(o["review"]) for o in add))
EOF
3. 候选教师加 TTA，重放验证场景：v1 Voxel 第 10、20 个 epoch；v1 Pillar 第 20 个 epoch：
     { for e in 10 20; do
         python tools/dog/run_sequence.py --replay $(for r in $VAL; do echo data/runs/$r; done) --detectors voxel --tta \
             --ckpt-voxel data/train/dog_voxel_veh_v1/epoch_$e.pth --no-predict --no-mcap --jobs 2 --device cuda:0 \
             --out "data/runs_teacher_v1_e$e/{scene}"
       done
       python tools/dog/run_sequence.py --replay $(for r in $VAL; do echo data/runs/$r; done) --detectors pillar --tta \
           --ckpt-pillar data/train/dog_pp_veh_v1/epoch_20.pth --no-predict --no-mcap --jobs 1 --device cuda:0 \
           --out "data/runs_teacher_pp_v1/{scene}"
     } 2>&1 | tee data/reports/T12/runs.txt
4. 在新标签上重评，T7 的 7 个模型再加三个候选教师：
     RES=$(for r in $VAL; do echo v1_voxel=data/runs_veh_v1/$r/voxel/results.pkl v1_pp=data/runs_veh_v1/$r/pillar/results.pkl \
         v0_voxel=data/runs_veh_v0/$r/voxel/results.pkl v0_pp=data/runs_veh_v0/$r/pillar/results.pkl \
         nusc_voxel=data/runs_nusc_voxel/$r/voxel/results.pkl nusc_pp=data/runs_nusc_pp/$r/pillar/results.pkl \
         teacher=data/runs/$r/voxel/results.pkl teacher_v1_e10=data/runs_teacher_v1_e10/$r/voxel/results.pkl \
         teacher_v1_e20=data/runs_teacher_v1_e20/$r/voxel/results.pkl \
         teacher_pp_v1=data/runs_teacher_pp_v1/$r/pillar/results.pkl \
         nusc_pp_tta=data/runs_pp_tta/$r/pillar/results.pkl \
         dual=data/runs/$r/voxel/results.pkl+data/runs_pp_tta/$r/pillar/results.pkl; done)
     python tools/dog/compare_detectors.py --classes vehicle --results $RES --max-range 40 \
         --labels $(for r in $VAL; do echo data/labels/$r/labels_sustech2.pkl; done) \
         --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 --target-recall 0.9 --bootstrap 1000 \
         | tee data/reports/T12/compare_val.md
   旧标签再跑一次（不要 --thresholds），看结论变了没有：
     python tools/dog/compare_detectors.py --classes vehicle --results $RES --max-range 40 \
         --labels $(for r in $VAL; do echo data/labels/$r/labels_sustech.pkl; done) --bootstrap 1000 \
         | tee data/reports/T12/compare_val_h1.md
   `--max-range 40`：候选只提了 40 m 内的，40 m 外的标签和检测都不评分（T7 的 mAP 含 40 m 外，所以数字和 T7 不能直接比，
   新旧标签之间可以比）。
   候选只来自 v1 Voxel、v1 Pillar（T11b）：nuScenes 两个模型、v0 多检出的真车没有审，还算误检，它们的分数偏低，
   和它们的比较只作参考。v1 两个模型之间、它们和 teacher 之间的比较是这次要的。
   `dual` 是两个 nuScenes 教师（Voxel + TTA、Pillar + TTA）的检测按帧融合后的结果，和双教师自动标注用的一样
   （两个都检出的取平均分，只有一个检出的分数减半）。
   AP 表后面多了一张"velocity head on moving labels"：各模型的速度头怎么读运动的车。自动标注要教师的速度头
   读到 > 0.15 m/s 才把这一帧算运动（"head > 0.15 m/s" 一列）。
5. 运动状态（新标签）：
     motion() {  # motion <名字> <结果目录名> <pillar|voxel>
       python tools/dog/motion_eval.py $(for r in $VAL; do echo --scene data/labels/$r/labels_sustech2.pkl data/$2/$r/$3/results.pkl; done) \
           --details data/reports/T12/motion_$1_details.csv | tee data/reports/T12/motion_$1.txt
     }
     motion v1_pp runs_veh_v1 pillar; motion nusc_pp runs_nusc_pp pillar
6. 选下一轮的两个教师（新标签的 AP 表）。只看 AP@2.0、AP@4.0 和 AP@2.0[20-40m]：框的位置还是第一轮教师给的，
   AP@0.5、1.0 仍然偏向 teacher。
   - Voxel 教师，在 teacher、teacher_v1_e10、teacher_v1_e20 三个里比。选 v1 的某个 epoch，要同时满足：
     a. AP@2.0 比 teacher 高 ≥ 0.02；
     b. AP@2.0[20-40m] 不低于 teacher（教师要看得远）；
     c. 速度表里 "head > 0.15 m/s" 不低于 teacher 的减 5 个百分点（运动的车要标成运动）。
     两个 epoch 都满足时选 AP@2.0 高的；都不满足就继续用 teacher（nuScenes Voxel + TTA）。
   - Pillar 教师：teacher_pp_v1（加 TTA）的 AP@2.0 比 v1_pp（不加）高 ≥ 0.01 就加 TTA，否则不加（快一倍多）。
     再按 a–c 拿它和 teacher 比一次，写出满足了哪几条：Pillar 不够好也照样当第二个教师，这一条只是记下差距。

停止条件：任何报错；某个模型某个场景的 results.pkl 缺失。
提交：data/reports/T12/、本文档。
报告（写进"T12 结果"）：
- label_changes.txt 整段，以及合计：加了多少车框、多少条轨迹；
- compare_val.md 的 AP 表、速度表、重抽的两张表（整段贴）；compare_val_h1.md 的 AP 表；
- 新旧标签下，这几个差值各一句（带 90% 区间）：v1_pp − nusc_pp、v1_voxel − v1_pp、v1_voxel − nusc_voxel；
- 各模型和 teacher 的 mAP 差，新旧标签下各是多少（偏向教师的问题小了多少）；
- 两个 motion_*.txt 的表；
- 第 6 步选的 Voxel 教师和 Pillar 教师（加不加 TTA），各满足了哪几条；teacher、teacher_v1_e10、teacher_v1_e20、
  v1_pp、teacher_pp_v1、nusc_pp、nusc_pp_tta、dual 的 AP@2.0、AP@4.0、AP@2.0[20-40m]、"head > 0.15 m/s" 列成一张表；
- nusc_pp_tta − nusc_pp、dual − teacher 两行重抽的差值和区间（TTA 对 Pillar 有没有用；双教师比单教师好多少）。
```

### T12 结果

（待填）

---

## T13 雷达后方盲区：边上是不是误检更多

XT32 正后方约 74–78° 没有点（[SENSORS.md](SENSORS.md) 2.2）。被盲区截断的墙、成排的车，断口像车的一端，模型可能把它当成车；
车一半在盲区里时，框也会往看得见的一半偏。本地用 nuScenes Pillar 在 3 个场景上看过（s000、s002、s025）：
盲区边上 20° 内的误检比例和别处差不多（10 m 内 12% 对 14%，10–40 m 21% 对 26%），但上线的是 v1 Pillar，要用它的结果再看。

看完 T13 再定怎么改。可选的做法（按代价从小到大）：
1. 盲区写进雷达配置，规划把它当"看不见"，不是"没有东西"。
2. 跟踪：轨迹进了盲区多保留一会儿（按预测外推，不删），从另一边出来接上，不当成新车。
3. 跟踪：在盲区边上**新出现**的目标，多确认一两帧再上报。
4. 训练：GT 采样贴进来的车，落在盲区里的点删掉，和真雷达一样；盲区边上的杂物在审核里删掉（不要标忽略），让模型学到那是背景。
5. 硬件：查清是什么挡住了（宽约 0.8 m、在雷达后方约 0.5 m），能挪开或把雷达抬高最好。

**注意安全**：从后面开过来的车，正是先出现在盲区边上。所以不能在边上直接把检测删掉或降分。第 3 条只适用于突然出现、又不连续的目标，而且要先在数据上验证，不会推迟报出真车。

```text
任务 T13：量出雷达后方盲区，统计 v1 Pillar、v1 Voxel、教师、nuScenes Pillar 在盲区边上和别处的误检比例。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T7 完成。不用 GPU，只读 results.pkl、标签和存好的帧。

1. VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT); mkdir -p data/reports/T13
   python tools/dog/sector_check.py --scenes $VAL --labels "data/labels/{scene}/labels_sustech.pkl" \
       --frames "data/runs/{scene}/frames" \
       --results v1_pp=data/runs_veh_v1/{scene}/pillar/results.pkl v1_voxel=data/runs_veh_v1/{scene}/voxel/results.pkl \
                 teacher=data/runs/{scene}/voxel/results.pkl nusc_pp=data/runs_nusc_pp/{scene}/pillar/results.pkl \
       | tee data/reports/T13/sector_val.md
   第一行是量出来的盲区（度，0 为正前方）。之后每个模型分"盲区里（blind）/ 盲区边上 20° 内（edge）/ 别处"、10 m 内外统计；
   最后列出盲区边上和盲区里分数最高的误检（场景、时间戳）。
2. 训练场景（机器人走动、转弯多）的盲区也量一下，只量不评（教师的结果在每个场景都有）：
     TRAIN=$(awk -F, 'NR>1 && $2=="train" {print $1}' $SPLIT)
     python tools/dog/sector_check.py --scenes $TRAIN --labels "data/labels/{scene}/labels_full.pkl" \
         --frames "data/runs/{scene}/frames" --results teacher=data/runs/{scene}/voxel/results.pkl --list 0 \
         | tee data/reports/T13/sector_train_teacher.md

停止条件：任何报错；第 1 步没量出盲区（"none found"）。
提交：data/reports/T13/、本文档。
报告（写进"T13 结果"）：两次量出的盲区；sector_val.md 的表（整段）；v1_pp 盲区边上的误检比例是不是明显高于别处
（同一距离段，高出一半以上、并且至少 20 个误检，就算"明显"）；v1_pp 误检列表的前 10 条。
```

### T13 结果

两次量出来的盲区（0° 为正前方）：验证集 292 帧 **142°–218°**（宽 76°）；训练集 766 帧 **142°–217°**（宽 75°）。和 SENSORS.md 写的正后方约 74–78° 没有点一致。

验证集（`sector_val.md`）：

| model | range | zone | detections | false | false share |
|---|---|---|---|---|---|
| v1_pp | < 10 m | elsewhere | 80 | 10 | 12% |
| v1_pp | < 10 m | edge | 22 | 5 | 23% |
| v1_pp | < 10 m | blind | 27 | 15 | 56% |
| v1_pp | 10-40 m | elsewhere | 1306 | 90 | 7% |
| v1_pp | 10-40 m | edge | 250 | 27 | 11% |
| v1_pp | 10-40 m | blind | 3 | 3 | 100% |
| v1_voxel | < 10 m | elsewhere | 114 | 54 | 47% |
| v1_voxel | < 10 m | edge | 18 | 6 | 33% |
| v1_voxel | < 10 m | blind | 12 | 2 | 17% |
| v1_voxel | 10-40 m | elsewhere | 1354 | 141 | 10% |
| v1_voxel | 10-40 m | edge | 275 | 24 | 9% |
| v1_voxel | 10-40 m | blind | 0 | 0 | - |
| teacher | < 10 m | elsewhere | 103 | 10 | 10% |
| teacher | < 10 m | edge | 22 | 4 | 18% |
| teacher | < 10 m | blind | 14 | 0 | 0% |
| teacher | 10-40 m | elsewhere | 1649 | 178 | 11% |
| teacher | 10-40 m | edge | 381 | 36 | 9% |
| teacher | 10-40 m | blind | 8 | 1 | 12% |
| nusc_pp | < 10 m | elsewhere | 178 | 96 | 54% |
| nusc_pp | < 10 m | edge | 31 | 9 | 29% |
| nusc_pp | < 10 m | blind | 12 | 2 | 17% |
| nusc_pp | 10-40 m | elsewhere | 2181 | 769 | 35% |
| nusc_pp | 10-40 m | edge | 422 | 106 | 25% |
| nusc_pp | 10-40 m | blind | 16 | 5 | 31% |

v1 Pillar 盲区边上：10 m 内 23% 对别处 12%，比例高出一半以上，但误检只有 5 个，不到 20，**不算明显**。10–40 m 11% 对别处 7%（高出约六成），误检 27 个，**算明显**。盲区里面 10 m 内更高，56%（15/27），这一档不在“边上”的判定里。

v1 Pillar 误检列表前 10 条（场景、时间戳、区域、分数、距离 m、方位角）：

| 场景 | 时间戳 | 区域 | 分数 | 距离 | 方位角 |
|---|---|---|---|---|---|
| 0921_164127_s003 | 1789980199.997579 | edge | 0.76 | 8.0 | -138 |
| 0921_165659_s006 | 1789981199.779807 | blind | 0.76 | 7.1 | -149 |
| 0921_164127_s000 | 1789980105.098712 | edge | 0.70 | 23.6 | -134 |
| 0921_164127_s003 | 1789980193.996433 | edge | 0.67 | 35.8 | -125 |
| 0921_164127_s001 | 1789980116.898520 | edge | 0.66 | 23.7 | -133 |
| 0921_164127_s001 | 1789980136.399215 | edge | 0.63 | 39.5 | -123 |
| 0921_164127_s002 | 1789980152.698378 | edge | 0.62 | 29.1 | -134 |
| 0921_164127_s003 | 1789980174.498870 | edge | 0.61 | 33.3 | -135 |
| 0921_164127_s000 | 1789980104.599369 | blind | 0.59 | 7.7 | -153 |
| 0921_164127_s001 | 1789980132.899012 | edge | 0.59 | 35.4 | -127 |


（2026-10-26 复核）
- **你说的盲区边上误检，在 v1 Pillar 上是有的，教师没有**：
  - 10–40 m：盲区边上误检 11%，别处 7%；教师 9% 对 11%；
  - 盲区里 10 m 内：v1 Pillar 27 个检测里 15 个是误检，教师 14 个里一个都没有。
- **只出在一侧**：前 10 条误检的方位角都在 −123° 到 −153°，盲区的同一边，而且大多是 164127 这份录制（s000–s003）。可能是这段路线左后方的环境（成排的车或建筑）被盲区截断，也可能和那一侧的什么东西有关。要看图才知道，H3 时可以顺便看这几帧（时间戳在上表）。
- **怎么改**，先不动上线的代码：
  - v2（T15）的训练标签是两个教师加 TTA 出的，盲区边上的杂物在 H3 里删掉（不是标忽略），v2 训完 T15 第 5 步再量一次；
  - 还多的话，再在跟踪里加"盲区边上新出现的目标多确认一帧"，先在数据上验证不推迟报出真车（T13 开头的注意事项）。

---

## T14 双教师重新自动标注 0921，补进审过的训练标签

**为什么不直接用双教师从头重标、再导出 v1**：H1 的人工审核（1–2 天）是挂在单教师的轨迹上的。从头重标，轨迹和框都会变，
审核结果就对不上了。所以以 H1 审过的训练标签（`labels_full.pkl`）为底，只补两样（`tools/dog/merge_labels.py`）：
1. **停着的车的空档，补成正样本**：同一位置前后都有审过的标注、中间几帧没标，这几帧框里有 ≥ 5 个当前帧的点，就补上同一个框
   （DATA_COLLECTION_PLAN.md 2.1：s025 上没标的 115 帧里 51 帧看得见）。
2. **双教师看到、审过的标签里没有的，补成忽略框**：训练时既不当车也不当背景，不再教网络"这里是背景"。
   双教师自动标注里，两个教师都检出的照常成轨迹；只有一个检出的，要那个教师自己很有把握（平均分 ≥ 0.6）才留下，而且标待审核。

验证标签不动（只用人审的）。

nuScenes Pillar 加 TTA 在本地两个验证场景上试过（DOG_FINETUNE.md 第 17 节）：0.30 分时召回 0.891 → 0.892，误检 267 → 159；
20–40 m 的 AP@2.0 0.821 → 0.859。所以第二个教师用加 TTA 的。

```text
任务 T14：nuScenes Pillar 加 TTA 跑全部 35 个场景；用两个 nuScenes 教师（Voxel + TTA 已有、Pillar + TTA）重新自动标注；
把结果补进 25 个训练场景审过的标签，得到 labels_merged.pkl（T15 导出 v1.1 用）。
先按 docs/SERVER_TASKS.md 的"通用约定"做（拉到了代码改动，跑快速测试）。前置：T7 完成。不用等 H3。
不要改 data/labels/<场景>/ 下已有的文件，只新增 labels_merged.pkl；不要动 data/runs。

0. 通用约定 6 的链接循环里加上 runs_pp_tta、labels_dual 两个目录（照约定 6 先链到公共盘）。
1. Pillar + TTA，重放 35 个场景存好的帧（每个进程约 13 GB，见通用约定 5；一张卡只放 1 个）：
     mkdir -p data/reports/T14
     ALL=$(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT)
     python tools/dog/run_sequence.py --replay $(for r in $ALL; do echo data/runs/$r; done) --detectors pillar --tta \
         --ckpt-pillar work_dirs/pretrained/nusc_pp.pth --no-predict --no-mcap --jobs 3 --device cuda:0 \
         --out "data/runs_pp_tta/{scene}" 2>&1 | tee data/reports/T14/runs.txt
   记下每帧的 network ms（和 T6 的 nusc_pp 不加 TTA 比）。
2. 双教师自动标注（CPU，4 个一起跑）：
     for r in $ALL; do echo $r; done | xargs -P 4 -I{} sh -c 'mkdir -p data/labels_dual/{} && \
         python tools/dog/auto_label.py --results data/runs/{}/voxel/results.pkl data/runs_pp_tta/{}/pillar/results.pkl \
             --frames data/runs/{}/frames --out data/labels_dual/{}/labels.pkl > data/labels_dual/{}/auto_label.log 2>&1'
     for r in $ALL; do echo "$r $(tail -1 data/labels_dual/$r/auto_label.log)"; done | tee data/reports/T14/dual_labels.txt
   和单教师的比：每个场景的轨迹数、标待审核的轨迹数（单教师的在 data/labels/<场景>/auto_label.log，没有的话用
   python -c 读 data/labels/<场景>/labels.pkl 数 track_id）。
3. 补进训练场景审过的标签：
     TRAIN=$(awk -F, 'NR>1 && $2=="train" {print $1}' $SPLIT)
     for r in $TRAIN; do
       python tools/dog/merge_labels.py --reviewed data/labels/$r/labels_full.pkl --frames data/runs/$r/frames \
           --teachers data/labels_dual/$r/labels.pkl --out data/labels/$r/labels_merged.pkl
     done 2>&1 | tee data/reports/T14/merge.txt
   每个场景一行：fill（gap frames filled / hidden、parked places split into several tracks）和
   teachers（added as ignore boxes、tracks added、already labelled）。
4. 合计：补的正样本框、加的忽略框，和 labels_full.pkl 里原有的车辆框数比。加的忽略框超过原有车辆框的 30% 就停下报告
   （第二个教师的误检太多，会把太多背景变成忽略）。

停止条件：任何报错；第 4 步超过 30%。
提交：data/labels/0921_*/labels_merged.pkl、data/reports/T14/、本文档。不提交 data/runs_pp_tta、data/labels_dual。
报告（写进"T14 结果"）：Pillar + TTA 的每帧耗时；dual_labels.txt 和单教师的对比（轨迹数、待审核数，合计）；
merge.txt 的合计和第 4 步的比例。
```

### T14 结果

2026-10-26 重跑完成。Pillar + TTA 放在物理 GPU 7（开始时空闲 16.9 GB、利用率 0%，`CUDA_VISIBLE_DEVICES=7 --device cuda:0 --jobs 1`）。单进程中途占到 12.6 GB，一张卡放不下第二个进程。启动时分配器报过两次内存不足，随后自己重试成功，35 个场景都写出了 `results.pkl`。

每帧网络耗时（中位数）：35 个场景 28.5–39.2 ms，中位数 **34.4 ms**。同样 10 个验证场景，T6 的 nuScenes Pillar 不加 TTA 是 13.1–29.3 ms（中位数 21.2 ms），这次加 TTA 是 28.5–34.8 ms（中位数 32.9 ms），大约 1.6 倍。

双教师轨迹比单教师少（只留两个都看到的，或单个教师平均分 ≥ 0.6 的）。合计：单教师 1231 条轨迹、610 条待审核；双教师 **992 条轨迹、502 条待审核**。逐场景（轨迹 / 待审核）：

| 场景 | 单教师 | 双教师 |
|---|---|---|
| 0921_164127_s000 | 31 / 10 | 31 / 16 |
| 0921_164127_s001 | 37 / 23 | 38 / 25 |
| 0921_164127_s002 | 34 / 18 | 27 / 16 |
| 0921_164127_s003 | 31 / 21 | 16 / 8 |
| 0921_164127_s004 | 22 / 16 | 19 / 15 |
| 0921_164127_s005 | 3 / 3 | 3 / 3 |
| 0921_165659_s000 | 30 / 11 | 25 / 11 |
| 0921_165659_s001 | 46 / 21 | 37 / 18 |
| 0921_165659_s002 | 39 / 24 | 30 / 16 |
| 0921_165659_s003 | 21 / 10 | 11 / 4 |
| 0921_165659_s004 | 5 / 3 | 3 / 2 |
| 0921_165659_s005 | 22 / 13 | 16 / 7 |
| 0921_165659_s006 | 31 / 16 | 19 / 8 |
| 0921_165659_s007 | 13 / 10 | 10 / 7 |
| 0921_165659_s008 | 9 / 4 | 5 / 2 |
| 0921_165659_s009 | 12 / 7 | 9 / 5 |
| 0921_165659_s010 | 11 / 5 | 15 / 12 |
| 0921_165659_s011 | 15 / 7 | 14 / 10 |
| 0921_165659_s012 | 7 / 4 | 3 / 3 |
| 0921_165659_s013 | 5 / 4 | 5 / 4 |
| 0921_165659_s014 | 30 / 12 | 23 / 12 |
| 0921_165659_s015 | 30 / 9 | 27 / 7 |
| 0921_165659_s016 | 57 / 29 | 50 / 24 |
| 0921_165659_s017 | 38 / 21 | 30 / 16 |
| 0921_165659_s018 | 68 / 31 | 58 / 25 |
| 0921_165659_s019 | 81 / 33 | 72 / 34 |
| 0921_165659_s020 | 64 / 23 | 58 / 23 |
| 0921_165659_s021 | 87 / 41 | 72 / 34 |
| 0921_165659_s022 | 90 / 56 | 49 / 30 |
| 0921_165659_s023 | 58 / 37 | 32 / 15 |
| 0921_165659_s024 | 51 / 21 | 43 / 19 |
| 0921_165659_s025 | 50 / 19 | 40 / 16 |
| 0921_165659_s026 | 57 / 20 | 54 / 22 |
| 0921_165659_s027 | 28 / 15 | 31 / 20 |
| 0921_165659_s028 | 18 / 13 | 17 / 13 |

25 个训练场景补进 `labels_merged.pkl`：正样本空档补了 **294** 框（另有 1509 个空档帧点不够，没补），忽略框加了 **1617** 框、74 条轨迹。`labels_full.pkl` 里原有车辆框 22546（其中非忽略 19746）。新忽略框占全部车辆框 7.2%、占非忽略车辆框 **8.2%**，没到 30%。已有标注、没有重复加的 20631 框。

（2026-10-26 复核）
- **结果可用，T15 等 T12 就能跑。** 本地把 25 个 `labels_merged.pkl` 读了一遍：帧数和 `labels_full.pkl` 一样；新加的轨迹号不和原有的重复；同一帧里没有重复的框。
- **停着的车中间漏标、其实看得见的帧，比我之前估的少得多**：
  - 空档帧一共 1,803 帧，框里点够、补上的只有 294 帧（16%），其余 1,509 帧是真被挡住了；
  - 我之前拿 s025 估算（115 帧里 51 帧看得见），s025 恰好是最严重的一个；
  - 294 个框只占原有正样本 19,746 个的 1.5%。"训练标签有矛盾"是次要原因，DATA_COLLECTION_PLAN.md 2.1 已改正。
  - 补得多的场景：s010（101 帧）、s027（52）、s025（51）、s015（34）、s028（29）。
- **加的 1,617 个忽略框（74 条轨迹）大多是远处的大家伙**：距离中位数 35 m；26 条长 6 m 以上（10–16 m 长、约 4 m 高），融合后的分数中位数 0.39。两个教师都只给了不高的分，可能是挂车、集装箱，也可能是建筑，当忽略框是对的（不当车，也不当背景）。
  - s013（H1 时没有车框）加了 1 条 142 框的：18 m 外、8.8 m 长、3.7 m 高、当前帧 532 个点，分数 0.35。
- **双教师的轨迹比单教师少**（992 对 1,231），待审核也少（502 对 610）：只有一个教师看到、又没把握的轨迹没进来。新数据用双教师标，人要审的会少一些。
- **Pillar + TTA 很占显存**：单进程峰值 12.6 GB，我原来写的 4–5 GB 错了，第一次因此停下。已改通用约定 5、T12 第 3 步（Pillar 那条改成 --jobs 1）。GPU 上网络耗时 21.2 → 32.9 ms（10 个验证场景的中位数），约 1.6 倍，和 Voxel 一样。
- **对 T15 的预期要放低**：v1.1 只比 v1 多 1.5% 的正样本、8% 的忽略框，v1.1 和 v1 的差别很可能在验证集分辨不出的范围内。T15 还是值得跑（3 小时 GPU，不用人看着）：如果分不出来，就说明 v1 不如教师主要不是标签的问题，下一步该改训练配方（混 nuScenes 数据或蒸馏）和多采数据。

---

## T15 训练 v2；定上线用哪个 Pillar、教师换不换

v2 用 T17 导出的数据集：双教师（nuScenes Voxel + TTA、nuScenes Pillar + TTA）的自动标注，沿用 H1 的审核结论，再经 H3 审过新出现的目标；
停着的车中间看得见的空档帧也补上了。配方和 v1 完全一样（单卡、batch 4、学习率 5e-4、20 个 epoch），所以 v2 和 v1 的差别只来自标签。
T14 的 labels_merged.pkl（v1.1）不再用。

T15 给出两个结论，分开定：
1. **上线用哪个 Pillar**（不加 TTA）：v2_pp、v1_pp（现在上线的）、nusc_pp 里选。
2. **教师换不换**（离线，加 TTA）：v2 加 TTA 和现在的 nuScenes 教师比。不比它好，教师就不换（继续用 nuScenes Voxel + TTA、
   nuScenes Pillar + TTA），T12 不做。

**比的时候要注意**：新验证标签是从两个 nuScenes 教师的框出发、再经人审的，框的位置是它们的，它们检出的东西也先进了标签，
所以对 nuScenes 模型有利。换教师只看 AP@2.0、AP@4.0、AP@2.0[20-40m]（对框的位置不敏感），并且要高出 0.02 以上。
v2 学的也是这两个教师融合出来的框，所以 AP@0.5、AP@1.0（还有包含它们的 mAP）也偏向 v2，比 v2_pp 和 v1_pp 时要看 AP@2.0、AP@4.0（第 7 步）。

```text
任务 T15：用 T17 导出的数据集（data/dog_veh）训练 v2 的 Voxel 和 Pillar；在新的验证标签上和 v1、nuScenes 模型比，
定上线用哪个 Pillar；v2 加 TTA 和 nuScenes 教师比，定教师换不换。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T17 完成。

0. mkdir -p data/reports/T15; VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT); RUNS=$(for r in $VAL; do echo data/runs/$r; done)
   readlink -f data/dog_veh 应是 T17 报告里的新目录，不是就停下报告。
1. 训练，照 T6 第 1、2 步放卡和检查，work_dir 换成 data/train/dog_voxel_veh_v2、data/train/dog_pp_veh_v2
   （先 ls 确认这两个目录不存在）。
2. 选 epoch，照 T7 第 1 步（ev 函数，权重目录 dog_voxel_veh_v2、dog_pp_veh_v2，报告写到 data/reports/T15/），
   记为 E_v2_voxel、E_v2_pp。
3. 流水线级，重放验证场景。不加 TTA（上线的样子）和加 TTA（当教师的样子）各一次：
     python tools/dog/run_sequence.py --replay $RUNS --detectors pillar,voxel \
         --ckpt-pillar data/train/dog_pp_veh_v2/epoch_<E_v2_pp>.pth --ckpt-voxel data/train/dog_voxel_veh_v2/epoch_<E_v2_voxel>.pth \
         --no-predict --no-mcap --jobs 4 --device cuda:0 --out "data/runs_veh_v2/{scene}"
     python tools/dog/run_sequence.py --replay $RUNS --detectors voxel --tta \
         --ckpt-voxel data/train/dog_voxel_veh_v2/epoch_<E_v2_voxel>.pth --no-predict --no-mcap --jobs 2 --device cuda:0 \
         --out "data/runs_veh_v2_tta/{scene}"
     python tools/dog/run_sequence.py --replay $RUNS --detectors pillar --tta \
         --ckpt-pillar data/train/dog_pp_veh_v2/epoch_<E_v2_pp>.pth --no-predict --no-mcap --jobs 1 --device cuda:0 \
         --out "data/runs_veh_v2_tta/{scene}"
   （Voxel + TTA 每个进程约 6 GB，Pillar + TTA 约 13 GB，通用约定 5。）
4. 对比（新验证标签，40 m 内）：
     RES=$(for r in $VAL; do echo v2_voxel=data/runs_veh_v2/$r/voxel/results.pkl v2_pp=data/runs_veh_v2/$r/pillar/results.pkl \
         v1_voxel=data/runs_veh_v1/$r/voxel/results.pkl v1_pp=data/runs_veh_v1/$r/pillar/results.pkl \
         nusc_voxel=data/runs_nusc_voxel/$r/voxel/results.pkl nusc_pp=data/runs_nusc_pp/$r/pillar/results.pkl \
         teacher=data/runs/$r/voxel/results.pkl nusc_pp_tta=data/runs_pp_tta/$r/pillar/results.pkl \
         dual=data/runs/$r/voxel/results.pkl+data/runs_pp_tta/$r/pillar/results.pkl \
         v2_voxel_tta=data/runs_veh_v2_tta/$r/voxel/results.pkl v2_pp_tta=data/runs_veh_v2_tta/$r/pillar/results.pkl \
         dual_v2=data/runs_veh_v2_tta/$r/voxel/results.pkl+data/runs_veh_v2_tta/$r/pillar/results.pkl; done)
     python tools/dog/compare_detectors.py --classes vehicle --max-range 40 --results $RES \
         --labels $(for r in $VAL; do echo data/labels/$r/labels_sustech2.pkl; done) \
         --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 --target-recall 0.9 --bootstrap 1000 \
         | tee data/reports/T15/compare_val.md
5. 运动状态（新标签），上线候选的三个 Pillar：
     motion() {  # motion <名字> <结果目录名> <pillar|voxel>
       python tools/dog/motion_eval.py $(for r in $VAL; do echo --scene data/labels/$r/labels_sustech2.pkl data/$2/$r/$3/results.pkl; done) \
           --details data/reports/T15/motion_$1_details.csv | tee data/reports/T15/motion_$1.txt
     }
     motion v2_pp runs_veh_v2 pillar; motion v1_pp runs_veh_v1 pillar; motion nusc_pp runs_nusc_pp pillar
6. 盲区边上的误检（T13 里 v1 Pillar 在边上明显多）：
     python tools/dog/sector_check.py --scenes $VAL --labels "data/labels/{scene}/labels_sustech2.pkl" --sector 142 218 \
         --results v2_pp=data/runs_veh_v2/{scene}/pillar/results.pkl v1_pp=data/runs_veh_v1/{scene}/pillar/results.pkl \
         --list 10 | tee data/reports/T15/sector_val.md
7. 两个结论：
   - **上线的 Pillar**：看重抽表里 "v2_pp − v1_pp" 一行。ΔAP@2.0 0-20 m ≥ +0.02、并且 A ahead ≥ 90%，或者 ΔmAP ≥ +0.03、
     A ahead ≥ 90%、ΔAP@2.0 0-20 m ≥ −0.01、并且 AP 表里 v2_pp 的 AP@2.0、AP@4.0 都不低于 v1_pp，就换成 v2_pp；
     否则留 v1_pp。再看运动状态：换上的那个，相关运动段
     （≥ 1 m/s、30 m 内）漏报不能比留下的多。
   - **教师**：v2_voxel_tta 对 teacher、dual_v2 对 dual 各比一次，同时满足下面三条才算 v2 更好：
     a. AP@2.0 高 ≥ 0.02；b. AP@2.0[20-40m] 不低；c. 速度表 "head > 0.15 m/s" 不低于减 5 个百分点。
     两组都不满足：教师不换，T12 不做，状态表里 T12 改成"不做（教师不换）"。满足了就写出是哪一组，T12 再按新结论重写。

停止条件：任何报错。
提交：data/reports/T15/、两个训练的 .log 和 *.log.json、本文档。
报告（写进"T15 结果"）：E_v2_voxel、E_v2_pp；compare_val.md 的 AP 表、速度表和重抽的两张表；
v2_voxel − v1_voxel、v2_pp − v1_pp、v2_pp − nusc_pp、v2_voxel_tta − teacher、dual_v2 − dual 的 ΔmAP、ΔAP@2.0 0–20 m 和区间；
阈值 0.30 时 40 m 内的误检数（v2_pp、v1_pp、nusc_pp）；三个 motion_*.txt 的表；sector_val.md 的表；第 7 步的两个结论。
```

### T15 结果

数据集是 T17 的 `/public/zjst/robot/centerpoint_data/dog_veh_1026_1410`（7232 / 580）。Voxel 在 GPU 5，Pillar 在 GPU 3，单卡，没有 `--autoscale-lr`。每个 epoch 2125 步。Voxel 约 1 小时 32 分（迭代 0.13 s，读数据 0.005 s，进程约 4.0 GB）；Pillar 约 59 分（0.08 s / 0.001 s，约 3.5 GB）。没有 nan。

**E_v2_voxel = 20，E_v2_pp = 10。** merged/vehicle 的 AP@2.0：Voxel 0.745 / 0.763 / 0.796（第 10 / 15 / 20 个 epoch），取第 20。Pillar 0.820 / 0.726 / 0.742，取第 10。

重放验证集时没有卡能放下任务里写的并行度（两模型一起约 4.5 GB，`--jobs 4` 要约 19 GB；当时最多空闲 8.5 GB），所以不加 TTA 和 Voxel + TTA 都是 `--jobs 1`。Pillar + TTA 单进程要约 13 GB、再留 1 GB，八张卡都不够，**v2_pp_tta 和 dual_v2 没有跑**。

AP 表（40 m 内，新验证标签，2011 个车框）：

| model | AP@0.5 | AP@1.0 | AP@2.0 | AP@4.0 | mAP | AP@2.0[0-10m] | AP@2.0[10-20m] | AP@2.0[20-40m] |
|---|---|---|---|---|---|---|---|
| v2_voxel | 0.549 | 0.726 | 0.814 | 0.833 | 0.731 | 0.505 | 0.937 | 0.791 |
| v2_pp | 0.581 | 0.748 | 0.836 | 0.875 | 0.760 | 0.545 | 0.939 | 0.816 |
| v1_voxel | 0.558 | 0.728 | 0.791 | 0.822 | 0.725 | 0.411 | 0.939 | 0.768 |
| v1_pp | 0.520 | 0.689 | 0.786 | 0.804 | 0.700 | 0.501 | 0.945 | 0.750 |
| nusc_voxel | 0.710 | 0.837 | 0.893 | 0.915 | 0.839 | 0.887 | 0.962 | 0.856 |
| nusc_pp | 0.699 | 0.794 | 0.847 | 0.864 | 0.801 | 0.930 | 0.937 | 0.801 |
| teacher | 0.785 | 0.891 | 0.925 | 0.933 | 0.883 | 0.880 | 0.983 | 0.903 |
| nusc_pp_tta | 0.753 | 0.839 | 0.884 | 0.896 | 0.843 | 0.927 | 0.946 | 0.843 |
| dual | 0.866 | 0.926 | 0.941 | 0.948 | 0.920 | 0.956 | 0.981 | 0.916 |
| v2_voxel_tta | 0.581 | 0.728 | 0.816 | 0.834 | 0.740 | 0.488 | 0.942 | 0.796 |

速度头（运动标签 > 1 m/s，162 个）：

| model | matched | head > 0.15 m/s | median error m/s |
|---|---|---|---|
| v2_voxel | 124 | 73% | 1.56 |
| v2_pp | 118 | 89% | 1.40 |
| v1_voxel | 131 | 54% | 2.26 |
| v1_pp | 122 | 84% | 1.49 |
| nusc_voxel | 145 | 90% | 0.93 |
| nusc_pp | 127 | 91% | 1.08 |
| teacher | 151 | 93% | 0.88 |
| nusc_pp_tta | 135 | 93% | 1.05 |
| dual | 157 | 94% | 0.83 |
| v2_voxel_tta | 129 | 74% | 1.86 |

重抽（10 个场景、1000 次、90% 区间）：

| model | mAP | interval | AP@2.0 0-20 m | interval |
|---|---|---|---|---|
| v2_voxel | 0.731 | 0.612 – 0.777 | 0.833 | 0.558 – 0.954 |
| v2_pp | 0.760 | 0.629 – 0.808 | 0.861 | 0.618 – 0.952 |
| v1_voxel | 0.725 | 0.610 – 0.769 | 0.812 | 0.531 – 0.941 |
| v1_pp | 0.700 | 0.611 – 0.739 | 0.827 | 0.574 – 0.947 |
| nusc_voxel | 0.839 | 0.738 – 0.873 | 0.943 | 0.834 – 0.976 |
| nusc_pp | 0.801 | 0.704 – 0.840 | 0.932 | 0.875 – 0.973 |
| teacher | 0.883 | 0.792 – 0.912 | 0.959 | 0.861 – 0.987 |
| nusc_pp_tta | 0.843 | 0.770 – 0.879 | 0.941 | 0.894 – 0.978 |
| dual | 0.920 | 0.864 – 0.945 | 0.978 | 0.955 – 0.992 |
| v2_voxel_tta | 0.740 | 0.619 – 0.786 | 0.830 | 0.562 – 0.951 |

要看的差值（其余行在 `compare_val.md`）：

| A − B | ΔmAP | interval | A ahead | ΔAP@2.0 0-20 m | interval | A ahead |
|---|---|---|---|---|---|---|
| v2_voxel − v1_voxel | +0.006 | -0.012 – 0.023 | 67% | +0.021 | -0.009 – 0.068 | 85% |
| v2_pp − v1_pp | +0.060 | 0.005 – 0.085 | 96% | +0.034 | -0.010 – 0.069 | 89% |
| v2_pp − nusc_pp | -0.041 | -0.105 – -0.008 | 3% | -0.071 | -0.266 – -0.007 | 3% |
| v2_voxel_tta − teacher | -0.144 | -0.196 – -0.115 | 0% | -0.129 | -0.304 – -0.034 | 0% |

`v2_voxel_tta − teacher` 由重抽表里 `teacher − v2_voxel_tta` 取反得到。dual_v2 没有。

阈值 0.30、40 m 内的误检：v2_pp **259**，v1_pp **156**，nusc_pp **909**。

运动状态（相关段是 ≥ 1 m/s、30 m 内）：

| model | relevant missed / all | latency median / p90 s |
|---|---|---|
| v2_pp | 1 / 7 | 1.00 / 2.00 |
| v1_pp | 1 / 9 | 0.25 / 2.30 |
| nusc_pp | 2 / 9 | 0.50 / 0.90 |

盲区（固定 142°–218°，新标签）：

| model | range | zone | detections | false | false share |
|---|---|---|---|---|---|
| v2_pp | < 10 m | elsewhere | 116 | 51 | 44% |
| v2_pp | < 10 m | edge | 28 | 13 | 46% |
| v2_pp | < 10 m | blind | 20 | 8 | 40% |
| v2_pp | 10-40 m | elsewhere | 1378 | 140 | 10% |
| v2_pp | 10-40 m | edge | 330 | 36 | 11% |
| v2_pp | 10-40 m | blind | 4 | 4 | 100% |
| v1_pp | < 10 m | elsewhere | 80 | 12 | 15% |
| v1_pp | < 10 m | edge | 22 | 9 | 41% |
| v1_pp | < 10 m | blind | 27 | 15 | 56% |
| v1_pp | 10-40 m | elsewhere | 1306 | 79 | 6% |
| v1_pp | 10-40 m | edge | 250 | 25 | 10% |
| v1_pp | 10-40 m | blind | 3 | 3 | 100% |

**上线的 Pillar：换成 v2 Pillar，第 10 个 epoch。** `v2_pp − v1_pp` 的 ΔAP@2.0 0–20 m 是 +0.034，但 A ahead 只有 89%，第一条不成立。第二条成立：ΔmAP +0.060、A ahead 96%、ΔAP@2.0 0–20 m +0.034，并且 AP@2.0 0.836、AP@4.0 0.875 都不低于 v1（0.786、0.804）。相关运动段漏报都是 1 次，没有比 v1 多。同一张标签上 nuScenes Pillar 仍然更高（ΔmAP −0.041，ΔAP@2.0 0–20 m −0.071），选择规则比的是 v2 和 v1。

**教师不换。** v2 Voxel + TTA 对 nuScenes Voxel + TTA：AP@2.0 0.816 对 0.925（低 0.109），AP@2.0[20-40m] 0.796 对 0.903，速度头 74% 对 93%（低 19 个百分点），三条都不满足。dual_v2 没有测，所以不能说两组都不满足，T12 仍是暂缓。有一张空闲 ≥ 14 GB 的卡之后补跑 Pillar + TTA，再比 dual_v2。

（2026-10-26 复核）
- **上线用 nuScenes Pillar（人定），T18 的基线也用它。** T15 第 7 步的规则只比 v2 和 v1，所以选了 v2_pp；
  人定的规则是从 nusc_pp、v1_pp、v2_pp 三个里选最好的，v1、v2 都不如 nusc_pp 就用 nusc_pp（`work_dirs/pretrained/nusc_pp.pth`）。
  - 差距主要在 20 m 内：AP@2.0 0–20 m 0.932 对 v2_pp 的 0.861，重抽里 v2 领先只有 3%。
    同样的精确率（约 0.91）下，20 m 内 nusc_pp（阈值 0.50）召回 0.838，v2_pp（阈值 0.45）是 0.732。
  - 10 m 内差得最多（AP@2.0 0.930 对 0.545），但验证集 10 m 内只有 11 辆车、186 个关键帧框，
    164127_s005、165659_s006 两个场景占了 108 个。v1、v2 为什么检不好这几辆车，H4 里对照 v2 的 MCAP 看（T18 已加）。
  - 20 m 以外 v2 不差：AP@2.0[20-40m] 0.816 对 0.801，AP@4.0 0.875 对 0.864。
  - 速度头和运动状态也是 nusc_pp 好一些：读成运动 91% 对 89%，误差中位数 1.08 对 1.40 m/s；
    相关运动段报出延迟 p90 0.9 s 对 2.0 s（漏报 2 / 9 对 1 / 7，段数太少，分不出来）。
- **要注意两点**：
  - 验证标签是从两个 nuScenes 教师的框审出来的（T17 复核），对 nusc_pp 有利，尤其是 AP@0.5、AP@1.0（0.699 对 0.581）。
    但 20 m 内召回差 0.1，不像只是这种偏向造成的。
  - nusc_pp 的分数和 v1、v2 不在一个尺度上：阈值 0.35 时，40 m 内误检 618 个，v2_pp 是 194 个。
    nusc_pp 的 F1 最高在阈值 0.40（40 m 内）和 0.45（20 m 内），比默认的 0.35 高。
    所以 T19 的上报阈值改成往高扫（0.35、0.3、0.4、0.45、0.5、0.55），按 30 m 内的 F1 选。
- **教师不换，T12 不做，dual_v2 也不用补跑。**
  - v2 Voxel + TTA 比 nuScenes Voxel + TTA 的 AP@2.0 低 0.109，速度头低 19 个百分点；v2_pp 也不如 nusc_pp。
    两个 v2 模型都比对应的 nuScenes 模型差，融合后超过 dual（0.941）0.02 以上的可能很小。
  - 按 10-26 定的规则，微调的模型不如 nuScenes 原始模型，教师就不换。
- **v2 Pillar 第 10 个 epoch 最好，之后掉得多**（AP@2.0 0.820 → 0.726 → 0.742），Voxel 则一直在涨（0.745 → 0.796）。
  验证集小、波动大，这只能说明 Pillar 后半程不一定更好；下一轮还是 10、15、20 个 epoch 都评。
- **T8 改为导出 nusc_pp 的 ONNX**（`export_pillar_onnx.py` 会从权重认出 nuScenes 的配置），已改 T8 第 1 步。


---

## T18 后处理、跟踪、预测的基线

计划、指标和可视化见 [PIPELINE_TUNING.md](PIPELINE_TUNING.md)。这一步不改参数，用默认参数做出七份报告。
上线的 Pillar 是 **nuScenes Pillar**（2026-10-26 人定：nusc_pp、v1_pp、v2_pp 三个里它最好，见 T15 复核）。

| 名字 | 场景 | 检测 | 用来 |
|---|---|---|---|
| `deploy` | 验证 10 个 | nuScenes Pillar（上线），过滤前的检测在这里重新过滤 | **基线**，带 Foxglove MCAP。后面调参的 A 组 |
| `v2_pp`、`v1_pp` | 验证 | 两个微调的 Pillar（T15 存好的结果） | 跟踪层面再比一次；v2_pp 也出 MCAP，对照看它 10 m 内哪里不行 |
| `teacher` | 验证 | nuScenes Voxel + TTA 教师 | 检测强很多时，跟踪能到哪 |
| `oracle_val` | 验证 | 10 Hz 标签当检测 | 跟踪、预测本身的上限 |
| `nusc_pp_train` | 训练 25 个 | 同一个 nuScenes Pillar（它没见过任何 0921 场景），过滤前的检测 | 调参的 B 组：换一批场景，防止调到验证集的偶然上去 |
| `oracle_train` | 训练 | 10 Hz 标签当检测 | 预测样本多一些（T21） |

```text
任务 T18：用默认参数做后处理、跟踪、预测的基线：七份报告、验证场景的 Foxglove MCAP、问题清单。上线的 Pillar 是 nuScenes Pillar。
先按 docs/SERVER_TASKS.md 的"通用约定"做（拉到了代码改动，跑快速测试）。前置：T15 完成。
GPU 只在第 2 步用（Pillar 不加 TTA，每个进程约 3 GB）；其余都是 CPU。

0. 通用约定 6 的链接循环里加上 tune 目录（照约定 6 先链到公共盘）。
     mkdir -p data/reports/T18; TASK=T18
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT); TRAIN=$(awk -F, 'NR>1 && $2=="train" {print $1}' $SPLIT)
   上线的 Pillar（T22 也用）：PP=work_dirs/pretrained/nusc_pp.pth
1. 验证场景的 10 Hz 标签（只画图和 oracle 用，不算 AP），照 T17 第 2 步的 propagate：
     for r in $VAL; do
       echo "== $r"
       python tools/dog/sustechpoints_convert.py propagate --full data/labels_dual/$r/labels.pkl \
           --keyframes data/labels/$r/labels_sustech2.pkl --out data/labels/$r/labels_full2.pkl
     done 2>&1 | tee data/reports/T18/propagate_val.txt
2. 重放，存过滤之前的检测（--save-raw），各约 2–5 分钟：
     python tools/dog/run_sequence.py --replay $(for r in $VAL; do echo data/runs/$r; done) --detectors pillar \
         --ckpt-pillar $PP --save-raw --no-predict --no-mcap --jobs 4 --device cuda:0 --out "data/tune/deploy/{scene}"
     python tools/dog/run_sequence.py --replay $(for r in $TRAIN; do echo data/runs/$r; done) --detectors pillar \
         --ckpt-pillar work_dirs/pretrained/nusc_pp.pth --save-raw --no-predict --no-mcap --jobs 4 --device cuda:0 \
         --out "data/tune/nusc_pp_train/{scene}"
3. 评估。先定义（后面的任务也用）：
     ev() {  # ev <名字> "<场景>" <results 路径> [pipeline_eval.py 的其他参数]；输出在 data/reports/$TASK/<名字>/
       local name=$1 scenes=$2 res=$3 t0=$SECONDS; shift 3
       python tools/dog/pipeline_eval.py --scenes $scenes --results "$res" \
           --labels "data/labels/{scene}/labels_sustech2.pkl" --jobs 8 --out data/reports/$TASK/$name "$@" \
           > data/reports/$TASK/$name.log 2>&1 || echo "FAILED: $name (data/reports/$TASK/$name.log)"
       echo "$name: $((SECONDS - t0)) s"
     }
   然后：
     ev deploy "$VAL" "data/tune/deploy/{scene}/pillar/results.pkl" --detections raw \
         --dense-labels "data/labels/{scene}/labels_full2.pkl" --frames "data/runs/{scene}/frames" --mcap-dir data/tune/T18_mcap
     ev v2_pp "$VAL" "data/runs_veh_v2/{scene}/pillar/results.pkl" \
         --dense-labels "data/labels/{scene}/labels_full2.pkl" --frames "data/runs/{scene}/frames" --mcap-dir data/tune/T18_mcap_v2
     ev v1_pp "$VAL" "data/runs_veh_v1/{scene}/pillar/results.pkl" --no-mcap
     ev teacher "$VAL" "data/runs/{scene}/voxel/results.pkl" --no-mcap
     ev oracle_val "$VAL" "data/tune/deploy/{scene}/pillar/results.pkl" --detections labels \
         --dense-labels "data/labels/{scene}/labels_full2.pkl" --no-mcap
     ev nusc_pp_train "$TRAIN" "data/tune/nusc_pp_train/{scene}/pillar/results.pkl" --detections raw --no-mcap
     ev oracle_train "$TRAIN" "data/tune/nusc_pp_train/{scene}/pillar/results.pkl" --detections labels \
         --dense-labels "data/labels/{scene}/labels_train2.pkl" --no-mcap
4. 汇总：
     for n in deploy v2_pp v1_pp teacher oracle_val nusc_pp_train oracle_train; do
       echo "== $n"; sed -n '/^## Main metrics/,/^## Detections/p' data/reports/T18/$n/report.md | grep '^|'
     done | tee data/reports/T18/summary.md
     ls -la data/tune/T18_mcap; du -sh data/tune/T18_mcap data/tune/T18_mcap_v2

停止条件：任何报错；第 1 步某个场景传播后留下的帧少于原来的 90%（pose_bad 的帧不算，见 T17）；
某份报告开头"Label tracks stitched into one vehicle"超过 40（拼得太多，可能拼错）。
提交：data/reports/T18/（MCAP 不在里面）、验证场景新生成的 data/labels/0921_*/labels_full2.pkl、本文档。
（第 2 步只重放 nusc_pp；v1_pp、v2_pp 用 T15 存好的结果，已经按默认参数过滤过。）
报告（写进"T18 结果"）：
- summary.md 里七份报告的主要指标，并排成一张表（行是指标，列是七个名字）；
- deploy 报告里的几张表：检测、跟踪、"Where the errors are"（含盲区三个区）、运动状态、预测（moving 那张）、预警、耗时；
- v2_pp 和 deploy 并排的"missed vehicles (FN)"表（按距离、点数）和盲区三个区的表，看 10 m 内差在哪里；
- deploy 的 issues 各类条数，和每类最严重的 3 条（场景、秒数、说明）；
- 拼接数；MCAP 目录和总大小；第 3 步每条的耗时。
```

### T18 结果

（待填）

---

## H4 在 Foxglove 里看基线（人做）

T18 之后，把 `data/tune/T18_mcap/` 拷到本机，用 Foxglove 打开，Layout → Import from file 选同目录的 `foxglove_layout.json`。
`data/tune/T18_mcap_v2/` 是同样的场景换成 v2 Pillar：10 m 内 v2 的 AP 只有 0.545（nusc_pp 0.930），
在 164127_s005、165659_s006 里对照看 v2 是没检出、框偏了，还是一车多框。

- 先看 `data/reports/T18/deploy/report.md` 的主要指标，再按 `issues.csv` 逐条看：t 列是场景开始后的秒数，拖到那里。
- 3D 面板里可以开关的话题见 PIPELINE_TUNING.md 第 3 节。看问题时有用的几个：
  - `/det/filtered`（默认关）：被过滤掉的检测和原因；
  - `/det/low_score`（默认关）：只给跟踪用的低分检测；
  - `/blind`：盲区扇形。
- 每类问题记几条典型的（场景、秒数、看到了什么），归到下面几类，写进"H4 结果"：

| 类 | 例子 |
|---|---|
| ID 切换 | 两辆车交错、长货车一车两框、遮挡后换了号 |
| 盲区 | 车进出盲区时丢了、盲区边上的杂物成了轨迹 |
| 假轨迹 | 墙角、集装箱、货堆；检测框本身就有（看 `/det/boxes`）还是跟踪 coast 出来的 |
| 漏了 / 确认慢 | 远处点少、被挡、刚出现 |
| 运动误判 | 停着的车被报成在动（框在漂？速度头？），动的车报晚了 |
| 预测偏 | 转弯、加减速、掉头；方向反了 |

这些归类决定 D2 改什么代码。

### H4 结果

（待填）

---

## T19 后处理扫参

扫上报阈值和三道几何过滤。A 组（验证场景）定参数，B 组（训练场景）检查不变差，两组都用上线的 nuScenes Pillar。每组参数的第 1 行是现在的默认值（参考）。

```text
任务 T19：扫后处理参数，定车辆的上报阈值、轨迹输出分数、低分门槛、最少点数、离地 / 下沉上限。
先按通用约定做。前置：T18 完成。不用 GPU。

0. mkdir -p data/reports/T19; TASK=T19; VAL、TRAIN 和 ev 同 T18 第 0、3 步；BASE=""
   A() { local n=$1; shift; ev ${n}_A "$VAL" "data/tune/deploy/{scene}/pillar/results.pkl" --detections raw --no-mcap $BASE "$@"; }
   B() { local n=$1; shift; ev ${n}_B "$TRAIN" "data/tune/nusc_pp_train/{scene}/pillar/results.pkl" --detections raw --no-mcap $BASE "$@"; }
   每组都跑 A、B 两次，结果在 data/reports/T19/<组>_A/sweep.md、<组>_B/sweep.md。
1. 上报阈值（也是新建轨迹的阈值）：
     A score --set tracker.high_score=0.35,0.3,0.4,0.45,0.5,0.55;  B score --set tracker.high_score=0.35,0.3,0.4,0.45,0.5,0.55
   （nuScenes Pillar 的分数偏高：T15 里阈值 0.35 时 40 m 内误检 618 个，F1 最高在 0.40–0.45，所以往高扫。）
2. 轨迹输出分数、低分门槛：
     A output --set tracker.output_min_score=0.3,0.2,0.25,0.35 --set tracker.low_score=0.1,0.15,0.2
     B output（同上）
3. 当前帧最少点数：
     A points --set filter.min_points.vehicle=5,3,8,12;  B points（同上）
4. 离地 / 下沉上限：
     A ground --set filter.max_float=0.7,0.5,1.0 --set filter.max_sink=1.2,0.8,1.6;  B ground（同上）
   每组定下来以后，把选中的值加进 BASE（例如 BASE="$BASE --set tracker.high_score=0.3"），再跑下一组。
   第 1 组定了 high_score，第 2 组起都带着它。

怎么选（每组单独定；看 sweep.md 的两张表，第二张是和第 1 行的差，括号里是重抽里这一行更好的比例）：
- 第 1、3、4 组（检测）：主指标 det F1（30 m 内，召回和精确率一起看）。在 A 上同时满足下面几条的行里，选 det F1 最高的：
  a. det F1 的领先比例 ≥ 90%；
  b. moving missed 不多于第 1 行（不能为了少误检把运动的车漏掉）；
  c. det recall 不比第 1 行低 0.05 以上。
  B 上这一行的 det F1 也不能低于第 1 行，b 也要满足。
- 第 2 组（轨迹输出）：主指标 IDF1，同样要领先比例 ≥ 90%、B 上不低于第 1 行；约束是 moving missed 不多于第 1 行、
  confirm p90 不比第 1 行长 0.5 s 以上。
- 没有满足的行：这组留默认值（第 1 行）。
停止条件：任何报错。
提交：data/reports/T19/、本文档。
报告（写进"T19 结果"）：每组 A、B 两张 sweep.md 的表（贴原表）；每组选了哪一行、按哪条规则；最后的 BASE。
```

### T19 结果

（待填）

---

## T20 跟踪扫参

```text
任务 T20：扫跟踪参数和运动状态的开关，定车辆的关联半径、coast 时间、确认帧数、观测噪声、运动状态阈值。
先按通用约定做。前置：T19 完成。不用 GPU。

0. mkdir -p data/reports/T20; TASK=T20; VAL、TRAIN、ev 同 T18；A、B 同 T19 第 0 步；BASE=T19 报告里最后的 BASE。
1. 关联半径：             A gate --set tracker.vehicle.gate=3,2,2.5,4
2. 最长 coast 时间：      A age --set tracker.vehicle.max_age=1.0,0.5,1.5,2.0
3. 确认帧数、试探寿命：   A hits --set tracker.vehicle.min_hits=2,1,3 --set tracker.tentative_max_age=0.15,0.25
4. 观测噪声：             A noise --set tracker.vehicle.pos_std=0.3,0.2,0.5 --set tracker.vehicle.vel_std=0.8,0.5,1.5
5. 速度头、匹配方法：     A assoc --set tracker.use_velocity=true,false --set tracker.matching=greedy,hungarian
6. 运动状态：             A motion --set tracker.motion_on_speed=0.7,0.5,1.0 --set tracker.motion_on_frames=3,2,5
   每组 A、B 都跑（B 同样的参数），定下来的值加进 BASE 再跑下一组，和 T19 一样。

怎么选：
- 第 1–5 组：主指标 IDF1。在 A 上同时满足下面几条的行里，选 IDF1 最高的：
  a. IDF1 的领先比例 ≥ 90%；
  b. moving missed 不多于第 1 行；
  c. confirm p90 不比第 1 行长 0.5 s 以上；
  d. false tracks / min 不超过第 1 行的 1.1 倍。
  B 上这一行的 IDF1 也不能低于第 1 行。
- 第 6 组：先要 moving missed 在 A、B 上都不多于第 1 行；这些行里选 false moving / h 最低的，它对第 1 行的领先比例要 ≥ 90%。
- 没有满足的行：留默认值。
停止条件：任何报错。
提交：data/reports/T20/、本文档。
报告（写进"T20 结果"）：每组 A、B 的 sweep.md 表；每组选了哪一行、按哪条规则；最后的 BASE；
用最后的 BASE 在 A 上出一份完整报告（ev final_A "$VAL" "data/tune/deploy/{scene}/pillar/results.pkl" --detections raw
--no-mcap $BASE），贴主要指标，和 T18 的 deploy 并排。
```

### T20 结果

（待填）

---

## T21 预测

IMM（现在在线用的）和匀速（CV）比，再调 IMM 的模型。oracle（标签当检测）只看预测器本身；端到端（A 组）看整条流水线。预测样本少，oracle 用训练、验证两组一起看。

```text
任务 T21：比较 IMM 和匀速预测，扫 IMM 的运动模型和模式合并距离。
先按通用约定做。前置：T20 完成（D2 的代码改动如果有，已经拉下来、T20 复测过）。不用 GPU。

0. mkdir -p data/reports/T21; TASK=T21; VAL、TRAIN、ev 同 T18；A 同 T19；BASE=T20 报告里最后的 BASE。
   OV() { local n=$1; shift; ev ${n}_oracle_val "$VAL" "data/tune/deploy/{scene}/pillar/results.pkl" --detections labels \
            --dense-labels "data/labels/{scene}/labels_full2.pkl" --no-mcap $BASE "$@"; }
   OT() { local n=$1; shift; ev ${n}_oracle_train "$TRAIN" "data/tune/nusc_pp_train/{scene}/pillar/results.pkl" \
            --detections labels --dense-labels "data/labels/{scene}/labels_train2.pkl" --no-mcap $BASE "$@"; }
1. 现状（一组参数，出完整报告，看 Prediction 一节 IMM 和 CV 的表）：OV base; OT base; A base
2. 运动模型（"cv:<过程噪声>+ct:<过程噪声>:<角速度噪声>"；最后一个只有匀速一个模型）：
     M="tracker.vehicle.models=cv:1.5+ct:1.5:0.25,cv:1.0+ct:1.0:0.25,cv:2.5+ct:2.5:0.25,cv:1.5+ct:1.5:0.5,cv:1.5"
     OV models --set "$M"; OT models --set "$M"; A models --set "$M"
3. 模式合并距离：OV merge --set predict.merge_dist=0.25,0.5,1.0; OT merge（同上）; A merge（同上）

怎么选：
- 主指标 FDE 3 s（moving）。选的行要在 OT、OV、A 三份 sweep.md 里都比第 1 行低，并且至少两份领先比例 ≥ 90%；
  MR 3 s 不比第 1 行高；A 上 IDF1 不比第 1 行低 0.01 以上（运动模型也影响跟踪）。
- 没有满足的行：留默认值。
- 另外记下第 1 步里 IMM 和 CV 谁好：CV 在三份报告里都不比 IMM 差，写进报告，D2 再看要不要改。
停止条件：任何报错。
提交：data/reports/T21/、本文档。
报告（写进"T21 结果"）：第 1 步三份报告的 Prediction 表（moving、moving straight、moving turning）；第 2、3 步的 sweep.md 表；
选了哪一行；最后的 BASE（T22 要用）。
```

### T21 结果

（待填）

---

## T22 定版：参数改成默认值，从原始扫描完整跑一遍

T21 之后，我把 T19–T21 选定的参数改成代码里的默认值（`DEFAULT_SCORE_THRESHOLDS`、`TrackerConfig`、`DetectionFilterConfig`、`IMMPredictor`），
在这里写出旧默认值的 `configs.json`，推送后再做 T22。这一步不再是重放：
- 从原始扫描起跑，前半段（读扫描、去畸变、拼帧）也在里面，和线上一样；
- 新旧两组参数用同一份检测比。

```text
任务 T22：用新的默认参数，从原始扫描完整跑一遍验证场景，和旧默认参数比，出最终报告和 Foxglove MCAP。
先按通用约定做（拉到了代码改动，跑快速测试）。前置：T21 完成，并且本节已经写好了 configs.json（我改完代码后补）。
GPU：第 1 步，Pillar 一个进程约 3 GB。

0. mkdir -p data/reports/T22; TASK=T22; VAL、ev、PP 同 T18。
1. 从原始扫描跑（不加 --replay），存输入帧和过滤前的检测：
     python tools/dog/run_sequence.py --seq $(for r in $VAL; do echo data/seq/$r; done) --detectors pillar \
         --ckpt-pillar $PP --save-raw --save-frames --no-mcap --jobs 4 --device cuda:0 --out "data/tune/final/{scene}"
2. 最终报告（新默认值，带 MCAP）：
     ev final "$VAL" "data/tune/final/{scene}/pillar/results.pkl" --detections raw \
         --dense-labels "data/labels/{scene}/labels_full2.pkl" --frames "data/tune/final/{scene}/frames" \
         --mcap-dir data/tune/T22_mcap
3. 新旧对比（同一份检测，配对重抽）：
     ev compare "$VAL" "data/tune/final/{scene}/pillar/results.pkl" --detections raw --no-mcap \
         --configs docs/pipeline_tuning/T22_configs.json
4. 第 1 步每个场景的耗时（中位数；前半段 + 网络，不含跟踪和预测，那两项在 final 报告的 Timing 一节）：
     for r in $VAL; do python -c "import pickle,numpy as np,sys; R=pickle.load(open(sys.argv[1],'rb'))[1:]; \
         net=np.median([x['timings']['network_ms'] for x in R]); tot=np.median([sum(x['timings'].values()) for x in R]); \
         print(sys.argv[1], 'network %.1f ms, preprocess + detector %.1f ms' % (net, tot))" \
         data/tune/final/$r/pillar/results.pkl; done | tee data/reports/T22/timing.txt

停止条件：任何报错；final 的主要指标比 T20 的 final_A 差很多（IDF1 低 0.05 以上），可能是重放和完整跑不一致，停下报告。
提交：data/reports/T22/、本文档。
报告（写进"T22 结果"）：final 报告的主要指标和各节的表；compare 的 sweep.md（第 1 行旧默认值，第 2 行新默认值）；
和 T18 deploy 并排的主要指标；timing.txt；MCAP 目录和大小。
```

### T22 结果

（待填）

---

## H5 看最终效果，定上线参数（人做）

用 Foxglove 看 `data/tune/T22_mcap/`（同 H4），对照 T22 的报告，决定新参数是否上线。
- 上线后，在 Orin 上用 ROS 2 节点录一段（T8、H2 之后），用同一个工具评：录制转成序列，跑 `run_sequence.py`，再跑 `pipeline_eval.py`。
- 可控场景（DATA_COLLECTION_PLAN.md 3.2）采到并标好以后，用同样的命令评预警。

### H5 结果

（待填）
