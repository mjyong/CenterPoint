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
4. 做完以后：
   - 把报告写进本文档该任务的"结果"小节（替换"（待填）"）；
   - 把状态表里这一行改成"完成"（或"停在第 N 步"）；
   - 原始输出放在 `data/reports/<任务编号>/`；
   - 按"提交"一项提交，推送到 `claude/me/v01`。
   - 权重（`.pth`）、点云帧、`.pkl`、mcap 都不提交。
5. 选 GPU：`python tools/dog/batch_label.py --split $SPLIT --dry-run` 会打印每张卡的显存和占用。用空闲的卡，`CUDA_VISIBLE_DEVICES=<卡号>`。
6. 大的输出写到 `/public/zjst/robot`（仓库盘只剩约 27 GB，T0）（和 `data/seq`、`data/runs` 同一块盘）。空闲空间用 `df -h /public/zjst/robot` 看，至少 80 GB。不要用 `df -h .`：那是仓库所在盘，后面的导出、训练权重、推理结果都不写到那里。第一次往下面这些目录写之前，先链过去（已经是这个链接就不用再做）：
   ```bash
   for d in dog_veh train deploy runs_veh_v0 runs_nusc_pp runs_veh_v1 runs_veh_v1_view runs_r2 labels_r2 seq_lever; do
     mkdir -p /public/zjst/robot/centerpoint_data/$d
     ln -sfn /public/zjst/robot/centerpoint_data/$d data/$d
   done
   ```
7. 测试：只在 T0、或者 `git pull` 拉到了代码改动（`dog_perception/`、`tools/`、`det3d/`、`configs/`）时跑，而且只跑快速的：
   ```bash
   pip install pytest-xdist       # 第一次
   python -m pytest tests/dog -m "not slow" -n 8 -q --durations=10
   ```
   本地约 1 分钟。完整测试（多 11 个用样例 bag 和训练的慢测试）只在任务里写了时才跑：去掉 `-m "not slow"`。
   每个测试进程最多用 4 个 CPU 线程（`tests/dog/conftest.py`，`DOG_TEST_THREADS` 可改）。之前 T0 跑了 29 分钟，
   多半是每个进程都开了和核数一样多的线程，在共享服务器上互相抢。

## 状态

| 编号 | 任务 | 谁做 | 依赖 | 预计耗时 | 状态 |
|---|---|---|---|---|---|
| T0 | 环境和数据检查 | Agent | — | 10 分钟 | 完成 |
| T1 | 车辆审核材料 | Agent | T0 | 30 分钟 | 完成；标注规则变了，T1R 里重做 |
| T2 | 导出 v0 数据集（未审核的自动标注） | Agent | T0 | 30 分钟 | 完成；T1R 里重做 |
| T1R | 按新规则重做自动标注（遮挡点数、待审核、静止毛刺），重做 T1、T2，统计 | Agent | T0 | 40 分钟 | 完成 |
| T1S | 训练集也导出到 SUSTechPOINTS 逐框审核（和验证集分组，带分数 / 速度 / 运动，红框待审） | Agent | T1R | 20 分钟 | 完成 |
| T3 | 训练 v0：Voxel、Pillar 两张卡同时跑 | Agent | T1R | 1.5–2 小时（数据读取已加速） | 完成 |
| T3D | 每个模型 2 张卡训练 v0（数据并行），和单卡比 | Agent | T3 | 40 分钟 | 待做 |
| T4 | 评估 v0 | Agent | T3（有 T3D 的结果就一起评） | 1 小时 | 待做 |
| H1 | 人工审核车辆：35 个场景全在 SUSTechPOINTS 里逐框审（每 5 帧取 1 帧） | 人 | T1S | 1–2 天 | 完成（junyongmou，2026-10-23） |
| T5 | 应用审核结果，导出 v1 数据集 | Agent | H1 | 30 分钟 | 待做 |
| T6 | 训练 v1 | Agent | T5 | 2–6 小时 | 待做 |
| T7 | 评估 v1，选 epoch 和上报阈值 | Agent | T6 | 1 小时 | 待做 |
| T8 | 导出 Pillar v1 的 ONNX 并校验 | Agent | T7 | 10 分钟 | 待做 |
| T9 | （可选）用 v1 Voxel 当教师重新自动标注，准备第二轮 | Agent | T7 | 1 小时 | 待做 |
| T10 | （可选）比较 0921 的两个 INS 杠杆臂 | Agent | T0 | 1 小时 | 待做 |

- T1R 之后，T1S 和 T3–T4 可以同时进行；T1S 之后人开始审核（H1），v0 同时在训练、评估。
- 2026-10-22 标注规则改了三处（[DOG_FINETUNE.md](DOG_FINETUNE.md) 第 14 节）：遮挡和远处的框按点数处理；待审核改为"平均分 < 0.5 并且 ≥ 0.6 分的帧不到 5 帧"；短暂的假运动去掉。T1、T2 的材料和数据集都是旧规则做的，要在 T1R 里重做。
- v0 只用未审核的自动标注，主要作用是把 GPU 训练流程跑通、得到一个基线，**不上线**。要上线的是 v1。

## 背景：已经有的东西

| 内容 | 位置 | 来自 |
|---|---|---|
| 35 个场景（0921_164127_s000–s005、0921_165659_s000–s028），每个约 30 s | `data/seq/0921_*` | 5.1–5.2 |
| 激光里程计 + RTK 融合后的位姿（第六版，含倾斜） | 各序列的 `odom.npy` | 5.10 |
| 划分：训练 25 个、验证 10 个，不丢场景 | `data/survey/split/split.csv` | 5.10 |
| 教师推理（nuScenes Voxel + TTA）、模型输入帧 | `data/runs/<场景>/voxel/results.pkl`、`frames/` | 5.11 |
| 自动标注 | `data/labels/<场景>/labels.pkl`（已提交） | 5.11 |

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

0. 保留第一次（T1R 之前的标注）训练的权重，T4 拿来对比；再测一下数据读取：
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
- **结论要看评估**：T4 把双卡 v0 和单卡 v0 放在一起评。第 20 个 epoch 的车辆 AP 相差不到 0.01，T6 就用双卡；差得多，T6 仍用单卡（反正也只要 1 小时）。

代码（2026-10-23）：`tools/train.py` 现在认 torchrun 设的 `LOCAL_RANK`（以前两个进程都跑到 0 号卡）；分布式时不再调用没安装的 apex（每张卡的 BatchNorm 各算各的，和单卡一样；配置里写 `sync_bn = True` 才跨卡同步）；梯度只由 DistributedDataParallel 平均一次。本地用 CPU（gloo）跑通了双进程训练。

```text
任务 T3D：Voxel、Pillar 各用 2 张卡训练 v0（20 个 epoch），和 T3 的单卡 v0 比速度，T4 再比精度。
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

（待填）

---

## T4 评估 v0

**注意**：
- 验证集现在是没审核的自动标注，由教师自己生成，对教师有利。这一步只能说明训练没跑坏、和教师差多少；真正的比较在 T7。
- v0 的 truck head 只见过 13 条 truck 轨迹（T2：1,668 个框，另有 4,084 个忽略），truck AP 低是预期的，要等 H1 审完、v1 再看。

```text
任务 T4：评估 v0。模型级（验证集 infos_val）和流水线级（验证场景整段跑）都做，并扫描上报阈值。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T3 完成。

1. 模型级，第 10、15、20 个 epoch。另外把这两组的 epoch_20 也各评一次（work_dir 用 <权重目录>/eval_e20）：
   - data/train/dog_*_veh_v0_oldlabels：看标注规则改动的影响；
   - data/train/dog_*_veh_v0_2gpu（T3D 做了的话）：和单卡 v0 比，决定 T6 用不用双卡。
     mkdir -p data/reports/T4
     for e in 10 15 20; do for m in voxel pp; do
       python tools/dist_test.py configs/dog/dog_centerpoint_${m}_xt32_veh.py --work_dir data/train/dog_${m}_veh_v0/eval_e$e \
           --checkpoint data/train/dog_${m}_veh_v0/epoch_$e.pth 2>&1 | grep -E "Evaluation|time per frame" \
           > data/reports/T4/dist_test_${m}_e$e.txt
     done; done
2. 流水线级：v0（第 20 个 epoch，不加 TTA）和 nuScenes Pillar（现在线上能用的基线）在 10 个验证场景上各跑一遍：
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
     for r in $VAL; do
       python tools/dog/run_sequence.py --seq data/seq/$r --detectors pillar,voxel --no-mcap \
           --ckpt-pillar data/train/dog_pp_veh_v0/epoch_20.pth --ckpt-voxel data/train/dog_voxel_veh_v0/epoch_20.pth \
           --device cuda:0 --out data/runs_veh_v0/$r
       python tools/dog/run_sequence.py --seq data/seq/$r --detectors pillar --no-mcap \
           --ckpt-pillar work_dirs/pretrained/nusc_pp.pth --device cuda:0 --out data/runs_nusc_pp/$r
     done
   （CUDA_VISIBLE_DEVICES 设成一张空闲卡。）
3. 对比和阈值扫描（教师 = 5.11 的 nuScenes Voxel + TTA）：
     python tools/dog/compare_detectors.py --classes vehicle --labels $(for r in $VAL; do echo data/labels/$r/labels.pkl; done) \
         --results $(for r in $VAL; do echo teacher=data/runs/$r/voxel/results.pkl nusc_pp=data/runs_nusc_pp/$r/pillar/results.pkl \
                     v0_voxel=data/runs_veh_v0/$r/voxel/results.pkl v0_pp=data/runs_veh_v0/$r/pillar/results.pkl; done) \
         --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 | tee data/reports/T4/compare_val.md
4. 在线运动状态。工具在 T1R 之后多了"相关的运动段"（中位速度 ≥ 1 m/s、中位距离 ≤ 30 m）和明细 CSV（每个漏报段、每次误报）：
   - 教师检测，全部 35 个场景（T1R 第 5 步的训练、验证两张表合成一张，并带上新列）：
     ALL=$(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT)
     python tools/dog/motion_eval.py $(for r in $ALL; do echo --scene data/labels/$r/labels.pkl data/runs/$r/voxel/results.pkl; done) \
         --on-speed 0.5 0.7 1.0 --on-frames 2 3 5 --details data/reports/T4/motion_teacher_details.csv \
         | tee data/reports/T4/motion_teacher.txt
   - v0 Pillar 的检测（线上跑的就是 Pillar），验证集 10 个场景：
     python tools/dog/motion_eval.py $(for r in $VAL; do echo --scene data/labels/$r/labels.pkl data/runs_veh_v0/$r/pillar/results.pkl; done) \
         --on-speed 0.5 0.7 1.0 --on-frames 2 3 5 --details data/reports/T4/motion_v0_pp_details.csv \
         | tee data/reports/T4/motion_v0_pp.txt

停止条件：任何报错。
提交：data/reports/T4/、本文档。
报告（写进"T4 结果"）：
- 每个模型每个 epoch 的 merged/vehicle 一行（mAP、AP@0.5/1/2/4、分距离段 AP、ATE、ASE、AOE），以及 car、truck 各一行；
- compare_val.md 的两张表（AP 表、耗时表），以及每个模型的阈值表最后的推荐行；
- 单卡 v0、双卡 v0（T3D）、旧标注 v0 的 epoch_20 并排：merged/vehicle 的 mAP、AP@2.0；双卡和单卡差多少；
- motion_teacher.txt、motion_v0_pp.txt 的表；两个 details CSV 里 on_speed 0.70、on_frames 3 的行数（missed / false moving 各多少）；
- 一句话结论：v0 比 nuScenes Pillar 好还是差、差在哪个距离段。
```

### T4 结果

（待填）

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
   ignore: reviewed at one end、ignore: between keyframes only（没人看过的，作忽略框）。
3. 导出：验证场景用 labels_sustech.pkl（只有关键帧），训练场景用 labels_full.pkl（每一帧）：
     REC=$(for r in $(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT); do echo --recording data/runs/$r data/labels/$r/labels_sustech.pkl; done
           for r in $(awk -F, 'NR>1 && $2=="train" {print $1}' $SPLIT); do echo --recording data/runs/$r data/labels/$r/labels_full.pkl; done)
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
     rm -rf data/dog_veh/*      # 清空链接指向的公共盘目录里的 v0 数据集；不要写成 rm -rf data/dog_veh（只删掉链接，
                                # 新的数据集就会写到仓库盘上）
     python tools/dog/export_dataset.py $REC --val-recordings $VAL --classes vehicle \
         --out data/dog_veh --gt-db --jobs 8 2>&1 | tee data/reports/T5/export.txt

验证集只有逐框审核过的帧（每 5 帧取 1 帧，约 580 帧），训练集是每一帧（约 7,300 帧，略少于 T1R：第一个关键帧之前、
最后一个关键帧之后的几帧不要），这是预期的。
停止条件：某个场景导回的帧数和 T1S 导出的不一致；某个训练场景 propagate 后留下的帧少于原来的 90%；任何报错。
提交：data/labels/0921_*/labels_sustech.pkl、labels_full.pkl、data/reports/T5/、本文档。
报告（写进"T5 结果"）：import 的类别计数和 still marked review 合计；propagate 各计数的合计；
export 的各类正样本 / 忽略数，和 T1R 的导出（训练 car 13,346 / truck 3,652，忽略 3,663 / 2,003）对比。
```

### T5 结果

（待填）

---

## T6 训练 v1

```text
任务 T6：用 v1 数据集训练只检车辆的 Voxel 和 Pillar 模型 v1。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T5 完成。

和 T3 完全一样，只是名字换成 v1。T4 里双卡 v0 和单卡 v0 的车辆 AP@2.0 相差不到 0.01 时，改用 T3D 的双卡命令
（work_dir 换成 data/train/dog_{voxel,pp}_veh_v1，T7 里选 epoch 时照常）：
  work_dir data/train/dog_voxel_veh_v1、data/train/dog_pp_veh_v1，日志 data/train/dog_{voxel,pp}_veh_v1.log，
  报告目录 data/reports/T6/。停止条件、提交、报告同 T3（写进"T6 结果"）。
```

### T6 结果

（待填）

---

## T7 评估 v1，选 epoch 和上报阈值

现在验证集是人审过的，这一步的数字才算数。

```text
任务 T7：在审核过的验证集上评估 v1（以及 v0、教师、nuScenes Pillar），选 epoch，给出上报阈值的建议。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T6 完成。

1. 模型级：同 T4 第 1 步，模型换成 v1（epoch 10、15、20），另外把 v0 的 epoch_20 在新验证集上也评一次；
   输出到 data/reports/T7/。
2. 选 epoch：Voxel、Pillar 各选 merged/vehicle 的 AP@2.0 最高的那个 epoch（相差 < 0.005 时选更早的）。记为 E_voxel、E_pp。
3. 流水线级：同 T4 第 2 步，权重换成 v1 的 E_pp / E_voxel，输出到 data/runs_veh_v1/<场景>。
4. 对比和阈值扫描，标签换成审核过的验证标签：
     python tools/dog/compare_detectors.py --classes vehicle --labels $(for r in $VAL; do echo data/labels/$r/labels_sustech.pkl; done) \
         --results $(for r in $VAL; do echo teacher=data/runs/$r/voxel/results.pkl nusc_pp=data/runs_nusc_pp/$r/pillar/results.pkl \
                     v0_voxel=data/runs_veh_v0/$r/voxel/results.pkl v0_pp=data/runs_veh_v0/$r/pillar/results.pkl \
                     v1_voxel=data/runs_veh_v1/$r/voxel/results.pkl v1_pp=data/runs_veh_v1/$r/pillar/results.pkl; done) \
         --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 --target-recall 0.9 | tee data/reports/T7/compare_val.md
5. 给 v1_pp 的样例场景出一个 Foxglove 文件，人眼看一看：
     python tools/dog/run_sequence.py --seq data/seq/0921_165659_s004 --detectors pillar \
         --ckpt-pillar data/train/dog_pp_veh_v1/epoch_<E_pp>.pth --device cuda:0 --out data/runs_veh_v1_view/0921_165659_s004
   把生成的 mcap 复制到 data/examples/runs/0921_165659_s004_v1/ 提交（Git LFS）。

停止条件：任何报错。
提交：data/reports/T7/、上面的 mcap、本文档。
报告（写进"T7 结果"）：
- E_voxel、E_pp 及理由；
- compare_val.md 的 AP 表（所有模型），以及 v1_pp、v1_voxel 的阈值表和推荐行；
- 结论：v1 比教师、nuScenes Pillar 好多少，主要在哪个距离段；
- 上报阈值的建议值（现在代码里 vehicle 是 0.35），由人决定是否改 DEFAULT_SCORE_THRESHOLDS。
```

### T7 结果

（待填）

---

## T8 导出 Pillar v1 的 ONNX

```text
任务 T8：把 T7 选出的 Pillar v1 导出成两段 ONNX，并和 PyTorch 逐段对比。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T7 完成，已知 E_pp。

1. f=$(ls data/runs/0921_165659_s004/frames/*.npy | sed -n 150p)
   python tools/dog/export_pillar_onnx.py --checkpoint data/train/dog_pp_veh_v1/epoch_<E_pp>.pth \
       --out data/deploy/dog_pp_veh_v1 --check $f 2>&1 | tee data/reports/T8/export.txt
2. ls -la data/deploy/dog_pp_veh_v1

通过标准：export.txt 里 pfn / bev / heads 的相对误差都 < 1e-3，PyTorch 和 ONNX 的检测个数相同，
分数差 < 1e-3、中心差 < 1 cm（本地用迁移权重试过：分数差 2.7e-07、中心差 3e-06 m）。
停止条件：不满足通过标准；报错。
提交：data/reports/T8/、本文档。不提交 ONNX（在 data/deploy/ 下，被 .gitignore 忽略），报告路径。
报告（写进"T8 结果"）：export.txt 全文；ONNX 文件大小；它打印的 trtexec 命令（在 Orin 上建 FP16 引擎用）。
```

### T8 结果

（待填）

---

## T9（可选）用 v1 Voxel 当教师，准备第二轮

v1 Voxel 专门在这批数据上训过，当教师应该比 nuScenes 模型准，待审核的轨迹会少很多。第二轮的审核和训练，等这一步的结果出来再定。

```text
任务 T9：用 T7 选出的 v1 Voxel（加 TTA）重新跑全部 35 个场景的教师推理和自动标注，输出到新的目录，不覆盖第一轮。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T7 完成，已知 E_voxel。

1. python tools/dog/batch_label.py --split $SPLIT --ckpt-voxel data/train/dog_voxel_veh_v1/epoch_<E_voxel>.pth \
       --runs data/runs_r2 --labels data/labels_r2
2. 每个场景出一张车辆表，统计待定的轨迹数：
     for r in $(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT); do
       python tools/dog/review_labels.py table --classes vehicle --labels data/labels_r2/$r/labels.pkl \
           --out data/labels_r2/$r/labels_tracks_vehicle.csv
     done | tee data/reports/T9/tables.txt

停止条件：任何报错；磁盘不足（第 1 步会再存一份模型输入帧，约 36 GB；和 data/runs 里的完全一样，
做完可以删掉 data/runs_r2/*/frames）。
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
