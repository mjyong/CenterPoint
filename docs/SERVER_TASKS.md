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
3. 不要删除或覆盖 `data/seq`、`data/runs`、`data/labels` 里已有的东西，除非步骤里写了。
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
| T1R | 按新规则重做自动标注（遮挡点数、待审核、静止毛刺），重做 T1、T2，统计 | Agent | T0 | 40 分钟 | 待做 |
| T3 | 训练 v0：Voxel、Pillar 两张卡同时跑 | Agent | T1R | 1.5–2 小时（数据读取已加速） | 完成一版（未做 T1R，Voxel 5.4 小时）；T1R 导出后重跑 |
| T4 | 评估 v0 | Agent | T3 | 1 小时 | 待做 |
| H1 | 人工审核车辆：先抽查新规则，再按轨迹（全部场景）+ 逐框（10 个验证场景） | 人 | T1R | 几个小时 | 待做 |
| T5 | 应用审核结果，导出 v1 数据集 | Agent | H1 | 30 分钟 | 待做 |
| T6 | 训练 v1 | Agent | T5 | 2–6 小时 | 待做 |
| T7 | 评估 v1，选 epoch 和上报阈值 | Agent | T6 | 1 小时 | 待做 |
| T8 | 导出 Pillar v1 的 ONNX 并校验 | Agent | T7 | 10 分钟 | 待做 |
| T9 | （可选）用 v1 Voxel 当教师重新自动标注，准备第二轮 | Agent | T7 | 1 小时 | 待做 |
| T10 | （可选）比较 0921 的两个 INS 杠杆臂 | Agent | T0 | 1 小时 | 待做 |

- T1R 之后，H1 和 T3–T4 可以同时进行：人审核（H1），v0 同时在训练。
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

（待填）

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
- 日志在 `data/reports/T3/`：`dog_voxel_veh_v0.log`、`dog_pp_veh_v0.log`，以及两个 `*.log.json`（work_dir 里原名都是 `20261022_165525.log.json`，复制时改成了和模型一样的名字）。

---

## T4 评估 v0

**注意**：
- 验证集现在是没审核的自动标注，由教师自己生成，对教师有利。这一步只能说明训练没跑坏、和教师差多少；真正的比较在 T7。
- v0 的 truck head 只见过 13 条 truck 轨迹（T2：1,668 个框，另有 4,084 个忽略），truck AP 低是预期的，要等 H1 审完、v1 再看。

```text
任务 T4：评估 v0。模型级（验证集 infos_val）和流水线级（验证场景整段跑）都做，并扫描上报阈值。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T3 完成。

1. 模型级，第 10、15、20 个 epoch（另外把 data/train/dog_*_veh_v0_oldlabels 的 epoch_20 也评一次，
   work_dir 用 data/train/dog_${m}_veh_v0_oldlabels/eval_e20，看标注规则改动的影响）：
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
4. 在线运动状态，用 v0 Pillar 的检测（线上跑的就是 Pillar），和 T1R 里教师的基线比：
     python tools/dog/motion_eval.py $(for r in $VAL; do echo --scene data/labels/$r/labels.pkl data/runs_veh_v0/$r/pillar/results.pkl; done) \
         --on-speed 0.5 0.7 1.0 --on-frames 2 3 5 | tee data/reports/T4/motion_v0_pp.txt

停止条件：任何报错。
提交：data/reports/T4/、本文档。
报告（写进"T4 结果"）：
- 每个模型每个 epoch 的 merged/vehicle 一行（mAP、AP@0.5/1/2/4、分距离段 AP、ATE、ASE、AOE），以及 car、truck 各一行；
- compare_val.md 的两张表（AP 表、耗时表），以及每个模型的阈值表最后的推荐行；
- motion_v0_pp.txt 的表；
- 一句话结论：v0 比 nuScenes Pillar 好还是差、差在哪个距离段。
```

### T4 结果

（待填）

---

## H1 人工审核车辆（人做）

只审车：不用管行人、骑行者，也不用改 car / truck / bus。规则见 [DOG_FINETUNE.md](DOG_FINETUNE.md) 第 13 节"审核只看车"，审核工具的用法见第 ④ 节和第 10 节。

0. **先抽查新的待审核规则（约 15 分钟）**：
   - `data/reports/T1R/review_spot_check.csv` 列的是平均分 < 0.5、但有 ≥ 5 帧 ≥ 0.6 分，因此不再要审的车辆轨迹；
   - 随机抽 20 条，在对应场景的 review mcap 里看，`real_vehicle` 列填 `y` / `n`；
   - 全是 `y`：规则可用，接着做第 1 步；
   - 有 `n`：把 CSV 提交上来，先别往下审。我来提高分数或帧数，Agent 重跑 T1R 第 1、3、6、7 步，轨迹表会变。
1. **按轨迹（全部 35 个场景）**：
   - 在 Foxglove 里打开 `data/labels/<场景>/<场景>_review.mcap`，对照 `data/labels/<场景>/labels_tracks_vehicle.csv` 填 `action` 列；
   - 只需处理 `train_as = ignore` 的行（条数见 T1R 结果；旧规则下训练集 135 条、验证集 72 条）。
   - **按 `frames` 从多到少审，不到 20 帧的可以不审。** 旧规则下，不到 20 帧的占要审轨迹的 30%，框只占 4%；留空的仍是忽略框，不会被当成背景。
   - `few_points` 列：这条轨迹有多少帧被挡住或太远。这些帧会自动作忽略框或不要，不用因为这些帧填 `ignore`。
   - 先审训练集里**按尺寸算 truck**（`length_m` ≥ 5.5 或 `height_m` ≥ 2.5）的忽略行。
     - 旧规则下，训练集能当正样本的 truck 只有 13 条轨迹、1,668 个框，比验证集的 18 条、3,215 个框还少；忽略着的 truck 尺寸轨迹有 56 条、4,084 个框。
     - 其中 84% 在这 6 个场景（旧规则下的条数 / 框数）：0921_165659_s028（8 / 1,024）、s012（3 / 709）、s027（9 / 596）、s026（6 / 442）、s016（5 / 364）、s025（3 / 277）。
     - 审完这 6 个场景、再加上验证集的逐框审核，就能先做 T5、T6。
     - 其余场景 `action` 空着的行，导出时仍是忽略框；之后补审完，重跑 T5 就行。
   - 动作怎么填：

     | 情况 | 填 |
     |---|---|
     | 是车 | `ok` |
     | 不是车 | `drop` |
     | 看不清 | `ignore` |
     | 朝向反了 | 加 `+flip` |
     | 停着的车，表里 `moving_share` 却不是 0 | 加 `+static`（狗遇到运动的车才停，静止判断要准） |
     | 在动的车，表里 `moving_share` 是 0 | 写在 `note` 列（现在没有动作能把静止改成运动） |
     | 小三轮车、老年代步车（算骑行者） | `cyclist` |
     | 单独停着的挂车厢（算车辆） | `ok` |

   - 填好的 CSV 提交到仓库。
2. **逐框（10 个验证场景）**：
   - 在服务器上运行 `python tools/dog/sustechpoints_convert.py serve`，浏览器打开 8081 端口；
   - 补漏标的车、调框。
   - 标注结果存在服务器的 `third_party/SUSTechPOINTS/data/<场景>/`，T5 会导回。
3. 做完把状态表的 H1 改成"完成"，注明审核人和日期。

### H1 结果

（待填：审核人、日期、各场景改了多少条，有没有特殊情况）

---

## T5 应用审核结果，导出 v1 数据集

```text
任务 T5：应用 H1 的车辆审核结果，重新导出只有车辆的数据集（v1），覆盖 data/dog_veh。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：H1 完成（各场景的 labels_tracks_vehicle.csv 已提交 action 列；
验证场景已在 SUSTechPOINTS 里标完）。

1. 按轨迹的审核：
     for r in $(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT); do
       python tools/dog/review_labels.py apply --labels data/labels/$r/labels.pkl \
           --edits data/labels/$r/labels_tracks_vehicle.csv --out data/labels/$r/labels_reviewed.pkl
     done 2>&1 | tee data/reports/T5/apply.txt
2. 验证场景导回逐框审核的结果：
     for r in $(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT); do
       python tools/dog/sustechpoints_convert.py import --scene $r --frames data/runs/$r/frames \
           --out data/labels/$r/labels_sustech.pkl
     done 2>&1 | tee data/reports/T5/import.txt
3. 导出（有 labels_sustech.pkl 用它，否则用 labels_reviewed.pkl）：
     REC=$(for r in $(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT); do \
           f=data/labels/$r/labels_sustech.pkl; [ -f $f ] || f=data/labels/$r/labels_reviewed.pkl; \
           echo --recording data/runs/$r $f; done)
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
     rm -rf data/dog_veh/*      # 清空链接指向的公共盘目录里的 v0 数据集；不要写成 rm -rf data/dog_veh（只删掉链接，
                                # 新的数据集就会写到仓库盘上）
     python tools/dog/export_dataset.py $REC --val-recordings $VAL --classes vehicle \
         --out data/dog_veh --gt-db --jobs 8 2>&1 | tee data/reports/T5/export.txt

验证集只有逐框审核过的帧（SUSTechPOINTS 每 5 帧取 1 帧，T1 导出了 580 帧），这是预期的。
停止条件：apply 报未知动作；某个验证场景导回的帧数和 T1 导出的不一致；任何报错。
提交：data/labels/0921_*/labels_reviewed.pkl、labels_sustech.pkl、data/reports/T5/、本文档。
报告（写进"T5 结果"）：每个场景 apply 打印的那一行（汇总成表）；import 的类别计数；
export 的各类正样本 / 忽略数，和 T2 对比（正样本应明显变多，忽略应明显变少）。
```

### T5 结果

（待填）

---

## T6 训练 v1

```text
任务 T6：用 v1 数据集训练只检车辆的 Voxel 和 Pillar 模型 v1。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T5 完成。

和 T3 完全一样，只是名字换成 v1：
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
