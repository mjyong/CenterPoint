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
| T1 | 车辆审核材料 | Agent | T0 | 30 分钟 | 待做 |
| T2 | 导出 v0 数据集（未审核的自动标注） | Agent | T0 | 30 分钟 | 待做 |
| T3 | 训练 v0：Voxel、Pillar 两张卡同时跑 | Agent | T2 | 2–6 小时 | 待做 |
| T4 | 评估 v0 | Agent | T3 | 1 小时 | 待做 |
| H1 | 人工审核车辆：按轨迹（全部场景）+ 逐框（10 个验证场景） | 人 | T1 | 几个小时 | 待做 |
| T5 | 应用审核结果，导出 v1 数据集 | Agent | H1 | 30 分钟 | 待做 |
| T6 | 训练 v1 | Agent | T5 | 2–6 小时 | 待做 |
| T7 | 评估 v1，选 epoch 和上报阈值 | Agent | T6 | 1 小时 | 待做 |
| T8 | 导出 Pillar v1 的 ONNX 并校验 | Agent | T7 | 10 分钟 | 待做 |
| T9 | （可选）用 v1 Voxel 当教师重新自动标注，准备第二轮 | Agent | T7 | 1 小时 | 待做 |
| T10 | （可选）比较 0921 的两个 INS 杠杆臂 | Agent | T0 | 1 小时 | 待做 |

- T1 和 T2–T4 可以同时进行：T1 的材料给人审核（H1），v0 同时在训练。
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

按只检车辆的规则（[DOG_FINETUNE.md](DOG_FINETUNE.md) 第 13 节），未审核时的车辆样本：

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
2. 每个场景一个只画车的 Foxglove 文件（点云 + 标注框 + 教师原始框），8 个同时跑：
     awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT | xargs -P 8 -I{} python tools/dog/review_labels.py mcap \
         --vehicles-only --labels data/labels/{}/labels.pkl --frames data/runs/{}/frames \
         --results data/runs/{}/voxel/results.pkl --out data/labels/{}/{}_review.mcap
3. 验证集的 10 个场景导出到 SUSTechPOINTS（只有车，每 5 帧取 1 帧）：
     for r in $(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT); do
       python tools/dog/sustechpoints_convert.py export --vehicles-only --labels data/labels/$r/labels.pkl \
           --frames data/runs/$r/frames --scene $r --every 5
     done | tee data/reports/T1/sustech.txt

停止条件：任何命令报错。
提交：data/labels/0921_*/labels_tracks_vehicle.csv、data/reports/T1/、本文档。不提交 *_review.mcap 和 SUSTechPOINTS 数据。
报告（写进"T1 结果"）：训练、验证各有多少车辆轨迹、多少 train_as=ignore；review mcap 的总大小；
SUSTechPOINTS 每个验证场景导出的帧数；审核人员怎么打开（Foxglove 打开 <场景>_review.mcap；
SUSTechPOINTS 在服务器上 python tools/dog/sustechpoints_convert.py serve，端口 8081）。
```

### T1 结果

（待填）

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

（待填）

---

## T3 训练 v0

```text
任务 T3：在两张空闲 GPU 上同时训练只检车辆的 Voxel 和 Pillar 模型 v0（20 个 epoch）。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T2 完成（data/dog_veh/infos_train.pkl 存在）。

1. 选两张空闲的卡（batch_label.py --dry-run），各起一个训练：
     mkdir -p data/train data/reports/T3
     CUDA_VISIBLE_DEVICES=<卡1> nohup python tools/train.py configs/dog/dog_centerpoint_voxel_xt32_veh.py \
         --work_dir data/train/dog_voxel_veh_v0 > data/train/dog_voxel_veh_v0.log 2>&1 &
     CUDA_VISIBLE_DEVICES=<卡2> nohup python tools/train.py configs/dog/dog_centerpoint_pp_xt32_veh.py \
         --work_dir data/train/dog_pp_veh_v0 > data/train/dog_pp_veh_v0.log 2>&1 &
   只有一张空闲卡时，先跑 Voxel，跑完再跑 Pillar。
2. 开始 10 分钟后检查两个日志（tail -30）：
   - loss 是有限值，hm_loss 在下降；
   - 每次迭代耗时、每个 epoch 多少次迭代，按 20 个 epoch 估算总时长；
   - nvidia-smi 看显存和利用率。数据集在公共盘上，利用率长时间低于 50% 说明读数据跟不上，记下来报告（先不用改）。
3. 等训练结束（每个 epoch 存一个 data/train/<名字>/epoch_N.pth），期间每小时看一次日志。

停止条件（停下并报告日志最后 50 行，不要改配置重跑）：
  报错（OOM、spconv、数据加载）；loss 变成 nan / inf；估算总时长超过 10 小时；
  训练进程意外退出（这时可以用 --resume_from data/train/<名字>/latest.pth 续一次，再退出就停下）。
提交：把两个 .log 和 work_dir 里的 *.log.json 复制到 data/reports/T3/，提交它们和本文档。
报告（写进"T3 结果"）：卡号和型号；每次迭代耗时；每个 epoch 迭代数；总耗时；显存；
每 5 个 epoch 最后一次的 loss（总的、hm_loss、loc_loss，两个 head 分开写）。
```

### T3 结果

（待填）

---

## T4 评估 v0

**注意**：验证集现在是没审核的自动标注，由教师自己生成，对教师有利。这一步只能说明训练没跑坏、和教师差多少；真正的比较在 T7。

```text
任务 T4：评估 v0。模型级（验证集 infos_val）和流水线级（验证场景整段跑）都做，并扫描上报阈值。
先按 docs/SERVER_TASKS.md 的"通用约定"做。前置：T3 完成。

1. 模型级，第 10、15、20 个 epoch：
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

停止条件：任何报错。
提交：data/reports/T4/、本文档。
报告（写进"T4 结果"）：
- 每个模型每个 epoch 的 merged/vehicle 一行（mAP、AP@0.5/1/2/4、分距离段 AP、ATE、ASE、AOE），以及 car、truck 各一行；
- compare_val.md 的两张表（AP 表、耗时表），以及每个模型的阈值表最后的推荐行；
- 一句话结论：v0 比 nuScenes Pillar 好还是差、差在哪个距离段。
```

### T4 结果

（待填）

---

## H1 人工审核车辆（人做）

只审车：不用管行人、骑行者，也不用改 car / truck / bus。规则见 [DOG_FINETUNE.md](DOG_FINETUNE.md) 第 13 节"审核只看车"，审核工具的用法见第 ④ 节和第 10 节。

1. **按轨迹（全部 35 个场景）**：
   - 在 Foxglove 里打开 `data/labels/<场景>/<场景>_review.mcap`，对照 `data/labels/<场景>/labels_tracks_vehicle.csv` 填 `action` 列；
   - 优先处理 `train_as = ignore` 的行：训练集 135 条、验证集 72 条。动作怎么填：

     | 情况 | 填 |
     |---|---|
     | 是车 | `ok` |
     | 不是车 | `drop` |
     | 看不清 | `ignore` |
     | 朝向反了 | 加 `+flip` |
     | 停着的车 | 加 `+static` |
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
       python tools/dog/sustechpoints_convert.py import --scene $r --out data/labels/$r/labels_sustech.pkl
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

验证集只有逐框审核过的帧（SUSTechPOINTS 每 5 帧取 1 帧，约 575 帧），这是预期的。
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
