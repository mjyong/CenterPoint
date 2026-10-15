# 长 bag 切分成多个序列（场景）

把一段很长的录制（例如 4090 服务器上 30 GB 的 `d20car_predict/rosbag0921/rosbag2_2026_09_21-16_56_59`）切成多个 30 s 左右的序列，转换并逐个检查，再进入 [DOG_FINETUNE.md](DOG_FINETUNE.md) 的 ② 教师推理、③ 自动标注。最后一节是可以直接交给服务器上 AI Agent 的任务说明。

三个工具，都在样例 bag 上测过（`tests/dog/test_bag_split.py`）：

| 步骤 | 工具 | 做什么 | 30 GB 估计耗时 |
|---|---|---|---|
| 1 体检 | `tools/dog/bag_survey.py` | 只读 INSPVAX 和雷达 / IMU 的时间戳（不解点云），出总览图，按质量自动切出场景，写 `segments.csv` | 读一遍文件，5–10 min |
| 2 切分转换 | `tools/dog/rosbag_to_sequence.py --segments` | 读一遍 bag，同时写出所有场景的序列；点云边读边写，内存不随 bag 变大 | 10–20 min，约 0.6 MB/帧 |
| 3 可视化验证 | `tools/dog/sequence_preview.py` | 每个序列一张检查图 + 数值检查（OK / WARN / FAIL），汇总成 `index.html` | 约 10 min |

耗时按样例 bag 推算：3.2 MB/帧，10 Hz；30 GB 若只有这三个话题，约 9,400 帧、16 分钟。bag 里如果还有相机等话题，帧数会少一些。

## 为什么切、怎么切

- 一个序列装 1 万帧不好用：审核表、SUSTechPOINTS 场景、训练 / 验证划分都是按"录制"组织的。切成 30 s 一段（约 300 帧），每段就是一个独立的录制名。
- **在哪里切**：有空档（雷达相邻两帧 > 0.3 s、IMU > 0.1 s、INS > 0.2 s）或 RTK 不是固定解（pos_type ≠ 56）的地方一定断开，这些时段整段丢掉。剩下连续可用的时段，按 `--scene-length` 等分；短于 `--min-length` 的不要。
- **验证集**：默认把可用时间的最后 20% 划为验证集，并丢掉它前面紧挨着的一段（`keep = 0`），避免同一时刻附近几乎一样的帧同时出现在训练集和验证集。`survey.json` 里的 `val_path_within_10m_of_train` 是验证集轨迹有多少落在训练集轨迹 10 m 以内。同一个地方走两遍也算泄漏，比例高就手动挑另一片区域的场景做验证集。
- **一个 bag 只是一次采集**：切出来的几十个场景是工作单位，不代表场景多样。数据量还是要靠多录几次不同的地方（[DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) 第 3 节）。
- **同一个 bag 的所有场景共用一个 ENU 原点**（整个 bag 第一个好的 INS 解），各场景的世界坐标对得上。

## 0. 准备

```bash
cd <仓库目录> && git pull origin claude/me/v01
pip install -r requirements_dog.txt
export PYTHONPATH=.
BAG=/<路径>/d20car_predict/rosbag0921/rosbag2_2026_09_21-16_56_59
NAME=0921_165659                      # 场景名前缀，场景为 ${NAME}_s000、${NAME}_s001 ...
df -h .                               # 序列约 6 GB；之后教师的帧（第 4 步）约 35 GB
head -60 $BAG/metadata.yaml           # 话题、条数、时长、分成几个 .db3 文件
```

默认话题和样例 bag 相同：`/robot/topic/rdap/top_point_cloud2`、`/rte/rdap/rtk/imus/data_raw`、`/rte/rdap/rtk/inspvax`。名字不同时，三个工具都用 `--lidar-topic` / `--imu-topic` / `--inspvax-topic` 指定。

## 1. 体检

```bash
python tools/dog/bag_survey.py --bag $BAG --out data/survey/$NAME --prefix $NAME
```

输出 `data/survey/$NAME/` 下的 `survey.png`、`survey.json`、`segments.csv`。屏幕上的摘要在样例 bag 上是这样的：

```
bag 3 s: lidar 27 frames (10.0 Hz, max gap 0.10 s, 3.2 MB/msg), imu 100 Hz, ins 100 Hz
  /robot/topic/rdap/top_point_cloud2       log - header stamp: median +0.104 s (5-95%: +0.102 .. +0.109)
  /rte/rdap/rtk/imus/data_raw              log - header stamp: median +0.001 s (5-95%: +0.000 .. +0.002)
  /rte/rdap/rtk/inspvax                    log - header stamp: median +0.000 s (5-95%: +0.000 .. +0.002)
INS good 100%, pos_type {56: 310}, path 6 m; usable 3 s in 1 stretches
```

**看什么**：

| 项 | 正常 | 不正常时 |
|---|---|---|
| 雷达 | 10 Hz，最大间隔约 0.1 s | 有空档时自动在空档处断开；空档很多说明驱动或磁盘丢帧 |
| IMU、INS | 各 100 Hz | 同上 |
| pos_type | 基本全是 56（RTK 固定解） | 非 56 的时段被丢掉。不建议用 `--pos-types 56 55` 放宽到浮点解，位姿误差会进标签 |
| 时钟（log − header） | 雷达约 +0.10 s（时间戳是一帧的开始，晚一帧才写进 bag），IMU、INS 约 0；5–95% 区间宽度 < 50 ms | 几秒的偏差、区间很宽、或前后两段不一样，说明有传感器没对上 GPS 时间。**停下来报告**，后面的检查发现不了这种问题 |
| `survey.png` 轨迹 | 一条连续的路径；蓝色是训练、橙色是验证、灰色不要 | 轨迹跳变说明 INS 跳了 |
| `survey.png` 速度 | 和实际走法相符 | |

**改 `segments.csv`**（可选）：`keep` 改成 0 去掉不要的场景（例如狗一直站着不动的场景，`note` 里会标 `static`，留几段就够），`split` 改成 `val` / `train` 调整划分，`name` 可以改得更好认。只有 `keep = 1` 的行会被转换。

## 2. 切分转换

```bash
python tools/dog/rosbag_to_sequence.py --bag $BAG --inspvax-topic /rte/rdap/rtk/inspvax \
    --lidar-topic /robot/topic/rdap/top_point_cloud2 --imu-topic /rte/rdap/rtk/imus/data_raw \
    --segments data/survey/$NAME/segments.csv --out data/seq
```

- 每个场景一个目录 `data/seq/${NAME}_sNNN/`，格式和单个 bag 转出来的一样。
- 场景里的雷达帧按头时间戳落在 `[t_start, t_end)` 内取，和 `segments.csv` 的 `scans` 列一一对应。IMU 和定位前后各多留 `--pad`（1 s），去畸变和位姿缓冲要用。
- `meta.json` 的 `segment` 里记着场景名、起止时间和 `split`，后面的工具会读。
- 每个场景各自从第一帧估计机身离地高度。
- 每个场景打印一行 `wrote N scans, M imu, K odom -> ...`。没有雷达帧或没有定位的场景会打印 `skipped`。
- 只重做部分场景时加 `--only ${NAME}_s003 ${NAME}_s007`。

`meta.json` 最后写，目录里没有它就说明这个场景没写完。中断后重跑同一条命令即可。

## 3. 可视化验证

```bash
python tools/dog/sequence_preview.py --seq "data/seq/${NAME}_s*" --out data/preview/$NAME
```

输出 `data/preview/$NAME/index.html`：所有场景一张表，带缩略图，按状态着色。另有 `checks.csv` / `checks.json` 和每个场景一张 `<场景>.png`。每个场景每隔 5 帧做一次和线上完全一样的预处理（去畸变、去自身点、重力对齐 / 地面找平、多帧拼接），再做下面这些检查：

| 检查 | 含义 | OK | WARN | FAIL |
|---|---|---|---|---|
| `align_m` | 相隔约 0.5 s 的两帧，竖直结构（墙、杆、停着的车）上的点在世界系里的水平错位，取中位数 | ≤ 0.10 m | ≤ 0.20 m | > 0.20 m |
| `ground_z_m` | 检测系里 3–12 m 内地面的高度（应在 z = 0） | ≤ 0.10 m | ≤ 0.20 m | 更大 |
| `tilt_deg` | 同一片地面拟合平面的倾角 | ≤ 2° | ≤ 4° | 更大 |
| `lidar_gap_s` / `imu_gap_s` / `odom_gap_s` | 最大间隔 | 0.2 / 0.05 / 0.1 s | 0.5 / 0.2 / 0.5 s | 更大 |
| `rtk` | 场景内（含前后 1 s）的 pos_type 全是 56 | 是 | 否 | |

参考值：
- 运动样例 bag（1.5–2.5 m/s）：`align` 在相隔 0.1 s 时是 0.02 m，相隔 0.5 s 时是 0.04 m；地面 0.02 m，倾角 1.4°。
- 静止样例 bag：`align` 0.003 m。
- 把雷达外参的 180° 偏航故意改错，`align` 会变成 1.03 m（FAIL），测试里就是这样验证这项检查的。

**每张图怎么看**：
- **左：世界系地图。** 抽检帧里竖直结构的点叠在一起，颜色是时间，黑线是轨迹。墙和杆应该是一条细线；出现双线、拖影或扇形，说明位姿、外参或时间同步有问题。
- **右上：模型输入。** 中间一帧，正是检测器看到的样子（5 帧拼接，检测系，灰色是旧帧），虚线框是 ±40.8 m 检测范围。
- **右中：侧视。** 地面要贴着红线 z = 0。远处地形本身起伏（坡、路沿）是正常的。
- **下：曲线和文字。** 速度和错位随时间变化（红虚线是 FAIL 线）、雷达帧间隔、数值和问题列表。

**不正常时**：
- `align` FAIL 或地图里墙是双线：先看同一 bag 里是不是所有场景都这样。全都这样，多半是外参或话题设置错了，对照样例 bag 的设置（`--inspvax-topic` 会自动用 180° 偏航）。个别场景这样，看 `survey.png` 里那段 INS 是否跳变，跳变的场景 `keep = 0` 后重转。
- `ground_z` / `tilt` WARN：看侧视图。近处地面整体高出或低于红线（平地上），说明离地高度或找平有问题；坡道、台阶上出现 WARN 是正常的。
- `rtk` WARN：场景边上有非固定解。体检时已经避开，通常只是前后补的 1 s；内容正常就可以保留。
- **这里查不出的**：雷达和 INS 之间一个固定的时间差，在直线匀速时所有帧一起偏，错位看不出来。这靠第 1 步的时钟检查。

## 4. 之后：教师推理 + 自动标注（批量）

验证通过后，对每个场景跑 [DOG_FINETUNE.md](DOG_FINETUNE.md) 的 ② ③。4090 上 Voxel + TTA 约 0.12 s/帧，1 万帧约 20 分钟；`--save-frames` 的帧约 3.6 MB/帧，1 万帧约 35 GB。`run_sequence.py` 默认还会写一个 Foxglove MCAP（`data/runs/<场景>/voxel.mcap`），可以直接在 Foxglove 里看每个场景的点云、检测和轨迹。

```bash
for d in data/seq/${NAME}_s*; do
  r=$(basename $d)
  python tools/dog/run_sequence.py --seq $d --detectors voxel --tta \
      --ckpt-voxel work_dirs/pretrained/nusc_voxel.pth --save-frames --device cuda:0 --out data/runs/$r
  python tools/dog/auto_label.py --results data/runs/$r/voxel/results.pkl --out data/labels/$r/labels.pkl
done
```

然后是人工审核（④）：
- 训练集场景至少按轨迹审核；
- 验证集场景（`segments.csv` 里 `split = val`）用 SUSTechPOINTS 逐框看，导出时加 `--every 5`。

审核完导出数据集，验证集就是 `split = val` 的那些场景：

```bash
REC=$(for d in data/runs/${NAME}_s*; do r=$(basename $d); \
      f=data/labels/$r/labels_sustech.pkl; [ -f $f ] || f=data/labels/$r/labels_reviewed.pkl; \
      echo --recording $d $f; done)
VAL=$(awk -F, 'NR>1 && $11==1 && $10=="val" {print $1}' data/survey/$NAME/segments.csv)
python tools/dog/export_dataset.py $REC --val-recordings $VAL --out data/dog_v001 --gt-db --skip-review
```

## 5. 交给服务器上 AI Agent 的任务说明

下面整段可以直接复制给 Agent（路径按服务器实际情况改）：

```text
任务：把 d20car_predict/rosbag0921/rosbag2_2026_09_21-16_56_59（约 30 GB 的 ROS 2 bag）切成多个 30 s 左右的序列，并做可视化验证。
仓库：CenterPoint，分支 claude/me/v01。先 git pull，再 pip install -r requirements_dog.txt，所有命令在仓库根目录执行，export PYTHONPATH=.。
按 docs/BAG_SPLIT.md 第 0–3 节执行，NAME=0921_165659：

0. 检查磁盘至少 50 GB 空闲；看 $BAG/metadata.yaml，确认三个话题存在（名字不同就用 --lidar-topic/--imu-topic/--inspvax-topic）。
1. python tools/dog/bag_survey.py --bag $BAG --out data/survey/$NAME --prefix $NAME
   打开 data/survey/$NAME/survey.png 看一遍。
   停止条件：任何话题的 "log - header stamp" 中位数超过 0.3 s，或 5-95% 区间宽于 0.05 s；可用时间不到总时长的一半。遇到就停下来报告，不要继续。
2. 不改 segments.csv（除非某段 note 是 static 且这样的段超过 5 段：只保留其中 2 段，其余 keep 改 0），然后
   python tools/dog/rosbag_to_sequence.py --bag $BAG --inspvax-topic /rte/rdap/rtk/inspvax \
       --lidar-topic /robot/topic/rdap/top_point_cloud2 --imu-topic /rte/rdap/rtk/imus/data_raw \
       --segments data/survey/$NAME/segments.csv --out data/seq
   核对：每个场景写出的帧数等于 segments.csv 的 scans 列。
3. python tools/dog/sequence_preview.py --seq "data/seq/${NAME}_s*" --out data/preview/$NAME
   打开 index.html 和状态最差的 3 张 PNG，以及 1 张 OK 的 PNG，按 docs/BAG_SPLIT.md 第 3 节判断。
   FAIL 的场景先按第 3 节"不正常时"排查；无法解释的不要删数据，记下来报告。
不要修改或删除原始 bag；不要进入第 4 步（教师推理），等人确认后再做。

最后报告：
- survey 的摘要行（时长、帧数、各话题频率、时钟检查三行、pos_type、可用时间、场景数和训练/验证时长、val_path_within_10m_of_train）；
- segments.csv 的行数和改动（如有）；
- 转换日志里每个场景的 wrote 行（或 skipped 行）；
- preview 的汇总行（N OK / N WARN / N FAIL）和每个非 OK 场景的 problems 列，以及你看图的结论（墙是否是细线、地面是否贴着 z = 0）；
- 用掉的磁盘空间（du -sh data/seq data/preview）和每步耗时。
```
