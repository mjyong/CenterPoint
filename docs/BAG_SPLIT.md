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
- 每个场景的机身离地高度（`base_height`）取每 10 帧估一次的中位数，`meta.json` 的 `base_height_sweeps` 里记着用了几帧和 p10–p90。只看一帧会受脚下地形影响：0921 的 29 个场景，第一帧估出来的值在 0.59–1.05 m 之间，而狗站着的真实高度约 0.72 m。2026-10 之前转的序列只用了第一帧，按 3.1 节补救，不用重转。
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
| `ins_step_m` | INS 位置的跳变：每个定位样本的位移减去它自己的速度 × Δt，取最大值。多帧拼接在跳变两侧对不齐 | ≤ 0.15 m | ≤ 0.5 m | 更大 |

参考值：
- 运动样例 bag（1.5–2.5 m/s）：`align` 在相隔 0.1 s 时是 0.02 m，相隔 0.5 s 时是 0.04 m；地面 0.02 m，倾角 1.4°。
- 静止样例 bag：`align` 0.003 m。
- 把雷达外参的 180° 偏航故意改错，`align` 会变成 1.03 m（FAIL），测试里就是这样验证这项检查的。
- 运动样例 bag 的 `ins_step_m` 是 0.33 m（WARN）：最后 1 s 狗在刹车，INS 报的速度先冲到 2.5 m/s，位置每 0.2 s 被往回拉 0.1–0.33 m（共 5 次）。

**每张图怎么看**：
- **左：世界系地图。** 抽检帧里竖直结构的点叠在一起，颜色是时间，黑线是轨迹。墙和杆应该是一条细线；出现双线、拖影或扇形，说明位姿、外参或时间同步有问题。
- **右上：模型输入。** 中间一帧，正是检测器看到的样子（5 帧拼接，检测系，灰色是旧帧），虚线框是 ±40.8 m 检测范围。
- **右中：侧视。** 地面要贴着红线 z = 0。远处地形本身起伏（坡、路沿）是正常的。
- **下：曲线和文字。** 速度和错位随时间变化（红虚线是 FAIL 线），红色三角是超过 5 cm 的 INS 位置跳变；雷达帧间隔、数值和问题列表。

**不正常时**：
- `align` FAIL 或地图里墙是双线：先看同一 bag 里是不是所有场景都这样。全都这样，多半是外参或话题设置错了，对照样例 bag 的设置（`--inspvax-topic` 会自动用 180° 偏航）。个别场景这样，看 `survey.png` 里那段 INS 是否跳变，跳变的场景 `keep = 0` 后重转。
- `ground_z` / `tilt` WARN：看侧视图。近处地面整体高出或低于红线（平地上），说明离地高度或找平有问题；坡道、台阶上出现 WARN 是正常的。
- `rtk` WARN：场景边上有非固定解。体检时已经避开，通常只是前后补的 1 s；内容正常就可以保留。
- 错位只在个别时刻冲高（中位数正常）：看同一时刻有没有红色三角（INS 位置跳变）、速度是不是在急变。有就是定位的问题，不是外参。没有、而且都在转弯时，用 3.2 节查时间差。
- **这里查不出的**：雷达和 INS 之间一个固定的时间差，在直线匀速时所有帧一起偏，错位看不出来。第 1 步的时钟检查只能发现大的偏差，小的（几十毫秒）用 3.2 节在转弯处查。

### 3.1 修正已转好的序列（不用重转）

```bash
python tools/dog/sequence_meta.py --seq data/seq/${NAME}_s*                       # 只看表
python tools/dog/sequence_meta.py --seq data/seq/${NAME}_s* --base-height auto    # 每 10 帧估一次取中位数
```

只改 `meta.json`。原来的值会在 `base_height_first_sweep` 里保留一份。改完重跑第 3 步的 preview。

站着走的场景，结果应在 0.65–0.80 m 之间。趴着的场景（例如 16:41 开头和结尾）约 0.2–0.3 m，是对的。站着走却明显偏离的场景，可以直接写死：`--seq <场景> --base-height 0.72`。

### 3.2 雷达与 INS 的时间差（转弯检查）

```bash
python tools/dog/time_offset_check.py --seq <几个有转弯的场景> --out data/preview/time_offset
```

**原理**：时间差 δ 会让每帧用到 δ 秒之前（或之后）的位姿。
- 匀速转弯或直行时，每帧偏的角度一样，帧与帧之间仍然对得齐。
- 角速度变化时（开始转、停止转、原地掉头），相隔 0.5 s 的两帧偏差不同。错位大约是：距离 × 角速度变化 × δ。例如 10 m 处、角速度变化 90°/s、δ = 0.05 s 时，错位是 0.8 m。

**做法**：取角速度变化 ≥ 20°/s 的帧对，把雷达时间平移 −0.15…+0.15 s，每个偏移都重算 preview 里的错位，取错位最小的偏移。角速度几乎不变的帧对作为对照，应保持平稳。

**验证**：仿真里给雷达注入 50 ms 延迟，工具找到 −0.050 s，转弯帧对的错位从 0.77 m 降到 0.007 m。见 `tests/dog/test_time_offset.py`。

**输出**：
- `time_offset.csv`：每个场景 × 每个偏移的错位；
- 每个场景一张 `<场景>.png`：左边是错位随偏移的曲线，右边是偏移 0 和最佳偏移下各帧对的错位，以及角速度变化；
- `time_offset.json`：每个场景和总体的最佳偏移，以及结论。

**怎么判断**：
- 几个场景都在同一个非 0 的偏移处有明显最小值（|δ| ≥ 0.02 s，错位比 0 处小 1.5 倍以上），说明时钟有固定偏差。修正方法：
  - 写进序列：`python tools/dog/sequence_meta.py --seq data/seq/${NAME}_s* --lidar-time-offset <δ>`。`SequenceReader` 读的时候会加上，preview、自动标注、导出都会用到它，不用重转。
  - 写进线上节点的参数：`-p lidar_time_offset:=<δ>`。
- 曲线平，或者最小值就在 0：时钟没问题，转弯处的尖峰另有原因（多半是 INS 位置跳变，见上面的 `ins_step_m`）。
- 各场景的最佳偏移不一样：不是时钟问题，结论会直接写明。

### 3.3 0921 两份数据的复核结论（2026-10）

- **时钟**：IMU 陀螺的 z 轴和 INS 航向变化率在 0 延迟处相关最高（0.91），方向一致，所以 INS 航向没有延迟。
- **对齐**：中位数 0.003–0.083 m。尖峰集中在急转、急停的时刻，最大是 164127_s003 的 6.2 m（bag 第 105 s）。运动样例 bag 正好是这一刻之前的 3 s：INS 速度冲到 2.5 m/s 后骤降，位置每 0.2 s 被往回拉，最多 0.33 m。所以这个尖峰来自定位，不是外参或时钟。服务器上的 `ins_step_m` 会给出所有场景里这种情况有多频繁。
- **离地高度**：第一帧估计的值在 0.59–1.05 m 之间（中位数 0.71 m）。用 3.1 节改成多帧中位数。
- **验证集**：16:41 整份做验证只有 173 s，其中 s000 狗趴着不动、s005 基本没动，有效约 100 s；和 16:56 的路线是否重合也不知道。用 3.4 节调整。
- **INS 急停误差**（第二轮复核）：急停时 INS 位置超前约 0.45 m，并非落后；往回拉的跳变方向是对的，但没拉够。处理见 3.5 节。
- **线上节点**：之前没开 `auto_ground` / `auto_level`，离地高度默认 0.45 m，和训练数据不一致。现在用 `-p robot:=kygi` 一次设好（`dog_perception/node_config.py`）。`tests/dog/test_ros2_node.py` 把样例 bag 回放进节点，检查检测系点云和训练序列是否一致。

### 3.4 训练 / 验证划分（多份录制）

体检给每份录制各自分了训练和验证。录制有好几份时，用 `split_check.py` 统一检查和调整：

```bash
python tools/dog/split_check.py --seq data/seq/0921_* --out data/survey/split \
    --holdout 0921_165659 --holdout-s 90            # 先不加 --apply，只看结果
```

所有场景的轨迹都按各自 `meta.json` 里的 lat0/lon0 换到同一个 ENU 坐标系，再按顺序做三件事：

1. **静止的验证场景去掉**：机器人运动（> 0.3 m/s）的时间不到 30% 的验证场景，标为 `drop`。狗趴着或站着时，30 秒里同一个视角重复约 300 次，验证指标几乎只在量这一个视角。
2. **从训练录制里留出一段做验证**（`--holdout`）：在这份录制里找连续的训练场景，总长不少于 `--holdout-s`，选离其余训练路线最远的那段，标为 `val`；它两侧各一个场景标为 `drop`，作为缓冲（前后几秒看到的是同一批物体）。
   - 路线离得一样远时，选需要缓冲秒数少的（录制开头或结尾的一段只要一个缓冲）；再一样，选路程长的。
3. **空间重叠**：一个验证场景和一个训练场景，只要其中一条路线有超过 10%（`--max-share`）落在另一条路线的 10 m（`--radius`）以内，就算重叠（同一条街、同一批停着的车，前后只差几分钟）。对每个有重叠的验证场景，去掉代价小的一方：
   - 和它重叠的训练场景总时长比它长，就去掉这个验证场景；
   - 否则去掉那些训练场景。

**输出**（`--out`）：
- `split_check.csv`：每个场景的录制、时长、路程、运动占比、调整前后的划分、原因、路线落在另一方 10 m 内的比例；
- `split_check.png`：地图，蓝色训练、橙色验证、灰色去掉；
- `split_check.json`：调整前后的合计，以及验证路线落在训练路线 10 m 内的比例。

**写入**：加 `--apply` 后，写出 `split.csv`（每个场景 `train` / `val` / `drop`），并改每个 `meta.json` 的 `segment.split`（体检给的原值在 `segment.split_survey` 里保留一份）。重跑总是从体检的原值算起，结果不变。第 4 步按 `split.csv` 跑。

**参考目标**：
- 验证集里机器人在动的时间 ≥ 150 s，占全部运动数据的 12–25%；
- 验证路线落在训练路线 10 m 内的比例 ≤ 5%；
- 因为重叠和缓冲去掉的训练数据不超过原训练时长的 15%。

### 3.5 位姿融合：激光里程计 + RTK（`refine_poses.py`）

**问题**：狗急停时，kygi INS 的速度会冲过头（2.5 m/s），位置跑到真实位置前面约 0.5 m，并保持 1 秒以上；随后几次往回拉只修正了一部分。运动样例 bag 在 2.1 s 时超前 0.44 m，这是把激光帧两两配准测出来的。

**影响**：
- 这段时间拼接的 5 帧点云会有重影；
- 自动标注把静止物体固定在世界系里，那几帧的框会和点云错开约 0.5 m。

预览里的 `align_m` 看不出这个问题：误差沿行进方向，而墙大多和路平行，帧沿着墙滑动时墙照样对得上。所以加了 `shift` 检查（见下）。

```bash
python tools/dog/refine_poses.py --seq data/seq/${NAME}_s* --plots data/preview/poses
```

**每个场景的处理**：
1. KISS-ICP 激光里程计：用和线上相同的去重、去自身点，以及逐点时间。它给出帧与帧之间的运动，但会慢慢漂移；在没有特征的长墙边可能停住不动。
2. INS 健康判定：在 0.5 s 窗口内，看 INS 位置变化和它自己的速度积分是否一致。急停时两者差 1 m/s，正常走只差 0.1 m/s。超过 0.4 m/s 判为不可信，前后各扩 0.3 s；夹在两段不可信之间、短于 1 s 的"可信"小段，也算不可信。
3. 东、北两个方向分别做最小二乘融合：
   - INS 可信的地方由 INS 定住轨迹，激光只补帧与帧之间的运动；
   - INS 不可信的地方 INS 权重为 0，由激光把前后两段可信的 INS 连起来；
   - 两边都用 Huber 稳健权重，所以激光卡住或跳变、INS 偶尔的离群点，影响都会被降低。
4. 自检：不可信段两端都有可信 INS 时，检查激光走完这段后能否和对端的 INS 闭合。差距超过 `--gap-tol`（0.25 m 加上段长的 5%），就把这段记为**不可靠**（`pose_refine.bad_intervals`）：两种来源在这里都不可信，这段的帧不要标注。不可信段如果挨着序列开头或结尾，就无法检查，只记为 `unverified_intervals`。
5. 写回：
   - `odom.npy` 保留 INS 原来的行（100 Hz 时间点、只有航向的姿态、高度），东、北坐标换成融合后的轨迹（在帧与帧之间线性插值），速度换成它的导数。去畸变和 IMU 位姿传播的用法和以前完全一样；
   - INS 原始数据另存为 `odom_ins.npy`，以后每次都从它读，所以可以放心重跑；
   - `meta.json` 加上 `pose_source = "kiss-icp+ins"` 和 `pose_refine`（修正量统计、不可靠和未验证的时间段、可信段里激光 1 s 位移和 INS 的差）。

**验证**：
- 运动样例 bag 上，所有相隔 0.5 / 1 / 2 s 的帧对，用二维配准测出的位姿偏差：

  | 帧间隔 | INS（中位 / p90） | 融合后（中位 / p90） |
  |---|---|---|
  | 0.5 s | 0.11 / 0.32 m | 0.04 / 0.07 m |
  | 1 s | 0.25 / 0.55 m | 0.06 / 0.14 m |
  | 2 s | 0.50 / 0.64 m | 0.11 / 0.18 m |

- 合成数据（有真值）：INS 超前 0.5 m 的那段被修正到误差 < 6 cm；激光在这段失效时，这段会被标为不可靠。见 `tests/dog/test_refine_poses.py`。
- 不做 INS 健康判定（激光和 INS 同等对待）时，1 s 帧对的 p90 是 0.37 m：激光在走廊里会少算位移，健康的 INS 必须拿来定住轨迹。

**诊断输出**（`--plots` 目录）：
- 每个场景一张三栏图：轨迹；修正量、INS 不可信时段、被拒的激光步；三种位姿各自的 `shift`（纯 INS / 纯激光里程计 / 融合）；
- `<场景>_diag.npz`：几十 KB，存着各来源的位姿、健康判定和被拒的步，不用点云就能重新分析；
- `refine_summary.csv`：汇总表。

**怎么读三种位姿的 `shift` 对比**：
- 纯激光差（> 0.2 m）：激光里程计本身在这个场景不行，例如转角、开阔地、动态物体多；
- 纯激光好、融合反而差：融合有问题。

**0921 第一次融合（5.4）**：
- 29 + 6 个场景里，大多数的 `shift_p90_m` 下降了，但转角场景仍是 FAIL（s020 0.59 m、s026 0.61 m、s024 0.52 m、164127_s003 0.49 m）；s011（0.39→0.51 m）和 s027（0.33→0.42 m）反而变差；
- 大 bag 里 INS 不可信的时间比样例多得多（s020 约 60%），激光要单独接管 5–10 s；这些时段的融合轨迹有锯齿。
- 找到的一个原因：激光步是否失效，原来是和 INS 的位置步长比的。但 INS 不可信时，它的位置正在往回跳（最多 0.45 m），结果正常的激光步被拒掉，换成了带跳变的 INS 步长。
- 已改为：激光步和 INS 速度积分、以及前后几步的激光自己**都**对不上，才算失效；替换时用 INS 速度积分。见 `test_fuse_keeps_good_odometry_while_the_ins_jumps`。

**预览的 `shift` 检查**：对相隔约 1 s 的两帧，取离地 0.3–3 m 的点（墙、杆、树、车），在 ±0.8 m 范围内做二维搜索，找到把一帧叠到另一帧上最合适的平移量。这就是激光看到的两帧位姿误差，沿行进方向的误差也能测出来。状态按 p90 判定：≤ 0.20 m 为 OK，≤ 0.40 m 为 WARN，更大为 FAIL。运动样例 bag 用 INS 位姿时是 0.51 m（FAIL），融合后是 0.12 m（OK）。

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

用 3.4 节的 `split.csv` 时，循环换成它列出的场景（`drop` 的不跑）：

```bash
SPLIT=data/survey/split/split.csv
for r in $(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT); do
  python tools/dog/run_sequence.py --seq data/seq/$r --detectors voxel --tta \
      --ckpt-voxel work_dirs/pretrained/nusc_voxel.pth --save-frames --device cuda:0 --out data/runs/$r
  python tools/dog/auto_label.py --results data/runs/$r/voxel/results.pkl --out data/labels/$r/labels.pkl
done
```

审核完导出数据集，验证集就是 `split = val` 的那些场景：

```bash
REC=$(for d in data/runs/${NAME}_s*; do r=$(basename $d); \
      f=data/labels/$r/labels_sustech.pkl; [ -f $f ] || f=data/labels/$r/labels_reviewed.pkl; \
      echo --recording $d $f; done)
VAL=$(awk -F, 'NR>1 && $11==1 && $10=="val" {print $1}' data/survey/$NAME/segments.csv)
python tools/dog/export_dataset.py $REC --val-recordings $VAL --out data/dog_v001 --gt-db --skip-review
```

用 `split.csv` 时：

```bash
REC=$(for r in $(awk -F, 'NR>1 && $2!="drop" {print $1}' $SPLIT); do \
      f=data/labels/$r/labels_sustech.pkl; [ -f $f ] || f=data/labels/$r/labels_reviewed.pkl; \
      echo --recording data/runs/$r $f; done)
VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
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

### 5.2 第二轮：复核后的修正（0921 两份数据，序列已经转好）

```text
任务：对已转好的 0921 序列做三件事：修正离地高度、用新指标重跑预览、检查雷达与 INS 的时间差。不重转 bag，不进入第 4 步。
仓库：CenterPoint，分支 claude/me/v01。先 git pull，所有命令在仓库根目录执行，export PYTHONPATH=.。
序列：data/seq/0921_165659_s000..s028（训练）、data/seq/0921_164127_s000..s005（验证）。

1. 离地高度（docs/BAG_SPLIT.md 3.1 节）：
   python tools/dog/sequence_meta.py --seq data/seq/0921_* --base-height auto
   核对：站着走的场景应在 0.65-0.80 m；164127_s000、s005（狗趴着）约 0.2-0.3 m 是对的。
   站着走却在范围外的场景，单独执行 --seq <场景> --base-height 0.72。
2. 重跑预览（新增了 INS 位置跳变 ins_step_m，图里红色三角）：
   python tools/dog/sequence_preview.py --seq "data/seq/0921_165659_s*" --out data/preview/0921_165659
   python tools/dog/sequence_preview.py --seq "data/seq/0921_164127_s*" --out data/preview/0921_164127
   核对：所有场景的 ground_z_m 应比上一轮更接近 0；看 165659_s026、s019、s020 和 164127_s003 这 4 张 PNG，
   错位尖峰是否和红色三角（INS 跳变）或速度急变在同一时刻。
3. 时间差（3.2 节）：
   python tools/dog/time_offset_check.py --out data/preview/time_offset \
       --seq data/seq/0921_164127_s003 data/seq/0921_164127_s004 data/seq/0921_165659_s019 \
             data/seq/0921_165659_s020 data/seq/0921_165659_s026 data/seq/0921_165659_s018
   每个场景大约几分钟。只做检查，不要用 --lidar-time-offset 改序列，等人确认。
4. 把 data/preview（含 time_offset）提交并推送到 claude/me/v01，提交信息写清楚做了什么。不要提交 data/seq。

最后报告：
- sequence_meta.py 打印的表（每个场景 base_height 旧值 -> 新值、p10-p90）；
- 两份 preview 的汇总行和每个非 OK 场景的 problems；所有场景的 ins_step_m 和 ins_steps_over_10cm
  （checks.csv 里有），以及 >0.15 m 的场景数；
- 4 张 PNG 的结论：错位尖峰是否对应 INS 跳变或速度急变；
- time_offset.json 的 verdict，每个场景的 best_offset、turn_median_at_0、turn_median_at_best、n_turn；
- 每步耗时。
```

### 5.3 第三轮：调整训练 / 验证划分（0921 两份数据）

可以接在 5.2 后面一起做，两者互不影响。

```text
任务：按 docs/BAG_SPLIT.md 3.4 节调整 0921 两份数据的训练 / 验证划分。不重转 bag，不进入第 4 步。
仓库：CenterPoint，分支 claude/me/v01。先 git pull，所有命令在仓库根目录执行，export PYTHONPATH=.。
现状：0921_165659_s000..s028 全是训练；0921_164127_s000..s005 全是验证，其中 s000 是狗趴着不动（moving 0.00），
s005 也基本没动（0.27），有效验证只有约 100 s；两份录制是不是走过同一段路还不知道。

1. 先只看结果（不写入）：
   python tools/dog/split_check.py --seq data/seq/0921_* --out data/survey/split --holdout 0921_165659 --holdout-s 90
   打开 data/survey/split/split_check.png，看两份录制的路线是否重合、留出的那段在哪里。
2. 用打印的 after 行和最后一行，对照 3 个目标：
   a) 验证集运动时间（val ... moving）>= 150 s，且 val = 12-25% of the moving data；
   b) val path within 10 m of train <= 5%；
   c) 因为缓冲和重叠被 drop 的训练场景（split_before=train 且 split_after=drop）合计 <= 训练原时长的 15%。
   如果 a) 不满足，改 --holdout-s 120 再跑一次；
   如果 c) 不满足，改 --holdout-s 60 再跑一次；
   都不满足就停下，不要 --apply，报告三次的结果。
3. 满足时，用同样的参数加 --apply 再跑一次（写出 data/survey/split/split.csv，并更新每个 meta.json 的 segment.split）。
4. 重跑两份 preview，让 index.html 的 split 列反映新的划分：
   python tools/dog/sequence_preview.py --seq "data/seq/0921_165659_s*" --out data/preview/0921_165659
   python tools/dog/sequence_preview.py --seq "data/seq/0921_164127_s*" --out data/preview/0921_164127
5. 提交 data/survey/split（split.csv、split_check.csv、split_check.json、split_check.png）和 data/preview，
   推送到 claude/me/v01。不要提交 data/seq。

最后报告：
- 每次运行打印的整张表和 before / after 两行、最后一行；
- 最终用的 --holdout-s，哪些场景是 val、哪些 drop 及原因；
- 你看 split_check.png 的结论：两份录制的路线是否有重合，在哪里；
- 每步耗时。
```

### 5.4 第四轮：位姿融合（0921 两份数据，接在 5.3 之前做）

```text
任务：用激光里程计 + RTK 融合修正 0921 所有序列的位姿，并用新的 shift 检查对比修正前后。不重转 bag，不进入第 4 步。
仓库：CenterPoint，分支 claude/me/v01。先 git pull，再 pip install -r requirements_dog.txt（新增 kiss-icp），
所有命令在仓库根目录执行，export PYTHONPATH=.。

1. 修正前的基线（新版预览多了 shift 检查）：
   python tools/dog/sequence_preview.py --seq "data/seq/0921_165659_s*" --out data/preview/0921_165659_ins
   python tools/dog/sequence_preview.py --seq "data/seq/0921_164127_s*" --out data/preview/0921_164127_ins
2. 融合（docs/BAG_SPLIT.md 3.5 节），记下打印的整张表：
   python tools/dog/refine_poses.py --seq data/seq/0921_* --plots data/preview/poses
3. 修正后的预览：
   python tools/dog/sequence_preview.py --seq "data/seq/0921_165659_s*" --out data/preview/0921_165659
   python tools/dog/sequence_preview.py --seq "data/seq/0921_164127_s*" --out data/preview/0921_164127
4. 看图：data/preview/poses/ 里修正量最大的 3 个场景，以及修正后 shift_p90_m 仍然最高的 3 个场景的预览 PNG。
5. 提交 data/preview（含 poses 和 *_ins 目录），推送到 claude/me/v01。不要提交 data/seq。
   如果 refine_poses.py 报错或某个场景 LO 1s p90 > 0.5 m，不要改参数重跑，记下来报告。

最后报告：
- refine_poses.py 打印的整张表，以及所有 "pose bad" 行（不可靠时段）；
- 每个场景修正前后的 shift_p90_m 和 status（两份 checks.csv 对比），以及修正后 shift_p90_m > 0.2 m 的场景；
- 第 4 步看图的结论（修正后的轨迹是否平滑、地图里的墙和杆是否更清晰）；
- 每步耗时。
```

### 5.5 第五轮：修正后重新融合，并输出诊断（0921）

```text
任务：用修正后的 refine_poses.py 重新融合 0921 所有序列，提交诊断文件。不重转 bag，不进入第 4 步。
仓库：CenterPoint，分支 claude/me/v01。先 git pull，所有命令在仓库根目录执行，export PYTHONPATH=.。

1. 重新融合（输入自动取 odom_ins.npy，不会叠加上一次的结果），记下打印的整张表：
   python tools/dog/refine_poses.py --seq data/seq/0921_* --plots data/preview/poses
2. 重跑预览：
   python tools/dog/sequence_preview.py --seq "data/seq/0921_165659_s*" --out data/preview/0921_165659
   python tools/dog/sequence_preview.py --seq "data/seq/0921_164127_s*" --out data/preview/0921_164127
3. 提交 data/preview/poses（PNG、*_diag.npz、refine_summary.csv）和两个预览目录，推送到 claude/me/v01。
   不要提交 data/seq。

最后报告：
- refine_poses.py 打印的整张表（含 "LO rej" 和 "shift p90 ins / lo / fused" 两列）和所有 "pose bad" 行；
- shift_lo_p90 > 0.2 m 的场景（激光里程计本身不行），以及 fused 比 lo 差 0.05 m 以上的场景；
- 预览里每个场景 shift_p90_m 和 status，与上一轮（5.4）对比；
- 每步耗时。
```
