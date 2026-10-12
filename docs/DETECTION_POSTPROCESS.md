# 检测后处理：从 61 224 个候选到最终检测

本文按真实数据，一步一步讲清楚：6 个 SepHead 输出的 6 × 102 × 102 个候选，如何变成检测器最终输出的框，以及每一步为什么这样做。它是 [DOG_DATAFLOW.md](DOG_DATAFLOW.md) 4.3 节中 `predict → postprocess → filter_detections` 三行的展开，网络本身见 [CENTERPOINT_NETWORK.md](CENTERPOINT_NETWORK.md)。

**数据**：静止样例 bag（`samples/rosbag2_2026_09_21-16_41_27`）第 10 帧，5 帧累积共 198 710 个点，场景是停满车的停车场，有公交车和行人。检测器是 nuScenes 预训练 Pillar，用的是当前代码（外参 180°、地面找平、局部地面过滤）。数字会随帧变化，任意帧都可以复现：

```bash
python tools/dog/trace_postprocess.py --bag samples/rosbag2_2026_09_21-16_41_27 --frame 10 [--task 0]
```

这个工具会把每一步重新实现一遍，并逐框核对：结果与 det3d 原版 `CenterHead.predict`、与检测器自身的输出都完全一致。

---

## 0. 总览

```
6 个 task × 102 × 102 = 61 224 个候选（每个像素一个，取本 task 内分数最高的类别）
  │ ① 分数 > 0.1                                        349
  │ ② 中心在 post_center_limit_range 内                  349   这个范围就是网格本身，不会去掉框
  │ ③ 每个 task 内部做 circle NMS                        115
  │ ④ 每个 task 最多保留 83 个                           115   都没到上限
  ▼ 合并 6 个 task，label 加偏移，得到 0–9 共 10 类        box3d_lidar (115, 9)、scores (115,)、label_preds (115,)
  │ ⑤ 10 类 → 3 类，丢掉 barrier 和 traffic_cone          96
  │ ⑥ det3d 框 → 标准框，M 系 → D 系                       96
  │ ⑦ 合并类别后的类内 circle NMS（车 2 m，骑行者 0.9 m）   88   postprocess 的输出 (88, 7)
  │ ⑧ 地面支撑：框底相对局部地面在 −1.2 ~ +0.7 m 之内      85
  │ ⑨ 当前帧点数：车 ≥ 5，行人 / 骑行者 ≥ 3                 70
  │ ⑩ 跨类去重：中心落在另一类更高分框内的，去掉           59   检测器的输出 (59, 7)
  ▼
  ⑪ 按类别阈值上报（车 0.35、行人 0.30、骑行者 0.40）：18 个上报；另外 41 个低分框只给跟踪器第二阶段用
```

前四步在 det3d 的 `CenterHead.predict` 里（`center_head.py:294`、`:451`），第 ⑤–⑦ 步在 `CenterPointDetector.postprocess`（`centerpoint.py:211`），第 ⑧–⑩ 步在 `filter_detections`（`filters.py`）。

每个 task 的数量：

| task | 类别 | 候选 | 分数 > 0.1 | 范围内 | circle NMS 后 | 截断 83 后 | 各类别 |
|---|---|---|---|---|---|---|---|
| 0 | car | 10 404 | 97 | 97 | 24 | 24 | car 24 |
| 1 | truck + construction_vehicle | 10 404 | 65 | 65 | 13 | 13 | truck 6、construction_vehicle 7 |
| 2 | bus + trailer | 10 404 | 63 | 63 | 11 | 11 | bus 5、trailer 6 |
| 3 | barrier | 10 404 | 6 | 6 | 6 | 6 | barrier 6 |
| 4 | motorcycle + bicycle | 10 404 | 79 | 79 | 34 | 34 | motorcycle 21、bicycle 13 |
| 5 | pedestrian + traffic_cone | 10 404 | 39 | 39 | 27 | 27 | pedestrian 14、traffic_cone 13 |

---

## 1. 网络输出：6 个 SepHead

shared_conv 之后有 6 个 task head（SepHead），每个负责一组尺寸相近的类别，各自输出 6 张 102 × 102 的图。102 × 102 网格覆盖 M 系 ±40.8 m，每格 0.8 m（柱子 0.2 m × 输出步长 4）。

| 输出 | 通道 | 含义（每个像素） |
|---|---|---|
| `hm` | C_t | 每个类别的中心热力图 logit，sigmoid 之后是"本格是某类物体中心"的分数 |
| `reg` | 2 | 中心相对格子左上角的亚格偏移（0–1 格） |
| `height` | 1 | 框中心的 z（M 系，m） |
| `dim` | 3 | log(w, l, h)，取 exp 之后得到尺寸（m） |
| `rot` | 2 | (sin r, cos r)，r 是 det3d 约定的朝向角 |
| `vel` | 2 | 对地速度 (vx, vy)（M 系，m/s） |

6 个 task 的 C_t 依次为 1, 2, 2, 1, 2, 2，对应 car / truck + construction_vehicle / bus + trailer / barrier / motorcycle + bicycle / pedestrian + traffic_cone，合计 10 个 nuScenes 类别。

## 2. 第一步：逐像素解码（以最强的那辆车为例）

`predict` 把每个 task 的每个像素都解码成一个框（`center_head.py:342-417`）。下面是 task 0（car）中分数最高的像素，第 71 行、第 50 列：

| 量 | 原始输出 | 换算 | 结果 |
|---|---|---|---|
| 分数 | hm logit 1.244 | sigmoid | **0.776** |
| x | reg_x = 0.748 | (列 + reg_x) × 0.8 − 40.8 = (50 + 0.748) × 0.8 − 40.8 | **−0.20 m** |
| y | reg_y = 0.067 | (行 + reg_y) × 0.8 − 40.8 = (71 + 0.067) × 0.8 − 40.8 | **16.05 m** |
| z | height = −1.445 | 直接就是框中心 z（M 系） | **−1.445 m** |
| 尺寸 | dim = (0.724, 1.607, 0.588) | exp | **w 2.06、l 4.99、h 1.80 m** |
| 朝向 | rot = (sin, cos) = (0.993, 0.020) | r = atan2(sin, cos) | **r = 1.550 rad** |
| 速度 | vel = (−0.018, −0.007) | 直接使用 | **基本静止** |

得到 det3d 格式 `box9 = [x, y, z, w, l, h, vx, vy, r] = [−0.20, 16.05, −1.445, 2.06, 4.99, 1.80, −0.018, −0.007, 1.550]`，坐标系为 M（虚拟 nuScenes 雷达系，车头方向是 +y）。

一个 task 有 C_t 个类别时（例如 task 5 的 pedestrian 和 traffic_cone），每个像素取分数最高的那一类作为它的类别和分数（`hm.max(-1)`）。所以每个像素恰好产生一个候选，6 个 task 共 61 224 个。

## 3. 第二步：分数阈值（61 224 → 349）

`score > test_cfg.score_threshold = 0.1`（`center_head.py:461`）。

这 349 个候选的分数中位数只有 0.15，75% 分位 0.23，90% 分位 0.41，99% 分位 0.74，最高 0.91。阈值有意设得这么低：分数 0.1–0.35 的框虽然大多不可信，但跟踪器第二阶段要用它们来延续已有的轨迹（见第 ⑪ 步）。

**为什么一个物体会留下好几个候选。** CenterPoint 在推理时**没有** CenterNet 那种 3 × 3 max-pool 取局部峰值的步骤。热力图在物体中心附近是一个"小山包"，邻格的分数同样可能超过 0.1。上面那辆车峰值附近的 3 × 3 格（sigmoid 后）：

```
0.050  0.649  0.232
0.071  0.776  0.385      ← 中间是峰值 0.776
0.010  0.027  0.017
```

一辆车就有 4 个格子超过 0.1。所以 task 0 有 97 个候选，实际只有 24 个物体。这些重复要靠第 ④ 步的 circle NMS 去掉。

## 4. 第三步：范围过滤（349 → 349）

框中心必须落在 `post_center_limit_range` 内（`center_head.py:462-463`）。`configure_for_dog` 把它设成 `[−40.8, −40.8, −10, 40.8, 40.8, 10]`：xy 与网格范围相同，z 为 ±10 m。框中心只有在网格边缘、`reg` 把它推出界时才会被去掉，所以这一帧一个也没去掉。

如果只想输出更近的框（例如 ±30 m），改的就是这个范围。

## 5. 第四步：每个 task 内部的 circle NMS（349 → 115）

`_circle_nms`（`center_head.py:498`，`det3d/core/utils/circle_nms_jit.py`）的做法：
1. 把本 task 的候选按分数从高到低排序；
2. 依次取出当前最高分的框保留，把与它中心距离在半径内的、分数更低的框全部抑制；
3. 重复，直到处理完所有候选；
4. 最后只取前 83 个（`nms_post_max_size`）。

注意 det3d 的 `min_radius` 是**平方距离**（代码里比较的是 `dx² + dy² <= thresh`）：

| task | min_radius（m²） | 半径（m） | 理由 |
|---|---|---|---|
| car | 4.0 | 2.0 | 车长约 4.5 m，两辆车中心很难近于 2 m |
| truck + cv | 12.0 | 3.46 | 车更长 |
| bus + trailer | 10.0 | 3.16 | |
| barrier | 1.0 | 1.0 | |
| moto + bike | 0.85 | 0.92 | |
| ped + cone | 0.175 | 0.42 | 行人之间可以挨得很近 |

nuScenes 原配置用的是旋转 IoU NMS，需要编译 CUDA 算子。`configure_for_dog` 换成了 circle NMS，半径按上表从各 task 内的类别取，这也是 CenterPoint 官方 nuScenes 提交用的设置。

以那辆车为例：task 0 里还有 3 个候选在它 2 m 之内，全部被抑制。

| 像素偏移（行，列） | 分数 | 与峰值框的中心距离 |
|---|---|---|
| (−1, 0) | 0.649 | 0.08 m |
| (0, +1) | 0.385 | 0.25 m |
| (−1, +1) | 0.232 | 0.23 m |

可以看到，相邻格子的中心经过 `reg` 修正后几乎落在同一点上（0.08–0.25 m），这正是用中心距离去重有效的原因。

NMS 后 6 个 task 共 115 个框，没有一个 task 碰到 83 的上限，整帧上限是 6 × 83 = 498。

**不同 task 之间不做 NMS。** 同一辆车可能被 car 头和 truck 头各出一个框，这要到第 ⑦ 步处理。

## 6. 合并 6 个 task（115 个，10 类）

把 6 个 task 的结果按顺序拼接（`center_head.py:428-447`）。label 加上前面所有 task 的类别数作为偏移，变成全局类别 id 0–9：car 0、truck 1、construction_vehicle 2、bus 3、trailer 4、barrier 5、motorcycle 6、bicycle 7、pedestrian 8、traffic_cone 9。

输出 `box3d_lidar (115, 9) f32`、`scores (115,)`、`label_preds (115,) i64`。各类数量：car 24、truck 6、construction_vehicle 7、bus 5、trailer 6、barrier 6、motorcycle 21、bicycle 13、pedestrian 14、traffic_cone 13。

## 7. 第五到七步：postprocess（115 → 88）

`CenterPointDetector.postprocess`（`centerpoint.py:211`）。

**⑤ 10 类 → 3 类**（`NUSC_TO_DOG`）：
- car、truck、construction_vehicle、bus、trailer → vehicle；
- motorcycle、bicycle → cyclist；
- pedestrian → pedestrian；
- barrier、traffic_cone 没有映射，label 为 −1，直接丢掉（这里共 19 个：barrier 6、cone 13）。

剩 96 个。

**⑥ 框格式和坐标系**：
- `det3d_to_standard`：`[x, y, z, w, l, h, vx, vy, r]` → `[x, y, z, l, w, h, yaw]`，其中 `yaw = −r − π/2`，并交换 w 和 l。那辆车：yaw_M = −1.550 − 1.571 = −3.121。
- `ModelFrame.boxes_from_model`：M → D。M 是 D 绕 z 转 +90° 再下移 1.84 m 得到的，所以：
  - x_D = y_M = 16.05
  - y_D = −x_M = 0.20
  - z_D = z_M + 1.84 = 0.395
  - yaw_D = yaw_M − π/2 = −4.692 → 1.591
  - 速度做同样的旋转

结果是 D 系 `[16.05, 0.20, 0.395, 4.99, 2.06, 1.80, 1.591]`：车在正前方 16 m，朝向约 91°，横着停。

**⑦ 合并类别后的类内 circle NMS**（`classwise_circle_nms`，`MERGED_NMS_RADIUS`）。这里的半径单位是**米**：车 2.0 m，骑行者 0.9 m，行人不做。

| 类别 | 之前 | 之后 |
|---|---|---|
| vehicle | 48 | 40 |
| pedestrian | 14 | 14（不做） |
| cyclist | 34 | 34 |

被去掉的 8 个车辆框，都是不同的头在同一辆车上各出了一个框：

| 一个框 | 另一个框 | 中心距离 | 保留 |
|---|---|---|---|
| car 0.41 | truck 0.56 | 0.09 m | truck 0.56（作为 vehicle） |
| car 0.10 | truck 0.38 | 0.13 m | truck 0.38 |
| car 0.55 | truck 0.15 | 0.80 m | car 0.55 |
| construction_vehicle 0.11 | bus 0.24 | 1.74 m | bus 0.24 |

合并成 vehicle 以后，原来的 car / truck / bus 区分已经不重要，保留分数高的那一个即可。行人没有做这一步：在 nuScenes 片段评估中发现，它会导致同一个人身上出现两条轨迹（见 [NUSC_CLIP_EVAL.md](NUSC_CLIP_EVAL.md) 第 5 节）。

postprocess 输出 `boxes (88, 7) f64`、`velocities (88, 2) f64`，D 系，yaw 从 +x 起逆时针为正。

## 8. 第八到十步：filter_detections（88 → 59）

这三道过滤对所有分数一视同仁，去掉几何上不可能的框（`filters.py`）。

**⑧ 地面支撑（88 → 85）**

框底（z − h/2）相对**框下方的局部地面**，必须在 −1.2 m 到 +0.7 m 之间。局部地面的求法：当前帧在每个 2 m × 2 m 格内取最低点，再取框中心周围 4 m 圆内这些格子的 10% 分位。附近没有足够的地面格子时，退回用检测系的 z = 0。

| 被去掉的框 | 框底 | 局部地面 | 判定 |
|---|---|---|---|
| vehicle 0.15，(−38.8, 2.0) | +0.88 | 无（周围的点都在 1.6 m 以上，是墙），用 0 | 高出 0.88 m，挂在墙上 |
| vehicle 0.10，(−40.7, 1.2) | +1.27 | 无，用 0 | 同上 |
| vehicle 0.15，(2.0, −14.8) | −1.35 | 无（5 m 内一个点都没有），用 0 | 沉入地下 1.35 m，凭空的框 |

那辆车的情况：局部地面是 −0.48 m（那里比机器狗脚下低），框底 0.395 − 0.90 = −0.505，相对局部地面只低 2.5 cm，保留。如果还用 z = 0 作地面，它会显得沉入地下 0.5 m。本帧上报的车辆框，框底相对局部地面的中位数是 −0.21 m。

**⑨ 当前帧点数（85 → 70）**

框按 l、w、h 各放大 10% 后，统计**当前帧**（dt = 0）落在框内的点数：车要 ≥ 5，行人和骑行者要 ≥ 3。只数当前帧，是因为累积的旧帧点会让一个刚离开的物体仍显得"有点"。

| 被去掉的框 | 距离 | 点数 |
|---|---|---|
| vehicle 0.24，(−34.9, 22.7) | 41.7 m | 3 < 5 |
| vehicle 0.19，(−12.4, 38.8) | 40.7 m | 2 < 5 |
| vehicle 0.15，(26.0, 4.3) | 26.4 m | 4 < 5 |
| vehicle 0.12，(3.7, −13.2) | 13.7 m | 0 |
| vehicle 0.11，(0.4, −4.4) | 4.4 m | 0 |

远处的框只是点太少；而 4.4 m 处、0 个点的框是网络从周围的上下文"脑补"出来的。那辆车有 433 个点。

**⑩ 跨类去重（70 → 59）**

把框按分数排序；一个框的中心如果落在另一类、分数更高的框里（边界外扩 0.2 m），就去掉它。共去掉 11 个，例如：
- vehicle 0.21 在 (3.5, −1.5)：中心落在一个 cyclist 0.55（1.8 × 0.7 m）里，是同一个骑行者被车头又出了一个框；
- cyclist 0.36 在 (−29.3, −9.4)：中心落在一个 12.1 × 3.2 m 的 vehicle 0.49（公交车）里，是 bicycle 头在公交车侧面误触发；
- vehicle 0.13 和 cyclist 0.29 都落在同一个 pedestrian 0.31 里。

检测器输出 `Detections`：`boxes (59, 7)`、`velocities (59, 2)`、`scores (59,)`、`labels (59,)`（0 = vehicle，1 = pedestrian，2 = cyclist）。

## 9. 第十一步：谁用这些框

| 类别 | 检测器输出 | ≥ 上报阈值 | 上报阈值 |
|---|---|---|---|
| vehicle | 26 | 12 | 0.35 |
| pedestrian | 10 | 3 | 0.30 |
| cyclist | 23 | 3 | 0.40 |

- **上报 / 可视化**：只用分数达到阈值的 18 个（`Detections.above(DEFAULT_SCORE_THRESHOLDS)`）。
- **跟踪器**（`tracker.py:212`）全部 59 个都用：
  - 达到阈值的高分框：参加第一阶段关联，可以新建轨迹；
  - 0.1 到阈值之间的 41 个低分框：只参加第二阶段，只能延续已确认的轨迹，永远不新建轨迹。

  这样，远处或被遮挡、分数掉下来的物体，轨迹不会断。

## 10. 可调参数

| 参数 | 位置 | 现值 | 调大 / 调小的效果 |
|---|---|---|---|
| `score_threshold` | `configure_for_dog` | 0.1 | 调高：候选和低分框都变少，跟踪器第二阶段可用的框也少 |
| `post_center_limit_range` | `configure_for_dog` | 等于网格 | 设得比网格小，就只输出近处的框 |
| `min_radius`（每 task，m²） | `CIRCLE_NMS_SQ_RADIUS` | 见第 5 节 | 太小：同一物体留多个框；太大：挨得近的两个物体被合成一个 |
| `nms_post_max_size` | 配置 | 83 | 每 task 的上限，密集场景才会碰到 |
| `MERGED_NMS_RADIUS` | `centerpoint.py` | 车 2.0、骑行 0.9、行人 0（m） | 行人设 0.5 m：nuScenes 上行人 MOTA +0.07，AP −0.012（[NUSC_CLIP_EVAL.md](NUSC_CLIP_EVAL.md)） |
| `max_float` / `max_sink` | `DetectionFilterConfig` | 0.7 / 1.2 m | 框底相对局部地面允许的范围 |
| `ground_cell` / `ground_ring` / `ground_percentile` | 同上 | 2 m / 4 m / 10% | 局部地面的估计方式 |
| `min_points` | 同上 | 车 5、行人 3、骑行 3 | 调高：远处的车更容易被滤掉 |
| `cross_class_margin` | 同上 | 0.2 m | |
| `DEFAULT_SCORE_THRESHOLDS` | `filters.py` | 车 0.35、行人 0.30、骑行 0.40 | 上报阈值，也是新建轨迹的阈值 |
