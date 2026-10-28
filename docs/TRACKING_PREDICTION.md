# 跟踪与预测：原理与代码梳理

对应 [DOG_DATAFLOW.md](DOG_DATAFLOW.md) 的 4.4、4.5 节。那里只列调用链和 shape，本文展开到函数级，并解释 IMM 和 CV/CT 这两个运动模型。第 1 节讲原理，第 2、3 节讲代码，第 4 节讲已知局限。

文中的数值实验都可以复现：直接调用 `dog_perception.tracking.imm`，参数用 `DEFAULT_CLASS_PARAMS`，量测噪声按类别的 `pos_std` / `vel_std` 注入。

---

## 0. 总览

```
PerceptionPipeline.on_scan(scan)                              pipeline.py:55
├─ Preprocessor.process → frame（D 系点云，T_world_det）
├─ detector(frame.points) → Detections（D 系）
├─ dets.transform(T_world_det, "world")                       boxes.py:122    中心、朝向、速度转到 W 系
├─ MultiObjectTracker.step(dets_w, stamp) → [TrackState]      tracker.py:320  跟踪（含运动状态）
├─ _ego.append([t, x, y, vx, vy, 1])                          pipeline.py:75  机器狗自身的历史，给第二档用
└─ predictor(...) → [Prediction]                                               预测
     ├─ IMMPredictor(tracker)                                 kinematic.py:63 第一档
     └─ LearnedPredictor(tracker, ego_hist)                   learned.py:37   第二档；历史不足的轨迹回退第一档
PerceptionPipeline.to_local(out)                              pipeline.py:91  W 系 → D 系，给规划
```

关键设计：**每条轨迹只有一个 IMM 滤波器，它既是跟踪的状态估计器，也是第一档预测器。** 预测时直接把这个滤波器往前推，不需要第二个滤波器。

| 文件 | 行数 | 内容 |
|---|---|---|
| `tracking/imm.py` | 222 | CV、CT 两个运动模型和 IMM 滤波器 |
| `tracking/association.py` | 39 | 贪心匹配、匈牙利匹配 |
| `tracking/tracker.py` | 382 | 单条轨迹 `Track`、两阶段关联、生命周期、运动状态、`MultiObjectTracker` |
| `tracking/motion_eval.py` | 157 | 运动状态对照标注的评估（`tools/dog/motion_eval.py`） |
| `prediction/kinematic.py` | 80 | 输出格式 `Prediction`；第一档 `IMMPredictor` 和模态合并 |
| `prediction/features.py` | 152 | 第二档的输入构造，训练和线上共用 |
| `prediction/model.py` | 133 | 第二档网络 `TrajectoryPredictor` 和损失 |
| `prediction/learned.py` | 178 | 第二档线上封装、训练、评估（含第一档基线） |
| `prediction/dataset.py` | 203 | 跟踪日志记录、RTS 平滑、切窗生成训练样本 |
| `prediction/metrics.py` | 23 | minADE / minFDE / ADE1 / FDE1 / MR |

---

## 1. 原理：CV、CT 和 IMM

### 1.1 为什么要有运动模型

卡尔曼滤波每一帧做两件事：

1. **预测**：用运动模型把上一帧的估计推到这一帧，不确定性随之变大；
2. **更新**：用这一帧的检测修正预测，不确定性随之变小。

运动模型就是对"目标在两帧之间怎么动"的假设。真实目标会在直行、转弯、减速、停下之间切换，单一模型很难兼顾：

- 模型假设得太"硬"（比如认定是直线、过程噪声很小），转弯时预测总是落在外侧，估计滞后；
- 模型假设得太"软"（过程噪声很大），直行时又过度相信每一帧的噪声检测，估计抖动。

IMM 的做法是同时准备几个模型，让数据来判断目标此刻更像哪一个。

### 1.2 状态向量

所有模型共用同一个 5 维状态，定义在世界系（LIO 的重力对齐系）：

```
x = [px, py, vx, vy, ω]
     位置(m)   速度(m/s)  转向率(rad/s)
```

为什么用笛卡尔速度 `(vx, vy)`，而不是常见 CTRV 模型里的"速率 + 航向角"：

- **没有角度回绕。** 航向角在 ±π 处跳变，滤波和模态混合时都要特殊处理；笛卡尔速度没有这个问题。
- **静止目标是良态的。** 速率为 0 时航向角没有定义，滤波器容易发散；`(vx, vy) = (0, 0)` 则完全正常。
- **量测是线性的。** CenterPoint 速度头直接输出 `(vx, vy)`，量测矩阵 H 只是一个取前 4 维的常数矩阵，不需要线性化。
- **IMM 要求各模型的状态处在同一空间**，才能做加权混合。CV 和 CT 都用这 5 维，CV 只是不使用 ω。

### 1.3 CV 模型（Constant Velocity，匀速直线）

`CVModel`（`imm.py:32`）：

```
px' = px + vx·dt
py' = py + vy·dt
vx' = vx
vy' = vy
ω'  = 0          ← CV 认为目标不转弯，每次预测把 ω 清零
```

这是线性模型，状态转移矩阵 F 是常数（`imm.py:38`）。

**过程噪声** 用离散白噪声加速度模型（DWNA，`imm.py:17`）：假设目标每个轴上有一个随机的加速度，标准差为 `q_acc`（m/s²）。对每个轴的 (位置, 速度)：

```
Q_axis = q_acc² · | dt⁴/4   dt³/2 |
                  | dt³/2   dt²   |
```

`q_acc` 的直观含义是：目标在两帧之间能"意外地"加速、减速或侧向偏离多少。它越大，滤波器越相信检测、越不相信模型。

### 1.4 CT 模型（Coordinated Turn，协调转弯）

`CTModel`（`imm.py:51`）假设目标**速率不变、以恒定转向率 ω 转弯**，即速度向量每秒转过 ω 弧度，轨迹是一段圆弧：

```
vx' = cos(ωdt)·vx − sin(ωdt)·vy                    速度向量旋转 ωdt
vy' = sin(ωdt)·vx + cos(ωdt)·vy
px' = px + [ sin(ωdt)·vx − (1−cos(ωdt))·vy ] / ω   位置是旋转速度的积分（圆弧）
py' = py + [ (1−cos(ωdt))·vx + sin(ωdt)·vy ] / ω
ω'  = ω
```

实现上要注意三点：

- **非线性，所以用 EKF。** 方程对 ω 是非线性的，协方差传播用雅可比矩阵 `CTModel.F`（`imm.py:74`）。单测 `test_ct_jacobian_matches_numerical` 用数值差分验证了它。
- **ω → 0 时的处理。** 公式里有除以 ω。当 |ω| < 1e-4 时，`f` 和 `F` 都改用对 ω 的一阶展开。两者必须用同一套展开，否则状态预测和协方差预测互相矛盾（早期版本在这里有过不一致，已修复）。
- **过程噪声。** 位置和速度部分与 CV 相同（DWNA）；ω 做随机游走，方差每秒增加 `q_omega²`，表示转向率可以慢慢变化。

ω = 0 时 CT 退化为 CV。这一点在 1.6 节的实验里很关键。

### 1.5 IMM（Interacting Multiple Model，交互式多模型）

**思路：** 并行运行 M 个卡尔曼滤波器，每个对应一个运动模型，各自维护估计 `x_j, P_j`。同时维护模态概率 `μ_j`，表示"目标此刻按模型 j 运动"的概率。模态之间按马尔可夫链切换，转移矩阵为 Π（`Π[i,j]` 是从模态 i 切到 j 的概率）。最终输出是各滤波器按 μ 加权的合成。

每一帧分四步，对应代码如下：

**① 交互 / 混合**（`IMM.predict`，`imm.py:152-159`）

```
c_j     = Σ_i Π[i,j]·μ_i                       预测的模态概率：这一帧处在模态 j 的先验概率
W[i,j]  = Π[i,j]·μ_i / c_j                     已知这一帧处在 j，上一帧来自 i 的概率
x0_j    = Σ_i W[i,j]·x_i                       模态 j 的起点：所有滤波器估计的加权混合
P0_j    = Σ_i W[i,j]·( P_i + (x_i−x0_j)(x_i−x0_j)ᵀ )
```

这一步就是名字里的"交互"：每个滤波器不是从自己上一帧的结果出发，而是从所有滤波器的混合出发。好处是，一个长期不被看好的模态在目标切换行为时，能立刻拿到当前最好的估计作为起点，而不是从一个早已偏离的状态追赶。

**② 各模型分别预测**（`imm.py:160-162`）：每个滤波器用自己的 f / F / Q 把 `x0_j, P0_j` 推到当前时刻，并把 μ 设为先验 c。

**③ 各模型分别更新，并更新模态概率**（`IMM.update`，`imm.py:164-183`）

```
y_j = z − H·x_j                                残差
S_j = H·P_j·Hᵀ + R                             残差协方差
K_j = P_j·Hᵀ·S_j⁻¹                             卡尔曼增益
x_j ← x_j + K_j·y_j
P_j ← (I−K_j·H)·P_j·(I−K_j·H)ᵀ + K_j·R·K_jᵀ    Joseph 形式，数值上更稳
Λ_j = N(y_j; 0, S_j)                           似然：检测落在模型 j 预测处的可能性
μ_j ← c_j·Λ_j / Σ_k c_k·Λ_k                    贝叶斯更新
```

两个实现细节：似然在对数域计算，避免下溢；μ 有 1e-4 的下限，保证任何模态都不会彻底"死掉"，否则目标改变行为时它回不来。

**④ 合成输出**（`IMM.state`，`imm.py:186-190`）

```
x = Σ_j μ_j·x_j
P = Σ_j μ_j·( P_j + (x_j−x)(x_j−x)ᵀ )          第二项：模态之间的分歧也算作不确定性
```

**转移矩阵与帧间隔**（`IMM.transition`，`imm.py:135`）。参考值是 dt = 0.1 s 时保持原模态的概率为 0.95。帧间隔不是 0.1 s 时按 `0.95^(dt/0.1)` 缩放，保证切换的快慢以秒计，与帧率无关：

| dt | 保持概率 | 切换概率 |
|---|---|---|
| 0.1 s | 0.95 | 0.05 |
| 0.5 s | 0.774 | 0.226 |

0.95 意味着一个模态平均持续 1 / (1 − 0.95) = 20 帧，即 2 s。

### 1.6 实验：车辆从直行到转弯

车辆参数，5 m/s 直行 3 s，然后以 0.5 rad/s 转弯 3 s，10 Hz 量测 `[x, y, vx, vy]`，100 次随机实验的中位数（括号内为 10%–90% 分位）：

| t | 真实 ω | μ(CT) | CT 滤波器的 ω |
|---|---|---|---|
| 1.0 s | 0 | 0.41（0.39–0.47） | 0.03 |
| 3.0 s | 0 | 0.37（0.36–0.41） | 0.00 |
| 3.4 s | 0.5 | 0.42（0.36–0.51） | 0.11 |
| 3.6 s | 0.5 | 0.56（0.44–0.78） | 0.19 |
| 4.0 s | 0.5 | 0.85（0.71–0.92） | 0.37 |
| 5.0 s | 0.5 | 0.83（0.68–0.92） | 0.48 |
| 6.0 s | 0.5 | 0.82（0.64–0.93） | 0.47 |

两个现象：

- **直行时 μ 停在约 0.6 / 0.4，而不是 1 / 0。** 原因见 1.4 节末尾：ω ≈ 0 时 CT 就是 CV，两者对数据的解释一样好，似然比接近 1，μ 只由转移矩阵决定。这不是 bug，也正是为什么直行车辆通常输出两条预测轨迹（3.2 节）。
- **开始转弯后约 0.6 s，CT 占优；约 1 s 后 μ(CT) 达到 0.85；约 2 s 后 ω 估计接近真值。** 滞后来自两方面：检测有噪声，需要几帧才能确认"真的在转"；ω 不能直接观测，只能从速度方向的变化中推出来。

### 1.7 各类目标用哪些模型

`DEFAULT_CLASS_PARAMS`（`tracker.py:42`）：

| 类别 | 模型 1 | 模型 2 | 关联门限 | 保活时间 | 确认所需命中 | 位置噪声 | 速度噪声 |
|---|---|---|---|---|---|---|---|
| 车辆 | CV（q_acc 1.5） | CT（q_acc 1.5，q_ω 0.25） | 3.0 m | 1.5 s | 3 | 0.30 m | 0.8 m/s |
| 行人 | CV（q_acc 0.4） | CV（q_acc 2.0） | 1.5 m | 1.5 s | 2 | 0.15 m | 0.5 m/s |
| 骑行者 | CV（q_acc 1.5） | CT（q_acc 1.5，q_ω 0.4） | 2.5 m | 1.0 s | 2 | 0.20 m | 0.6 m/s |

三类共用：模态保持概率 0.95，ω 初始标准差 0.3 rad/s。

**行人为什么不用 CT。** 行人很少沿恒定转向率的圆弧走，更常见的是突然变向、急停、起步。所以用两个过程噪声不同的 CV：

- 低噪声 CV（0.4 m/s²）：平稳行走，估计平滑；
- 高噪声 CV（2.0 m/s²）：急停、变向，响应快。

残差变大时，IMM 自动把概率移到高噪声模型上，滤波器跟得更紧；恢复平稳后再移回来。仿真中，站立或匀速行走的行人 μ(高噪声) 平均约 0.25；随机变向的行人约 0.49。

---

## 2. 跟踪代码

### 2.1 输入：检测转到世界系

`Detections.transform(T_world_det, "world")`（`boxes.py:122`）。检测系 D 和世界系 W 都是重力对齐的，两者之间只差一个绕 z 轴的旋转加平移：

```
中心   p_w = R·p_d + t
朝向   yaw_w = yaw_d + Δyaw
速度   v_w = R[:2,:2]·v_d
```

**为什么在世界系跟踪。** 狗原地转向、步态颠簸时，在机体系里静止的行人看起来也在动；在世界系里它才是静止的。CenterPoint 输出的速度已经是对地速度（多帧拼接时做了自运动补偿），旋转到世界系后可以直接作为卡尔曼量测。单测 `test_static_pedestrian_static_in_world_while_robot_turns` 验证了这一点：机器人以 1 rad/s 原地转 5 s，静止行人的估计速度小于 0.2 m/s。

### 2.2 `MultiObjectTracker.step` 逐步

```
step(dets_w, stamp)                                                      tracker.py:320
1  所有轨迹 IMM 预测到当前时刻                       Track.predict         :324
2  按类别阈值把检测分成两组                                               :327-330
     high = score ≥ high_score[类别]      车辆 0.40 / 行人 0.30 / 骑行者 0.40
     low  = 0.1 ≤ score < high_score
3  第一阶段：high 与全部轨迹关联                    _associate            :332
4  第二阶段：low 只与第一阶段没匹配上的"已确认"轨迹关联                     :334-336
5  匹配上的轨迹：用检测更新（含运动状态）            Track.update          :340-343
     同时记录 last_matches[轨迹 id] = 检测下标（离线自动标注用）
6  没匹配上的轨迹：记一次漏检                        Track.mark_missed     :344-347
7  没匹配上的 high 检测，按分数从高到低：新建轨迹（tentative）            :349-359
     例外：和已有同类轨迹"一个的中心落在另一个框里"的不新建  _second_box_on_track
8  删除过期轨迹                                                           :361-363
9  返回 reported_tracks() 的 TrackState                                   :364
```

### 2.3 关联：`_associate`（`tracker.py:308`）

**代价**：检测中心与轨迹预测中心（IMM 合成状态）的 xy 欧氏距离。以下情况代价置为无穷，即不允许匹配：

- 类别不同；
- 距离超过该轨迹的门限。

**门限** `Track.gate()`（`:189`）随不确定性自适应：

```
σ    = sqrt( trace(P_xy) / 2 )               预测位置的平均标准差
gate = clip(3σ, 类别门限, 3 × 类别门限)
```

刚更新过的轨迹 σ 很小，门限就是类别门限（行人 1.5 m）。连续漏检时 σ 随预测增大，门限最多放大到 3 倍（行人 4.5 m），以便重新捕获遮挡后走出来的目标。

**匹配算法**（`association.py`）：

- `greedy_match`（默认）：把所有有效的（检测, 轨迹）对按代价从小到大排序，依次取用，已用过的检测或轨迹跳过。这是全局贪心，CenterPoint 官方跟踪也用这种方式。
- `hungarian_match`：`scipy.optimize.linear_sum_assignment`，最小化总代价。

目标稀疏时两者结果一致（单测 `test_matchers_agree_on_simple_case`）。目标密集、门限互相重叠时两者可能不同：贪心先锁定最近的一对，匈牙利保证总代价最小。通过 `TrackerConfig.matching` 切换。

### 2.4 为什么分两阶段关联

思路来自 ByteTrack。低分检测里大部分是误检，用它们新建轨迹会产生大量幽灵目标。但如果一个低分检测恰好落在某条已确认轨迹的预测位置附近，它多半就是这个目标，只是因为太远、被遮挡或点太少而得分低。32 线雷达下，远处和被遮挡的行人经常属于这种情况。所以：

- 第一阶段：高分检测与所有轨迹关联；
- 第二阶段：低分检测只用来延续第一阶段没匹配上的**已确认**轨迹；
- 低分检测**永远不新建轨迹**。

单测：`test_low_score_detections_do_not_spawn_tracks_and_tentative_die`、`test_lifecycle_coasting_and_low_score_recovery`。

**长货车不开第二条轨迹。** 长货车常被检成前后两个框：一个和轨迹匹配上，另一个没匹配上、分数又高，以前就会新开一条轨迹。现在没匹配上的高分框，如果和已有的同类轨迹"一个的中心落在另一个框里"，就不新建（`_second_box_on_track`）。对小车和行人基本没有影响：中心落进对方框里时，距离已经小于关联门限，本来就会匹配上；只对长于约 6 m 的车起作用。

### 2.5 单条轨迹 `Track`（`tracker.py:155`）

**新建**（`:156`）：

- 按类别参数构造 IMM，初始状态 `[x, y, vx_det, vy_det, 0]`，即新轨迹第一帧就有速度；
- 初始协方差 `diag(pos_std², pos_std², vel_std², vel_std², 0.3²)`；关闭速度量测时，速度方差取 4.0（2 m/s 标准差）；
- 高度 z、尺寸和朝向直接取自检测，`hits = 1`，状态为 tentative。

**更新**（`:194`）：

```
z = [x, y, vx, vy]                            检测中心 + CenterPoint 速度，世界系
H = [I₄ | 0]                                  取状态的前 4 维
R = diag(pos_std², pos_std², vel_std², vel_std²)
IMM.update(z, H, R)
```

关闭速度量测（`use_velocity=False`）时只用位置，H 为 2×5。

不在滤波器状态里的属性，用指数滑动平均（EMA）平滑：

| 属性 | 更新方式 |
|---|---|
| 高度 z | `z += 0.3·(z_det − z)` |
| 尺寸 l, w, h | `size += 0.2·(size_det − size)` |
| 分数 | `score = 0.7·score + 0.3·score_det` |
| 朝向 | 见下 |

**朝向**（`_update_yaw`，`:255`）：

- **行人**：框接近正方形，检测出的朝向基本没有意义。只在速度超过 0.4 m/s 时把朝向设为速度方向，否则保持不变。
- **车辆、骑行者**：
  1. 检测朝向与当前朝向相差超过 90° 时先翻转 180°。检测框的头尾本来就有歧义，这一步把它消掉；
  2. 速度超过 1 m/s 时，再保证朝向与运动方向一致，相反就再翻一次；
  3. 最后按 0.4 的系数做角度 EMA。

**运动状态**（`_update_motion`，`:233`）：狗遇到运动的车要停，遇到停着的车不停，所以每条轨迹单独给出 `motion`：

| 状态 | 什么时候进入 |
|---|---|
| `moving` | 最近 5 次检测里有 3 次"快"：位移速度和速度头都 > 0.7 m/s（行人 0.5） |
| `stopped` | 运动之后，1 s 内没有一次"快"，并且位移速度 < 0.3 m/s（行人 0.2）；刚停下的车可能马上再走 |
| `static` | 一开始就是；或者 `stopped` 持续 10 s |

- 位移速度（`displacement_speed`，`:224`）：这条轨迹最近 1 s 检测框中心的直线拟合斜率，不足 0.3 s 时不算。速度头是这一帧检测器给的速度。
- 两个都要过阈值：停着的车框会随视角漂（位移速度有零点几 m/s），速度头只看一帧、位姿一抖就跳。
- 不用 IMM 速度：它是用第一帧的速度头初始化的，开头几帧和速度头是同一个信号。
- 依据、仿真延迟和 0921 上的评估见 [DOG_FINETUNE.md](DOG_FINETUNE.md) 第 14 节；阈值用 `tools/dog/motion_eval.py` 调。

**漏检**（`mark_missed`，`:272`）：分数乘以 0.9，并把预测状态记入历史，标记为未观测。

**历史**（`_record`，`:276`）：每帧追加一行 `[t, x, y, vx, vy, observed]`，取 IMM 的合成状态，保留最近 3 s。第二档预测器用的就是它。

### 2.6 生命周期

```
     未匹配的 high 检测
            │
            ▼
       tentative ──── 累计命中 ≥ 2 ────► confirmed ◄─────────┐
            │                               │                 │ 重新匹配上
            │ 距上次匹配 > 0.15 s            │ 没匹配上         │
            ▼                               ▼                 │
          删除                           coasting ─────────────┘
                                            │ 距上次匹配 > max_age
                                            ▼  （行人、车辆 1.5 s，骑行者 1.0 s）
                                          删除
```

- tentative 的 0.15 s 意味着 10 Hz 下可以容忍 1 帧漏检（0.1 s），连续 2 帧（0.2 s）就删除。
- tentative 轨迹只能被高分检测延续（第二阶段只看已确认轨迹）。
- coasting 不是单独存储的状态，而是"距上次匹配的时间 > 0"。期间位置完全靠 IMM 预测，协方差随时间增大。

### 2.7 哪些轨迹会上报：`reported_tracks`（`tracker.py:367`）

同时满足以下条件才上报（预测也只对这些轨迹做）：

1. 已确认；
2. 正在 coasting 的，`output_coasting=True` 时也上报（默认开）；
3. 平滑后的分数 ≥ `output_min_score`（车辆 0.35、行人 0.25、骑行者 0.3）。

这条规则和分数衰减叠加，产生两个容易忽略的效果：

**漏检时，轨迹在被删除之前就会停止上报，分数越低停得越早。** 每漏检一帧分数乘 0.9。以行人为例（门限 0.25，1.5 s 后删除）：

| 漏检前的分数 | 还能上报的漏检帧数 |
|---|---|
| 0.9 | 12 帧（1.2 s） |
| 0.6 | 8 帧 |
| 0.4 | 4 帧 |
| 0.3 | 1 帧 |

**只靠低分检测维持的轨迹会存活，但不上报。** 第二阶段的低分检测会让分数的 EMA 逐渐收敛到检测分数。比如一直被 0.15 分的检测延续，分数最终会降到 0.15 附近，低于上报门限。这类轨迹不输出，但仍在参与关联；一旦高分检测回来，它会保持原来的 ID。这正是第 7 节"检测框过多"修复中加上的机制（[DOG_PIPELINE.md](DOG_PIPELINE.md) 第 7 节）。

### 2.8 输出 `TrackState`（`tracker.py:126`）

| 字段 | shape | 含义 |
|---|---|---|
| `track_id` | int | 轨迹 ID，从 1 递增，不复用 |
| `label` / `name` | int / str | 0 vehicle、1 pedestrian、2 cyclist |
| `stamp` | float | 当前帧时间 |
| `position` | (3,) | 世界系 [x, y, z]；xy 来自 IMM，z 来自 EMA |
| `velocity` | (2,) | 世界系 [vx, vy] |
| `yaw` | float | 世界系朝向 |
| `size` | (3,) | l, w, h |
| `score` | float | 平滑后的分数 |
| `covariance` | (4, 4) | [px, py, vx, vy] 的协方差，IMM 合成后的结果 |
| `mode_probs` | (2,) | 当前模态概率 μ |
| `age` / `hits` | float / int | 存活时间、累计命中次数 |
| `coasting` | bool | 这一帧没匹配上检测 |
| `history` | (K, 6) | `[t, x, y, vx, vy, observed]`，最近 3 s |
| `motion` | str | `static` / `moving` / `stopped`（2.5 节）；`moving` 属性等于 `motion == "moving"` |
| `time_since_moving` | float | 距上次处于 `moving` 的秒数 |
| `speed_displacement` / `speed_head` | float | 最近一次运动判断用的位移速度、速度头读数（m/s） |

### 2.9 参数一览：`TrackerConfig`（`tracker.py:75`）

2026-10-27 起，车辆的 `high_score` 0.35 → 0.40、`output_min_score` 0.3 → 0.35、`low_score` 0.1 → 0.2，
车辆的 `max_age` 1.0 → 1.5 s、`min_hits` 2 → 3（1.7 节的表）。这些值是在 0921 的验证场景上为上线的 nuScenes Pillar 调出来的，
见 [PIPELINE_TUNING.md](PIPELINE_TUNING.md) 和 SERVER_TASKS.md 的 T19–T20。
离线自动标注（`autolabel.label_tracker_config`）仍用调之前的值，0921 的标签就是用它们做的。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `class_params` | 1.7 节的表 | 按类别的模型、门限、保活时间、噪声 |
| `classes` | `("vehicle",)` | 只跟踪这些类别（检测器仍输出行人、骑行者） |
| `high_score` | 车辆 0.40 / 行人 0.30 / 骑行者 0.40 | 能新建轨迹、参与第一阶段的分数 |
| `low_score` | 0.2 | 参与第二阶段的最低分数 |
| `matching` | `"greedy"` | 或 `"hungarian"` |
| `use_velocity` | True | 是否把 CenterPoint 速度作为量测 |
| `tentative_max_age` | 0.15 s | tentative 轨迹允许的漏检时间 |
| `output_coasting` | True | coasting 的轨迹是否上报 |
| `output_min_score` | 车辆 0.35 / 行人 0.25 / 骑行者 0.3 | 上报所需的平滑分数 |
| `history` | 3.0 s | 历史保留长度 |
| `size_alpha` / `z_alpha` / `yaw_alpha` | 0.2 / 0.3 / 0.4 | EMA 系数 |
| `heading_from_velocity` | 1.0 m/s | 车辆和骑行者速度超过它，朝向对齐运动方向 |
| `motion_on_speed` / `motion_off_speed` | 车辆 0.7 / 0.3，行人 0.5 / 0.2，骑行者 0.7 / 0.3 m/s | 运动状态的"快"、"慢"阈值 |
| `motion_window` / `motion_on_frames` | 5 / 3 | 最近几次检测里几次"快"进入 `moving` |
| `motion_traj_window` | 1.0 s | 位移速度拟合用的时间窗 |
| `motion_off_time` / `stop_memory` | 1.0 s / 10 s | `moving` → `stopped`、`stopped` → `static` 的时间 |

### 2.10 调参：现象 → 先调什么

| 现象 | 先调 |
|---|---|
| 同一个目标频繁换 ID | 增大该类 `gate`；检查是否有类别跳变（行人 ↔ 骑行者，见 4 节）；目标密集时换 `hungarian` |
| 误检被确认成轨迹 | 提高 `high_score`；`min_hits` 从 2 改为 3 |
| 遮挡后轨迹断掉 | 增大 `max_age`；降低 `low_score`，让第二阶段多捡一些 |
| 速度抖动 | 增大 `vel_std`（少信检测速度），或减小 `q_acc` |
| 转弯时位置滞后 | 增大 CT 的 `q_omega` 或 `q_acc` |
| 遮挡时轨迹还活着却不上报 | 降低 `output_min_score`（注意 2.7 节的分数衰减） |
| 停着的车被报成运动 / 运动的车报得太晚 | 提高 / 降低 `motion_on_speed`、`motion_on_frames`，先用 `motion_eval.py` 在验证场景上扫 |

---

## 3. 预测代码

### 3.1 输出格式：`Prediction`（`kinematic.py:17`）

| 字段 | shape | 含义 |
|---|---|---|
| `track_id`、`label`、`stamp` | | 对应的轨迹 |
| `times` | (T,) | 未来时刻，默认 0.5, 1.0, …, 3.0 s |
| `modes` | (K, T, 2) | K 条候选轨迹的 xy |
| `probs` | (K,) | 每条轨迹的概率，和为 1 |
| `covs` | (K, T, 2, 2) | 每条轨迹每个时刻的位置协方差 |
| `source` | str | `"imm"` 或 `"learned"` |
| `frame` | str | `"world"`，经 `to_local` 后为 `"det"` |

`best` 返回概率最大的那条轨迹；`transform` 把整个预测转到另一个坐标系。

### 3.2 第一档：IMM 外推（`kinematic.py`）

```
IMMPredictor.__call__(tracker)                                       kinematic.py:74
for t in tracker.reported_tracks():
    predict_imm(t.imm, ...)                                          :67
    ├─ IMM.rollout(3.0, 0.5, mixed=False)                            imm.py:192
    │     对每个模态 j：从该滤波器自己的 x_j, P_j 出发，
    │     只用自己的模型、不加量测，连续预测 6 步（每步 0.5 s）
    │     → mode_means (M,6,5)、mode_covs (M,6,5,5)、mode_probs = 当前 μ
    └─ merge_modes(xy 部分, merge_dist=0.25 m)                       :40
          → Prediction(modes (K,6,2), probs (K,), covs (K,6,2,2))
```

**为什么每个模态单独外推，而不是用完整的 IMM 往前推。** 完整的 IMM 每一步都会把各模态混合，得到的是"直行"和"转弯"之间的一条平均轨迹，现实中没有车会这么走。每个模态单独外推，得到的是两个明确的假设（"保持直行"和"保持当前转向率"），再附上各自的概率，这才是多模态预测。`rollout(mixed=True)` 仍然可以算完整 IMM 的结果，但预测器不用它，所以传了 `mixed=False`：这样每条轨迹的耗时从约 2.1 ms 降到约 0.85 ms。

**模态概率在预测时间内保持不变**，直接取当前的 μ，不随转移矩阵演化。如果按 Π 演化，3 s 后两个模态的概率都会被拉向 0.5 / 0.5，失去区分度。

**模态合并**（`merge_modes`）：按概率从大到小排序；如果某个模态在**所有时刻**都与某组首个模态相距不到 0.25 m，就并入该组。每组按矩匹配合并：

```
w_j   = μ_j / Σ_组 μ          组内归一化
mean  = Σ w_j·mean_j
cov   = Σ w_j·( cov_j + (mean_j − mean)(mean_j − mean)ᵀ )
prob  = Σ_组 μ_j
```

**实际输出几条轨迹**（数值实验）：

- **车辆、骑行者直行时**：CT 滤波器的 ω 很小但不为零，3 s 内 0.02 rad/s 的差别就足以让终点偏开 0.25 m 以上。所以通常输出两条，"直行"约占 0.6、"略弯"约占 0.4（原因见 1.6 节）。
- **车辆、骑行者转弯时**：CT 的概率在 0.8 以上，"继续转弯"成为主模态。
- **行人**：低噪声 CV 和高噪声 CV 的卡尔曼增益不同，速度估计总有差别；差别超过约 0.08 m/s，3 s 终点就相距 0.25 m 以上。实验中只有约 1/5 的帧会合并成一条，多数时候输出两条：一条平滑（低噪声 CV），一条紧跟最近的观测（高噪声 CV）。主模态的 3 s 位置标准差，站立或匀速行走时约 1.2 m，随机变向时约 2.5 m。

旧文档写的是"行人两个 CV 会合成一条"，与实验不符，已一并更正（`kinematic.py` 文档字符串、DOG_DATAFLOW 4.5 节、DOG_PIPELINE 1.5 节）。如果规划侧希望行人只有一条轨迹，可以把合并判据改成按协方差归一化的距离（马氏距离），或者增大 `merge_dist`。

**协方差** 取每个模态 P 的 xy 部分。DWNA 噪声下，位置方差大致按 t³ 增长，再加上初始速度不确定性按 t² 传播的部分，所以 3 s 处的不确定性明显大于 0.5 s 处。

### 3.3 第二档：学习模型

#### 3.3.1 输入构造（`features.py`）

`build_inputs`（`:87`）为每个"焦点目标"构造一组以它自己为中心的输入。训练集生成和线上推理调用的是同一个函数，所以网络在线上看到的输入分布与训练时完全一致。

1. **历史重采样**（`resample_history`，`:38`）：把轨迹历史 `(K,6) [t, x, y, vx, vy, observed]` 插值到固定时间网格 `t_now − 0.1·(19, 18, …, 0)`，得到 `(20, 6) [x, y, vx, vy, observed, valid]`。轨迹出生之前的时刻 `valid = 0`，数值清零。
2. **目标坐标系**（`heading_of`、`to_agent_frame`，`:55`、`:60`）：原点放在目标当前位置；+x 指向目标的运动方向（速度超过 0.5 m/s 时），否则用轨迹朝向。位置和速度都旋转到这个坐标系。这样网络不需要学习平移和旋转不变性。
3. **邻居**：20 m 内的其他轨迹按距离排序，最多取 16 个，各自的历史也转到焦点目标的坐标系。机器狗在焦点目标 20 m 以内时，**固定放在第一个邻居槽位**，`is_robot = 1`，因为行人会避让机器人，这是狗场景里最强的交互信号。机器狗的历史来自 `PerceptionPipeline._ego`，速度由相邻两帧位置差分得到。
4. **障碍物栅格**（可选，`use_raster`）：以目标为中心 32 m × 32 m、0.5 m 分辨率的占据图 `(1, 64, 64)`。默认关闭，线上流水线目前也没有传入障碍物点。

| 张量 | shape | 说明 |
|---|---|---|
| `agent_hist` | (B, 20, 6) | 焦点目标过去 2 s |
| `agent_class` | (B, 3) | one-hot |
| `nbr_hist` | (B, 16, 20, 6) | 邻居历史，含机器狗 |
| `nbr_static` | (B, 16, 4) | 邻居类别 one-hot + `is_robot` |
| `nbr_mask` | (B, 16) | 1 表示槽位有效 |
| `raster`（可选） | (B, 1, 64, 64) | 障碍物占据 |

#### 3.3.2 网络（`model.py`）

```
agent_hist (B,20,6) ─ TemporalEncoder ───────────────────────► a (B,64)
nbr_hist (B·16,20,6) ─ TemporalEncoder（另一套权重）─► (B,16,64) + Linear(nbr_static) ─► n
[null, n_1 … n_16] ─ MultiheadAttention(query=a，4 头，屏蔽空槽) ─► s (B,64)   社交上下文
agent_class ─ Linear ────────────────────────────────────────► c (B,64)
[raster ─ 3 层 CNN ──────────────────────────────────────────► m (B,64)]       可选
concat[a, s, c (, m)] ─ MLP ─► h (B,128)
6 个可学习的模态查询 q_k (64)；z_k = [h, q_k] (B,6,192)
   ├─ traj_head MLP ─► (B,6,6,4) = 每个模态、每个时刻的 (μx, μy, log bx, log by)
   └─ cls_head  MLP ─► logits (B,6)
```

- **TemporalEncoder**（`:35`）：先用两层 MLP 把 6 维特征升到 64 维，再接 GRU（取最后时刻的隐状态）或 2 层 4 头的小 Transformer（可学习位置编码，屏蔽无效时刻，取最后一个 token）。
- **null token**：注意力里额外放一个可学习的"空"键值。没有邻居、或者邻居都无关紧要时，注意力可以落在它上面，而不是被迫平均到无关的邻居上。
- **模态查询**：6 个可学习向量，与上下文拼接后共享同一个输出头。不同的查询在训练中会分化成不同的机动假设（直行、左转、右转、减速……）。
- **参数量**：GRU 版约 15.8 万，Transformer 版约 24.5 万。`log_scale` 截断在 [−4, 3]。

#### 3.3.3 损失（`predictor_loss`，`model.py:107`）

```
best = argmin_k ADE_k                          与真值平均距离最小的模态（只算有效时刻）
reg  = Σ_t Σ_xy [ |y − μ_best| / b_best + log b_best + log 2 ]     Laplace 负对数似然，只算 best
cls  = CrossEntropy(logits, best)              让分类头学会"哪个模态会赢"
loss = reg + 0.5·cls
```

- **为什么只训练 best 模态（Winner-Takes-All）。** 如果对所有模态都算回归损失，它们会一起收敛到平均轨迹，多模态就塌缩了。WTA 让每个模态只在自己"赢"的样本上学习，从而各自专注于一种机动。
- **为什么用 Laplace 而不是高斯。** Laplace 对应 L1 误差，对少数离群样本（跟踪抖动、急转）不敏感。输出的尺度 b 换算成方差是 2b²。

#### 3.3.4 线上封装：`LearnedPredictor`（`learned.py:24`）

```
__call__(tracker, ego_hist)                                  learned.py:37
├─ tracks = tracker.reported_tracks()
├─ focal  = 历史 ≥ 0.5 s 的轨迹
├─ build_inputs(...) → 网络前向
├─ 每个焦点目标：
│    modes = agent_to_world(traj)                目标坐标系 → 世界系
│    probs = softmax(logits)
│    covs  = R · diag(2b²) · Rᵀ                  Laplace 尺度 → 方差，再旋转到世界系
└─ 历史不足 0.5 s 的轨迹 → 回退第一档 IMMPredictor.predict_imm
```

回退保证每条上报的轨迹都有预测，第二档可以直接替换第一档，不会出现空缺。

#### 3.3.5 训练数据（`dataset.py`）

**记录**：`TrackLogger`（`:31`）每帧记录三样东西，由 `run_sequence.py` 保存为 `track_log.npz`：

- 每条上报轨迹一行 `rows: [t, id, label, x, y, vx, vy, yaw, observed, det_x, det_y, det_yaw, det_vx, det_vy, det_score]`：前 9 列是跟踪器的估计，后 6 列是这一帧匹配到的检测（`TrackState.measurement`，coasting 时为 NaN；2026-10-28 之前的日志只有前 9 列）；
- 机器狗位姿 `ego: [t, x, y, yaw]`；
- 第一档线上实际输出的预测（概率最高的 2 个模态），用于在同一批样本上公平对比两档。

**切窗**：`build_samples`（`:173`）：

1. **真值**：每条轨迹只取实际观测到的帧，用 RTS 平滑器（`rts_smooth`，CV 模型，前向卡尔曼加后向平滑）得到非因果的平滑轨迹，作为真值来源（`track_targets`）。平滑什么由 `SampleConfig.targets` 定：`measured`（默认）是匹配到的检测框中心本身，带检测的抖动，但没有滞后；`filtered` 是跟踪器在这些帧更新后的位置，已经滤波过一次，车减速时会冲到前面（2026-10-28 之前的日志只能用这个）。T23 在 0921 上量过：运动车 3 s 处 measured 离标签 0.2 m，filtered 0.5–0.7 m。两种都是**伪真值**，不是车的真实轨迹：ID 切换会把别的车的未来当成目标，假轨迹会混进窗口，未来被遮挡的窗口被丢掉。它们离审过的标签有多远，`pipeline_eval.py` 的 "Training targets" 表会量（3.3.7 节）；
2. 每 2 帧取一个当前时刻 `t_now`；
3. 对 `t_now` 时存在、且已有 ≥ 0.5 s 历史的轨迹：
   - 未来 3 s 必须都在轨迹寿命内；
   - 未来时段内至少 80% 的帧真的观测到了目标（不是 coasting）；
   - 满足条件就从平滑轨迹上插值出 0.5, 1.0, …, 3.0 s 的真值（`window_target`）；
4. 输入用 `build_inputs` 构造，历史取跟踪器的**因果**估计，与线上完全一致；
5. 真值转到目标坐标系保存；同时保存同一窗口的第一档线上预测。

**输入因果、真值非因果**：输入必须是线上能拿到的东西，即跟踪器实时输出的、带滞后和噪声的估计；真值则可以利用未来信息去掉跟踪噪声，让网络学"目标真实怎么走"，而不是"跟踪器会输出什么"。

**训练集和验证集要按录制划分**，不能按窗口划分。相邻窗口高度重叠，按窗口划分会把验证集的信息泄露进训练集。

#### 3.3.6 训练、评估与指标

`train_predictor`（`learned.py:74`）：AdamW（学习率 2e-3，权重衰减 1e-4）、OneCycle 学习率、batch 256、30 个 epoch、梯度裁剪 5。命令行入口：

```bash
python tools/dog/train_predictor.py --logs "work_dirs/rec_*/pillar/track_log.npz" \
    --val-logs work_dirs/rec_009/pillar/track_log.npz --encoder gru --out work_dirs/pred/gru.pt
python tools/dog/train_predictor.py --sim 10 --encoder transformer      # 没有实车数据时用仿真日志
```

`evaluate_kinematic`（`learned.py:125`）在同一批样本上给出三个基线：

- `imm`：日志里记录的第一档**线上**输出，这是公平的对比对象；
- `imm_refit`：在样本历史上重新跑一遍 IMM。历史本身已经是滤波后的结果，再滤一遍会产生额外滞后，只在日志里没有线上预测时作为替代；
- `cv`：用最后一帧速度匀速外推。

指标（`metrics.py`）：

| 指标 | 含义 |
|---|---|
| minADE / minFDE | K 个模态里最好的那个的平均位移误差 / 终点误差。衡量"真实轨迹是否被某个模态覆盖" |
| ADE1 / FDE1 | 概率最高的那一个模态的误差。衡量"最可能的那一条准不准" |
| MR@2m | 所有模态的终点误差都超过 2 m 的样本比例 |

仿真结果见 [DOG_PIPELINE.md](DOG_PIPELINE.md) 第 4 节：第二档在 minFDE 和 MR 上大幅领先；分布外场景下 FDE1 不如 IMM。所以规划侧应使用全部模态及其概率，不要只取 `best`。

#### 3.3.7 在狗的数据上训练（2026-10-28 起，服务器任务 T23）

1. **训练数据：跟踪日志，不用标注。**
   - `tools/dog/pipeline_eval.py --track-logs DIR` 从存好的检测重跑跟踪（定版参数，CPU），每个场景写一个 `DIR/<场景>.npz`；
   - 日志里同时记下线上 IMM 的预测，评估时和第二档在同一批窗口上比；
   - 生成日志时预测要用 IMM（默认），不要加 `eval.learned_model`。
2. **样本不均衡。** 窗口大多来自停着的车。`train_predictor.py` 的两个选项：
   - `--static-keep 0.3`：3 s 内移动不到 `--moving-disp`（1.5 m）的窗口只留 30%，运动的全留；
   - `--mirror`：每个训练窗口再加一份左右镜像。
   - 验证窗口不抽、不镜像。结果按全部、运动、静止三组分别打印：静止的车被预测成要开走，会让狗误停。
3. **训练目标选哪种。** `train_predictor.py --targets measured|filtered`（3.3.5 节，默认 measured）。
   - 评估工具每次都把每个预测样本的训练目标（按同样规则从本次跟踪日志切出）和标注的未来比，报告里是 "Training targets" 表，`report.json` 里是 `target_<filtered|measured>_err3_moving`；
   - 哪种离标签近就用哪种。`train_predictor.py` 打印的验证表用的也是这种伪真值，能不能替换 IMM 只看下一步对标签的比较。
4. **和标签比（端到端）。**
   - `pipeline_eval.py --set "eval.learned_model=,<model.pt>"`：第 1 行 IMM，第 2 行第二档，同一批轨迹配对重抽；
   - 指标名是 `pred_*`，指线上实际在用的预测器；`CV_*` 是匀速；
   - 运动车的误差还拆成沿标注路径的纵向（`pred_along3_moving`，`pred_along_bias3_moving` 为带符号的均值，正 = 预测快了）和离开路径的横向（`pred_across3_moving`，`prediction/metrics.py` 的 `along_across`）。横向占比小，说明误差在速度和起停上，道路地图 / 占据栅格帮不上。
5. **上线**：ROS 2 节点参数 `predictor_model:=<model.pt>`。历史不足 0.5 s 的轨迹自动回退 IMM（3.3.4 节）。

**0921 上的第一版（T23，2026-10-28）：没有替换 IMM。** 详见 SERVER_TASKS.md T23 复核：
- 概率最高的那条路线偏慢：端到端 3 s 的纵向 bias，GRU −1.8 m、Transformer −2.5 m，FDE 3 s 比 IMM 高（3.03、3.52 对 2.80）。训练场景的车多在减速，验证场景不是，41 辆运动车学到的先验换了场景就不对。
- 模式覆盖得好：minFDE 1.25 对 2.79，MR 0.19 对 0.52。停着的车不会被往外推。
- 耗时：单线程 10 条轨迹约 16 ms、30 条约 40 ms（GRU），超过 10 ms 的门槛。上线前要只对在动的轨迹跑网络，邻居历史每帧只编码一次，Orin 上用 TensorRT。
- IMM 的误差主要在纵向（验证 2.72 m，横向 0.58 m），也就是速度估计；道路地图暂不做（[DRIVABLE_AREA_PLAN.md](DRIVABLE_AREA_PLAN.md)）。

### 3.4 给下游

- `PerceptionPipeline.to_local(out)`（`pipeline.py:91`）把轨迹和预测从世界系转到当前帧的检测系 D（重力对齐、航向跟随机身、原点在地面），这是 2D 局部规划器习惯的坐标系。
- ROS 2 节点把每个模态画成一条 6 点折线（`~/predictions`）。给规划的正式消息还没定义，见 DOG_DATAFLOW 4.6 节。

---

## 4. 已知局限与可改进点

| 局限 | 影响 | 可能的改法 |
|---|---|---|
| 关联只看中心距离，且要求类别相同 | 行人和骑行者之间类别跳变时会断开、新建轨迹，ID 改变 | 允许相近类别跨类关联并加罚代价；轨迹类别改为按历史投票 |
| z、尺寸、朝向用 EMA，不在滤波器状态里 | 没有 z 方向速度；朝向与速度方向的一致性靠规则维护 | 一般够用；上下坡或楼梯场景可以把 z 加入状态 |
| 第一档的模态概率在 3 s 内固定不变 | 直行车辆的"略弯"分支始终约占 0.4 | 规划侧按概率加权使用；或对 ω 很小的 CT 模态直接并入 CV |
| 行人两个模态大多不合并（3.2 节） | 行人通常输出 2 条很接近的轨迹 | 合并判据改为马氏距离 |
| coasting 轨迹的预测从纯预测状态出发 | 遮挡越久，预测越不可靠 | 规划侧结合 `coasting` 标志和协方差使用 |
| 第二档在 0921 上没过门槛，障碍物栅格未接入线上 | 线上仍是 IMM；第二档概率最高的路线偏慢，单线程 16–40 ms | 厂区新数据来了再训（3.3.7 节）；栅格 / 可行驶区域的方案见 [DRIVABLE_AREA_PLAN.md](DRIVABLE_AREA_PLAN.md)，暂不做 |
| 运动状态只看最近 1 s 的位移和速度头 | 位姿跳变时，几个静止物体会一起被报成运动（运动样例 bag 1.8–2.3 s）；线上用 INS 位姿，跳变比离线修正的多 | 线上位姿质量差时（INS 跳变）暂停进入 `moving`；用线上位姿在验证场景上再评一次 |
| 全部为 Python 实现 | x86 上跟踪约 6 ms；第一档预测在本次修改后每条轨迹约 0.85 ms | 部署到 Orin 时移植 C++（DOG_PIPELINE 8.3 节） |
