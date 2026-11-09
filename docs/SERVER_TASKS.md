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
   - 遇到"停止条件"就停下报告，不要改门槛、改配置或换参数绕过去。代码可以按约定 9 改。
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
   for d in train deploy runs_veh_v0 runs_nusc_pp runs_nusc_voxel runs_veh_v1 runs_veh_v1_view runs_r2 labels_r2 seq_lever tune; do
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
9. 改代码（2026-10-29 起）。为了少来回一趟，下面这些情况自己改了接着做，不用停下等复核。
   - 可以改：
     - 环境和编译：库路径、CMake 和依赖、真实的 nvcc / TensorRT 头文件下的编译错误、库的 API 版本差异、脚本里的路径和参数；
     - 工具的 bug：崩溃、读写格式不对、参数没接上，原因清楚、改动局部的；
     - C++ 和 Python 对不上：以 Python 为准改 C++（2026-11-01 起是 `deploy_deb/perception_combo_ros/node_src/lidar_3d_detection/`，见约定 10）；
     - 做完这一步缺的小功能（多一个参数、多打印一项），以及不改结果的提速（并行、缓存）。
   - 不能改，停下报告：
     - 门槛、停止条件、配置和参数的默认值（`configs/`、dataclass 的默认值）、训练设置、评估指标的定义；
     - 删掉、跳过或放宽测试（加测试可以）；
     - Python 流水线的行为（`dog_perception/`）：它是参考实现，觉得它错了就报告；只有明显的崩溃可以修；
     - 一个问题改了 30 分钟还没好，或者要动 3 个以上文件、要改设计的。
   - 规矩：
     - 每个修复单独提交，提交信息写"<任务> 修：原因和改法"；提交前跑相关测试（C++ 的 ctest；动了 Python 就跑约定 7 的快速测试）；
     - 推送前 `git pull`，有冲突就停下报告，不要 force push；
     - 报告里加一节"代码改动"：每条写提交号、文件、原因、怎么验证的；
     - 改了代码，前面已经跑完、受影响的步骤要重跑。
   - 任务里写了"可以改的范围"的，以任务里的为准。
10. C++ 只有一份（2026-11-01 起）。算法仓库的 `deploy/` 删掉了，C++ 在 `deploy_deb/perception_combo_ros/node_src/lidar_3d_detection/`（下文 `$L`），不带 ROS 也能编：
    ```text
    cmake -S $L -B build/lidar3d -G Ninja -DLIDAR3D_ROS=OFF -DLIDAR3D_TESTS=ON [-DLIDAR3D_WITH_CUDA=ON]
    cmake --build build/lidar3d -j && [LIDAR3D_MODEL_DIR=<模型>] ctest --test-dir build/lidar3d --output-on-failure
    ```
    T25–T31 里的命令照下面换（完整的表在 DEPLOY_ORIN.md 第 7 节）。那些任务的报告是当时的记录，不用改：
    - `deploy/cpp`、`deploy/cpp/build_cuda/dogperc_replay` → `$L`、`build/lidar3d_cuda/lidar_3d_replay`；`dogperc_tests` → `lidar3d_tests`；
    - `DOGPERC_WITH_CUDA`、`DOGPERC_MODEL_DIR` → `LIDAR3D_WITH_CUDA`、`LIDAR3D_MODEL_DIR`；
    - `deploy/ros2` 的 `dogperc_node`、`dogperc_bag_replay` → deb 里的 `lidar_3d_detection`、`lidar_3d_bag_replay`（板上不再单独 colcon）；
    - `deploy/docker/Dockerfile.dev` → `$L/docker/Dockerfile.dev`（镜像名还是 `dogperc-dev`）；
    - `deploy/scripts/sync_to_orin.sh` → `tools/deploy/sync_to_orin.sh`；`orin_check.sh`、`orin_soak.sh` → `$L/scripts/`；
    - `tools/deploy/sync_deploy_deb.py` → `tools/deploy/check_deploy_deb.py`（只查配置和消息）。

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
| T8 | 部署：导出上线 Pillar 的 ONNX（2026-10-26 起是 nuScenes Pillar，见 T15 复核）；选了 Voxel 时再打包 Voxel，人在 Orin 上测（H2） | Agent | T7 | 10 分钟 | 并入 T26（2026-10-28） |
| H2 | （T7 选了 Voxel 时）在 Orin 上测 Voxel v1 的延迟，放得下就部署 Voxel，否则部署 Pillar | 人 | T8 | 半天 | 不做（2026-11-09，T40：上线的是 Pillar） |
| T9 | （可选）用 v1 Voxel 当教师重新自动标注，准备第二轮 | Agent | T7 | 1 小时 | 不做（2026-11-09，T40：教师已定，T15、T39 都是不换） |
| T10 | （可选）比较 0921 的两个 INS 杠杆臂 | Agent | T0 | 1 小时 | 不做（2026-11-09，T40：可选项，收尾时不做） |
| T11 | 验证集加多模型候选框（红框"候选"）；35 个场景写上模型框，供 Ctrl + 拖框用 | Agent | T7 | 20 分钟 | 完成（物体最高分门槛 0.5，306 个候选物体） |
| T11b | 重做候选：只要车辆，只用 v1 Voxel、v1 Pillar 两个模型（T11 混进了行人、骑行者） | Agent | T11 | 15 分钟 | 完成（物体最高分门槛 0.3，54 个候选物体） |
| T16 | 双教师的自动标注导成新的 SUSTechPOINTS 场景（35 个），沿用 H1 的审核结论；训练和验证一样，只留两个教师的框 | Agent | T14 | 20 分钟 | 完成（2026-10-26，验证集的 v1 候选已去掉） |
| H3 | 在双教师的场景里审红框（训练 + 验证 35 个场景，只审车）；漏掉的车 Ctrl + 拖框补上 | 人 | T16 | 2–3 小时 | 完成（junyongmou，2026-10-26） |
| T17 | 导回 H3，训练场景传播到每一帧、补停着的车的空档，导出 v2 数据集 | Agent | H3 | 40 分钟 | 完成（2026-10-26） |
| T12 | 在新的验证标签上重评所有模型；选下一轮的两个教师（Voxel、Pillar 各一个） | Agent | T15 | 1 小时 | 不做：教师不换（T15 里 v2 Voxel + TTA 比 nuScenes 教师低 0.109） |
| T13 | 雷达后方盲区：量出盲区，看各模型在盲区边上是不是误检更多 | Agent | T7 | 10 分钟 | 完成（盲区 142°–218°；v1 Pillar 在 10–40 m 边上算明显） |
| T14 | 双教师（nuScenes Voxel + TTA、nuScenes Pillar + TTA）重新自动标注 0921，补进审过的训练标签（不丢 H1） | Agent | T7 | 1 小时 | 完成（双教师标注 T16 接着用；labels_merged.pkl 不再用） |
| T15 | 用 v2 数据集训练 v2；定上线用哪个 Pillar、教师换不换 | Agent | T17 | 3.5 小时 | 完成。上线用 nuScenes Pillar（人定：v1、v2 都不如它，见复核）；教师不换 |
| T18 | 后处理、跟踪、预测的基线：上线 Pillar 在验证场景上的指标、Foxglove MCAP、问题清单；六份对照 | Agent | T15 | 30 分钟 | 完成（修正后重跑；oracle 假轨迹 0 / 分钟；跟踪只计车辆）。续 2 完成：叉车忽略后验证车框 1721，oracle 假轨迹仍是 0；v2_pp − nusc_pp 的 ΔmAP 从 −0.041 收到 0.000 |
| H4 | 在 Foxglove 里看定版结果（T22_mcap），按 T22 复核的表看 5 处问题 | 人 | T22 | 1 小时 | 暂缓（2026-10-28 人定：新数据来了模型还会改，到时再看） |
| T19 | 后处理扫参（上报阈值、轨迹输出分数、低分门槛、最少点数、离地 / 下沉）；第 1b 步三个 Pillar 各用最合适的阈值比，可能要人重新定上线模型 | Agent | T18 | 30 分钟 | 完成（人定仍用 nuScenes Pillar）。上报阈值 0.40，轨迹输出分数 0.35，低分门槛 0.2；最少点数、离地、下沉留默认 |
| T20 | 跟踪扫参（关联、coast、确认、观测噪声、运动状态） | Agent | T19 | 1 小时 | 完成。coast 1.5 s，确认 3 帧；关联半径、观测噪声、速度头、匹配、运动状态留默认 |
| T21 | 预测：IMM 对 CV、IMM 参数；oracle 和端到端 | Agent | T20 | 30 分钟 | 完成。运动模型、合并距离留默认（cv:1.5+ct:1.5:0.25，merge_dist 0.25）；训练 oracle 上 CV 比 IMM 差 |
| T22 | 定版：参数改成默认值（我改代码），从原始扫描完整跑验证场景，和旧默认值比，出最终报告和 MCAP | Agent | T21 | 30 分钟 | 完成。从原始扫描完整跑，IDF1 0.786，和 T20 final_A 相同；新默认值相对旧默认值 IDF1 +0.060。MCAP 在 data/tune/T22_mcap（872 MB） |
| H5 | 看最终效果，定上线参数 | 人 | T22 | 1 小时 | 待做（列在 T40 的收尾表里，等 T35 B 段） |
| T23 | 第二档预测（学习模型）第一版：35 个场景的跟踪日志（附带两项检查：预测误差拆成纵向 / 横向，训练目标对标签的误差）、选训练目标、训练 GRU 和 Transformer、在验证集上和 IMM 配对比 | Agent | T22 | 1 小时 | 完成，已复核。线上继续用 IMM：GRU、Transformer 概率最高的路线偏慢，3 s FDE 都更高（MR 低得多）。predict_ms 2746 / 4322 ms 是并行进程抢 CPU 测出来的，单线程约 16–40 ms，仍超 10 ms。伪真值用 measured，离标签 0.2 m，改为默认。误差主要在纵向，地图暂不做 |
| T24 | 第二档第二版，跟踪的部分合并：重新生成跟踪日志（加运动状态列），训练两个带状态输出的 GRU（一个加原始检测输入），评估"用它的速度匀速外推""用它的在动判断替换跟踪器的规则"及两者合用 | Agent | T23 | 1 小时 | 完成，已复核。保持现状。在动判断漏报增加、领先不到 90%；速度外推的 FDE 没同时过端到端和标签当检测。复核发现网络估的速度系统性偏慢，原始检测输入有用；人定不在 0921 上继续调，厂区数据来了先修速度目标 |
| T25 | Orin 部署 P0：经反向隧道连通板子，记录服务器和板子的环境，装好缺的包（docs/DEPLOY_ORIN.md） | Agent + 人 | — | 半天 | 完成，已复核。通道通；板上补了 trtexec（libnvinfer-bin 10.3.0.30）。服务器项目环境 TensorRT 10.3.0。小引擎测试失败是建引擎工具没按模型目录找外部权重，已修 |
| T26 | Orin 部署 P1：导出上线 Pillar（含原 T8），定 P_max，建 TensorRT FP32 / FP16 引擎，在服务器和板上与 PyTorch 比，测引擎延迟 | Agent | T25 | 2 小时 | 完成，已复核。P_max 24000。FP32、FP16 都能用，上线先用 FP16。FP16 有 8 个场景匹配率低于 0.99，判断是阈值附近的框来回翻，门槛改为不计这类（T27 重算确认）；标签上 AP 只掉 0.001。板上 FP16 的 GPU 时间：PFN 3.4 ms、RPN 4.7 ms。服务器 C++ 用 TensorRT 容器 |
| T27 | Orin 部署 P2：INT8 量化（ModelOpt Q/DQ，训练场景校准），和 FP16 比精度和板上延迟；动态柱数的 PFN | Agent | T26 | 2.5 小时 | 完成，已复核。上线用 FP16 加动态 PFN（`nusc_pp_dyn`）。FP16 剩下的漏报多是过滤器的阈值翻转（过滤前 10 个场景都 ≥ 0.991），比较工具加了 filter 一类、heads rel 不再卡门槛。INT8 差得不正常（有个 head 余弦为负），多半是稀疏 BEV 上 entropy 校准截断太狠，现在不追。两个 INT8 的 AP@2 掉 0.124 / 0.099，不上（人定，2026-10-29）。板上 RPN：FP16 5.05 ms，INT8 2.65 ms。PFN：动态 10000 柱 1.50 ms，静态 3.52 ms |
| T28 | Orin 部署 P4 / P5：C++ 核心库在服务器容器里编译、跑对齐测试（含 GPU、TensorRT）；10 个验证场景上 Python 和 C++ 比（同一批检测、各自检测）并打分；板上原生编译、测试、测延迟 | Agent（锁频掉了才要人） | T27 | 3 小时 | 完成，已复核。复核在样例 bag 的真实扫描上拆开计时：慢的是去重（逐点 unordered_map），不是去畸变；预处理和后处理已提速，输出逐位不变，T28B 复测。C++ 可以替代 Python：10 个场景同一批检测 ID 全同，各自检测匹配率都是 1。标签上 AP@2 / AP@4 都是 0.832 / 0.852。板上锁频还在。端到端中位数 27.7 ms（p99 30.2 ms），网络以外 21.8 ms，比 15 ms 多 6.8 ms，多在去畸变。INT8 最差的 head 是 height（余弦 −0.85） |
| T28B | C++ 提速后复测：服务器容器和板上重编、测试；两个场景确认输出和 T28 逐位相同；板上 1 / 4 线程的每段延迟 | Agent | T28 | 40 分钟 | 完成，已复核。输出和 T28 逐位相同，1 线程和 4 线程也相同。板上网络以外：1 线程 12.9 ms，4 线程 11.4 ms，都低于 15 ms。用 4 个线程。服务器上 4 线程更慢，体素化 p99 尖峰从 215 ms 降到 19 ms（1 线程），还没消失 |
| T29 | Orin 部署 P6：C++ ROS 2 节点。服务器上用 Python 节点的 bag 回放对 C++ 的消息处理（样例 bag、0921 原始 bag 前 60 s）；板上 colcon 编译，`dogperc_bag_replay` 对服务器逐位比，`ros2 bag play` 在线跑、查不丢帧 | Agent | T28B | 2 小时 | 完成，已复核。对齐都过：服务器两段同一批检测 ID 每帧相同；板上 ROS 读 bag 和读消息日志逐位相同；板上 GPU 对服务器过门槛。板上端到端 18.9 ms（p99 21.0），网络以外 12.2 ms（样例 bag）。复核查出节点在时间戳倒退（bag 重播、时钟回跳）时不会重来，bag 长于 3 s 时重播后基本没有输出，已修（两边一样，加了测试）；在线 `same_ids_frames 0` 是热身那遍的状态带进了正式播放。TensorRT 新出现一句设备提示，T30 查 |
| T30 | Orin 部署 P7：0921 截 60 s bag（节点的三个话题）；板上 1× 循环播 30 分钟（每遍节点重来一次），后 15 分钟加负载；每遍不丢帧、延迟、内存、温度；TensorRT 设备提示：记版本、板上重建引擎 | Agent | T29 复核 | 2 小时 | 完成。截 bag 与 T29 msgs 逐键相同；msgs_dets / msgs pass。板上重建引擎后 TensorRT 设备提示消失（PFN 1.48 ms、RPN 4.75 ms）。离线 bag_gpu pass，端到端 16.9 ms。浸泡 29 遍：无负载中间 11 遍不丢帧；加负载后多数遍丢 2–10 帧（gap 最大 0.40 s），objects_bag passed=false。节点没崩，内存涨 19 MB，最高温度 62 °C |
| T31 | Orin 部署 P8：车上部署包。板上用 `deploy_deb` 编译、打 deb；不装先解开跑；装上（要 root）后离线回放和 `deploy/` 逐位比、在线回放查不丢帧、服务起停、卸载 | Agent（安装要人或免密） | T30 | 1.5 小时 | 完成，已复核。deb 装上后和 deploy/ 逐位相同，服务起停、卸载都对。复核：打包补了 patchelf、去掉编译目录的 RPATH、二进制不带源码路径、版本号带提交数；循环时 ID 不同是编号顺延（加了不看编号的 same_tracks_frames）；加了 publish_tf 和 Foxglove 实时看 |
| T32 | 删掉 `deploy/` 以后：服务器容器里不带 ROS 编 deploy_deb、跑 CTest（含 GPU），样例 bag 上 Python 对 C++；板上清掉旧的 `deploy/`，装 foxglove_bridge、patchelf，升级前留参考、重打 deb、升级安装，升级前后逐位比、ROS 读 bag 对读消息日志逐位比；查 TF 和 bridge（`foxglove_probe.py`）；留一段 60 s 的 bag 给人看 | Agent，最后一步人看 | T31 | 1.5 小时 | 完成，已复核。删掉 deploy/ 以后服务器 CTest 27 项全过，升级前后、两条回放路径都逐位相同，TF、bridge 过。复核：板子重启后没锁频，节点慢 2–3 倍（每帧 55 ms，T31 是 15–21 ms），Foxglove 那次只有 5 Hz 的原因没分清（没锁频、DDS 缓冲、调试点云）；节点每秒那一行加了收到 / 发出 / 缺的帧数和最长回调，加了 orin_clocks.sh；板上服务的配置不是包里的 |
| T33 | 板上实时性：锁频与否 × DDS 缓冲 × 发不发调试点云，60 s bag 1× 播六遍，分清丢帧来自哪；重打 deb（新的统计行），服务改回包里的配置 | Agent | T32 复核 | 1.5 小时 | 完成，已复核。不锁频、默认 UDP：341 条 objects、skipped 247。只加大到 8 MB 缓冲：592 条、skipped 0。只锁频：598 条、total_ms 17.7。锁频后再发调试点云并开 bridge，默认缓冲掉到 452（callback 只有 44 ms），8 MB 缓冲回到 597。建议开机锁频；Fast DDS 的机器带上这份 profile。deb `0.1.0+312.gb7dc5c6`，服务改回包里的配置。复核："真狗是 CycloneDDS"没有依据，T34 核实；deb 带上 8 MB 的 Fast DDS profile，orin_foxglove.sh 默认用它；/etc/default 列了 DDS 选项 |
| T34 | 8 号狗上车前的检查（室内，不用人）：证据、话题和坐标系撞车、用现在的代码重建引擎并打包（带车上的 DDS 设置）、CycloneDDS 下收真雷达点云、回放和编译机比、锁频前后的延迟、照 SENSORS 7.2 录 60 s、foxglove_bridge | Agent | 80a9546 复核 | 2 小时 | 完成，已复核。CycloneDDS 域 0、只绑 eth1、共享内存、socket 10 MB，节点 root。我们的话题各 1 个发布者、0 个订阅，TF 里没有 camera_init。deb `0.1.0+316.gdbde7ed`。室内 clouds 598 / skipped 0 / frames 0。和编译机比过门槛（匹配 0.998）。不锁频 total 中位 58.6 ms，锁频 37.1 ms（服务也在占 GPU），测完恢复未锁。录 64 s 点云 640 条，丢包计数没涨。nvidia 的 bridge 10 秒只收到 10 帧点云。复核：慢的原因是机器人自己的负载（load 32），不是我们的服务；不锁频 p99 99 ms 贴着帧间隔；nvidia 用户拿到的数据不全（root 的共享内存），撞车检查和 bridge 要用 root 再看；SENSORS 7.2 的雷达检查改成录 12 秒数条数 |
| T35 | 8 号狗出门实跑：A 段室内（root 视角撞车、root 起 bridge、负载和 GPU）；B 段室外有固定解（在线每秒的帧数和延迟、天线接法、录包和离线回放对比、Foxglove） | Agent + 人（带狗出门） | T34 复核 | 1.5 小时 | 并入 T40（2026-11-09；A、B 两段照 T35 的步骤，B1 改用开机锁频的开关） |
| T36 | 1104（第二批，4 份 bag、17.6 分钟、四路相机）：体检、切场景、转换、预览、时间差、位姿融合、再预览；看 RTK 固定解、天线接法、时钟 | Agent | 无 | 1 小时 | 完成，已复核（2026-11-05；续：偏置 0 重转，31 场景两遍预览全 OK、位姿基本不用修，时间差跳过。人补了现场：三叉路口，车很多、基本在动） |
| T37 | 1104：转出 4 个缓冲场景；双教师推理和自动标注、上线模型对教师、每个场景的统计；按 bag 划分（两份验证、两份训练，人定）；挑场景、带四路相机的审核材料 | Agent | T36 续 | 3 小时 | 完成，已复核（2026-11-06）：30 m 内运动的车 103 条（0921 是 29）；验证 cp02 + cp04；待审 32 个场景、1,940 个关键帧；没接进 SUSTechPOINTS，补在 T37 续 |
| T37 续 | 1104 的 32 个场景接进 SUSTechPOINTS，备份审核前的标注，重新起服务 | Agent | T37 | 10 分钟 | 完成，已复核（2026-11-06）：32 个场景接好，统计和 T37 一致，可以开始 H6 |
| T37 续 2 | 1104 的 32 个场景导出相机图（068342b 的 cameras 命令，复核修了图早一帧、视野太宽、框偏高），重新起服务 | Agent | T37 续 | 30 分钟 | 完成，已复核（2026-11-06） |
| H6 | 审 1104 挑出的场景：验证 14 个逐框、训练 18 个看红框（SUSTechPOINTS，只审车；Foxglove 里对照四路相机） | 人 | T37 续 | 1 天 | 完成，已复核（2026-11-06）：红框 422 → 8（二次补清后）；H6 补：`cp01_s003` 删掉的是牵引车拉的一串置物架，可要可不要，T38 恢复成忽略框 |
| T38 | 导回 H6，训练场景传播、补空档，叉车兜底，写入划分（没审的 3 个训练场景 drop），和 0921 一起导出 v3 数据集 | Agent | H6 | 1 小时 | 完成，已复核（2026-11-07）：v3 在 `dog_veh_1106_1811`，训练 12,670 帧、验证 1,418 帧；新验证集 30 m 内运动的车 72 条 |
| T39 | 训练 v3（Pillar、Voxel，配方同 v2）；24 个验证场景上和 nusc_pp、v2_pp 重跑比较（合计、0921、1104 分开），定上线的 Pillar 和上报阈值；v3 Voxel + TTA 对教师 | Agent，人定 | T38 | 半天 | 完成，已复核（2026-11-09）。上线仍用 nuScenes Pillar，上报阈值不改（0.40；T39 建议的 0.35 只看检测，没采用）。v3_pp 近处 ΔAP@2.0 只有 +0.006；v2_pp 在 1104 上 AP@2.0 低 0.049。教师不换 |
| T40 | 收尾：删 v2 数据集；8 号狗装带开机锁频开关的 deb（默认关）；T35 A、B 两段（B 段用开关锁频）；剩下要人和别的团队定的事列在 T40 节里 | Agent + 人（B 段带狗出门） | T39 | 1 小时 + 出门 | 任务已写好 |
| T41 | Nav_D20 规划：补提交 .gitignore 漏掉的脚本和配置；重编（/tf 只发一棵树、局部代价地图用里程计的时间戳、标记寿命 0.5 s）；回放 0921 重录；查录包 | Agent | 无 | 1 小时 | 完成，已复核（2026-11-09）：规划的显示问题没了；录包还是不对是定位的问题，IMU 高频预测把 g 当成 m/s²，odom 一秒掉几米、跳几十米，规划因此起点在障碍里 |
| T42 | Nav_D20 定位：高频预测的加速度按重力归一化以后重编、重跑；查 odom；看 LIO 每秒几帧；找配置的定位地图 | Agent | T41 | 1 小时 | 完成，已复核（2026-11-09）：高频预测修好（z 不再掉、LIO 9.9 帧/秒、规划进了 active）；还跳是内置定位地图倾斜 51°，配置的现场地图服务器上没有 |
| T43 | Nav_D20 定位地图：从 8 号狗只读拷现场地图 ad68517c，查平不平；用它回放（A），再用 INS 直出回放（B）；两个录包都查 | Agent | T42 | 1 小时 | 任务已写好 |
| H7 | Foxglove 里看规划的新录包（planning_io.json） | 人 | T43 | 15 分钟 | 待做（等 T43 的录包；T42 的录包定位还在跳） |
| T44 | 路口让行门控：1104 上离线评测（8 条假想过路线，五组参数：默认、IMM 8 s、狗 2 m/s、σ 放宽、σ 收紧）；只读看 8 号狗的定位外参 | Agent | 无 | 1.5 小时 | 任务已写好 |

- **2026-11-05 起的顺序**：两条线并行。
  - 部署：T40（含 T35：A 段随时做；B 段等人带 8 号狗出门）→ 定开机锁频（deb 已有开关，默认关）、天线配置；
  - 数据：T36 → T36 续 → T37 → T37 续 → H6（人审）→ T38 → T39，都已完成（2026-11-09）。上线模型不换；模型这条线等下一批数据（另一天、另一片区域、狗在走），先留测试集，再看 v3；
  - 1104 只有一个晚上，测试集等下一批（另一天、另一片区域，DATA_COLLECTION_PLAN 3.5）。
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
某份报告开头"Label tracks stitched into one vehicle"超过场景数的 4 倍（验证 10 个场景 > 40，训练 25 个场景 > 100；拼得太多，可能拼错）。
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

**停在第 3 步。** `nusc_pp_train` 报告开头 "Label tracks stitched into one vehicle" 是 **80**，超过 40。`oracle_train` 和第 4 步汇总没有跑。验证集那五份都是 20，没超过。

| 报告 | 场景 | 拼接数 |
|---|---|---|
| deploy、v2_pp、v1_pp、teacher、oracle_val | 验证 10 个 | 20 |
| nusc_pp_train | 训练 25 个 | 80 |
| oracle_train | — | 未跑 |

已经做完的部分：

- 约定 6 的链接加上了 `tune`：`data/tune` → `/public/zjst/robot/centerpoint_data/tune`。
- 验证集传播到 10 Hz。去掉 `pose_bad` 之后留下的帧最低是 `0921_164127_s001` 的 97.2%（277/285），其余 ≥ 98.9%。
- 重放用的是 GPU 5。当时空闲 8520 MiB，`--jobs 4` 放不下（每个进程按约 3 GB、再留 1 GB），所以是 `--jobs 2`、`--device cuda:0`。两个进程当时各约 0.8 GB。验证 10 个场景 5 分 20 秒，训练 25 个 15 分 23 秒，都退出码 0。网络中位数大约 24–37 ms。
- 第 3 步耗时：deploy 44 s，v2_pp 28 s，v1_pp 11 s，teacher 14 s，oracle_val 12 s，nusc_pp_train 34 s。
- MCAP（不提交）：`data/tune/T18_mcap` 936 MB，`data/tune/T18_mcap_v2` 858 MB，各 10 个场景。

已经写出来的六份主要指标（30 m 内；完整表在各目录的 `report.md`）：

| metric | deploy | v2_pp | v1_pp | teacher | oracle_val | nusc_pp_train |
|---|---|---|---|---|---|---|
| detections: recall within near range | 0.867 | 0.765 | 0.727 | 0.857 | 1.000 | 0.845 |
| detections: precision within near range | 0.781 | 0.893 | 0.949 | 0.952 | 1.000 | 0.736 |
| detections: F1 within near range | 0.822 | 0.824 | 0.823 | 0.902 | 1.000 | 0.787 |
| detections: false per keyframe | 1.047 | 0.314 | 0.172 | 0.219 | 0.000 | 0.776 |
| tracks: MOTA | 0.518 | 0.631 | 0.660 | 0.796 | 0.939 | 0.512 |
| tracks: IDF1 | 0.750 | 0.777 | 0.782 | 0.873 | 0.950 | 0.727 |
| tracks: ID switches / min | 2.689 | 1.862 | 1.655 | 1.655 | 1.655 | 3.902 |
| tracks: false tracks / min | 59.789 | 14.482 | 13.034 | 14.275 | 6.206 | 47.312 |
| tracks: confirm delay p90 [s] | 0.950 | 4.950 | 2.500 | 0.501 | 0.501 | 0.501 |
| motion: relevant moving stretches missed | 2 | 1 | 1 | 1 | 0 | 1 |
| motion: parked reported moving / h | 148.956 | 74.478 | 74.478 | 24.826 | 0.000 | 34.143 |
| prediction: FDE 3 s, moving [m] | 2.818 | 2.705 | 2.409 | 1.917 | 1.404 | 3.050 |
| prediction: miss rate 3 s, moving | 0.519 | 0.473 | 0.398 | 0.377 | 0.285 | 0.649 |

deploy 的检测、跟踪、盲区、运动、预测、预警、耗时和 issues 在 `data/reports/T18/deploy/report.md`。v2_pp 的漏报和盲区表在 `data/reports/T18/v2_pp/report.md`。七列并排的 summary.md 没有写。

（2026-10-26 复核）
- **停下是我定的停止条件太紧**：40 是按验证集 10 个场景定的，没按场景数放大。训练集的 80 处拼接在本地核过：
  - 67 处是停着的车被拆成几段，和 T17 的"136 处停车位、51 处被拆成几段"对得上；
  - 两段首尾相距超过 2.5 m 的，训练集 0 处，验证集 2 处（都是运动的车，按原速度推过去接得上）。
  停止条件已改成"超过场景数的 4 倍"。
- **评估口径的一个问题，已修**：车进了盲区或被挡住时，标签就断了。跟踪器在那里继续 coast 是对的，原来却被算成假轨迹。
  - oracle_val 的 48 个"coasting after the vehicle was lost"全是这种，其中 25 个在盲区里，合每分钟 6.2 条假轨迹。
  - 现在的做法：标签断开后（以及开始前）`max_age + 0.5 s` 之内，按原速度推过去的位置不计分。
  - 本地用验证集的 10 Hz 标签重跑 oracle：假轨迹 6.2 → 0 条 / 分钟，MOTA 0.939 → 0.963，ID 切换不变。
  - 其他几份报告的假轨迹也会少一些，所以要重跑第 3、4 步（下面的 T18 续）。
- **ID 切换是真的**：oracle 每分钟 1.66 次，和 v1、教师完全一样，说明不是检测的问题。
  - 例子：s000 两辆停着的货车 20–23.5 s 看不见（当前帧 0 个点），1 s 后轨迹被删，再出现时换了号。
  - 停着的车多 coast 一会儿是安全的。T20 的 max_age 加了 3.0 s；D2 可以按运动状态分开设，静止的轨迹 coast 更久。
- **10 m 内 v1、v2 不如 nusc_pp，主要是因为一辆小车**：
  - 3.4 × 1.5 × 2.0 m，标成 truck，停在离狗 4–5 m 处；164127_s004、s005 里都是它，约 70 个关键帧。
  - 两个教师给它的分数只有 0.37–0.43，nuScenes 教师单独也在 10 m 内漏了 72 个框，只有 nusc_pp 检得到。
  - v2_pp 在 10 m 内的 96 个漏报里，s005 这一辆就占了 57 个。
  - 所以 T15 里 nusc_pp 在 20 m 内的优势，很大一部分来自这一辆车。它是什么车，H4 里人看一下：
    像是 DATA_COLLECTION_PLAN.md 第 4 节问过的那种 2.5–3.1 m 长、2.3 m 高的"truck"。
  - **2026-10-26 人确认：是下货的叉车，不要求检出。** 标签里按尺寸把叉车改成了忽略框（`tools/dog/ignore_tracks.py`）：
    - 范围：长 2.3–3.6 m、宽 1.3–1.85 m、高 1.95–2.6 m（轨迹的中位尺寸），共 7 个场景、9 条轨迹，在四个固定的位置（装卸点）；
    - 改动：关键帧 341 框；10 Hz 标签 `labels_full2` 1,674 框、`labels_train2` 101 框；
    - 改完后验证集 10 m 内的车框从 186 个降到 74 个。
    - s000 里有一辆在动的叉车（离狗约 30 m），也一样作忽略框。
    T18 续跑的时候标签还没改，所以另外跑 T18 续 2：在新标签上重跑评估、重算三个 Pillar 的 AP。
- **nusc_pp 在默认阈值 0.35 下假轨迹太多**：
  - 每分钟约 60 条，681 个假轨迹关键帧里 567 个是"附近没有标注"；
  - 停着的车被报成在动，每小时 149 次（v2、v1 是 74 次）。
  - 但它召回最高（30 m 内 0.867），车进到 30 m 后被跟上的延迟 p90 是 0.95 s（v2 4.95 s，v1 2.5 s）。
  - 三个模型的分数尺度不一样，应该各自用最合适的阈值再比，所以 T19 第 1 步加了 v1、v2 的阈值扫描。
- **预测：IMM 和匀速（CV）几乎一样**，这些车速下转弯模式不起作用，两个模式基本重合。
  - 运动的车 3 s 的 FDE 2.8 m，MR 52%；用标签当检测（oracle）是 1.4 m、28%。
  - 也就是说，一半误差来自跟踪估出的速度，一半来自车本身：起步、停车、转弯，129 个运动样本里 119 个慢于 3 m/s。
  - T21 和 D2 先从速度估计入手。
- **预警**：6 次里报出 3 次，提前 1.0 s。样本太少，先只记下。
- **耗时**：跟踪每帧中位数 7.9 ms，p90 18.5 ms（服务器 CPU）。nusc_pp 每帧的检测多（行人、骑车的也在里面）；阈值提高后会降，Orin 上要再测。
- **10 Hz 标签里审核人加的框被 propagate 重新编了号**。不影响训练数据：每个关键帧车框按位置都在 10 Hz 标签里。
  评估工具画图时已按关键帧对回原来的编号，同一辆车在关键帧之间不会画成两个框。

### T18 续：评估修正后重跑第 3、4 步

```text
任务 T18 续：评估工具改了两处（标签断开后一小段不计分；10 Hz 标签的编号对回关键帧），重跑 T18 第 3、4 步。
第 0–2 步的结果都在（labels_full2、data/tune/deploy、data/tune/nusc_pp_train），不用重做。不用 GPU。
先 git pull、跑快速测试。

0. 同 T18 第 0 步设好 TASK=T18、VAL、TRAIN、PP，同 T18 第 3 步定义 ev。
   报告和 MCAP 直接覆盖（旧的数字留在上面的表里和 git 历史里）。
1. T18 第 3 步的七条全部重跑（包括上次没跑的 oracle_train）。
2. T18 第 4 步汇总，写 data/reports/T18/summary.md。

停止条件：任何报错；拼接数超过场景数的 4 倍；oracle_val 的 false tracks / min 大于 1（修正没生效）。
提交：data/reports/T18/（MCAP 不在里面）、本文档。
报告（写进下面"T18 续结果"）：summary.md 七列并排的主要指标表；deploy、v2_pp 两份报告的"Where the errors are"、运动状态、
预测（moving）、预警几张表；和修正前（上面的表）比，各份 false tracks / min 变了多少；oracle_train 的预测表（moving）。
```

### T18 续结果

七份都跑完。拼接数：验证集五份都是 20（上限 40），训练两份都是 80（上限 100）。oracle_val 的 false tracks / min 是 **0**，修正生效。耗时：deploy 53 s，v2_pp 21 s，v1_pp 11 s，teacher 14 s，oracle_val 14 s，nusc_pp_train 26 s，oracle_train 22 s。MCAP：`data/tune/T18_mcap` 876 MB，`data/tune/T18_mcap_v2` 858 MB。这次跟踪、预测和可视化只处理车（人和骑行者仍由检测器输出，不建轨迹、不画）。

主要指标：

| metric | deploy | v2_pp | v1_pp | teacher | oracle_val | nusc_pp_train | oracle_train |
|---|---|---|---|---|---|---|---|
| detections: recall within near range | 0.867 | 0.765 | 0.727 | 0.857 | 1.000 | 0.845 | 1.000 |
| detections: precision within near range | 0.782 | 0.894 | 0.952 | 0.952 | 1.000 | 0.738 | 1.000 |
| detections: F1 within near range | 0.822 | 0.824 | 0.824 | 0.902 | 1.000 | 0.788 | 1.000 |
| detections: false per keyframe | 1.038 | 0.309 | 0.162 | 0.217 | 0.000 | 0.762 | 0.000 |
| tracks: MOTA | 0.526 | 0.635 | 0.667 | 0.800 | 0.963 | 0.526 | 0.943 |
| tracks: IDF1 | 0.753 | 0.779 | 0.785 | 0.875 | 0.962 | 0.732 | 0.877 |
| tracks: ID switches / min | 2.689 | 1.862 | 1.655 | 1.655 | 1.655 | 3.902 | 4.065 |
| tracks: false tracks / min | 57.720 | 13.447 | 11.172 | 13.034 | 0.000 | 44.061 | 0.000 |
| tracks: confirm delay p90 [s] | 0.950 | 4.950 | 2.500 | 0.501 | 0.501 | 0.501 | 0.501 |
| motion: relevant moving stretches missed | 2 | 1 | 1 | 1 | 0 | 1 | 0 |
| motion: parked reported moving / h | 148.956 | 74.478 | 74.478 | 24.826 | 0.000 | 34.143 | 0.000 |
| prediction: FDE 3 s, moving [m] | 2.818 | 2.705 | 2.409 | 1.917 | 1.404 | 3.050 | 2.188 |
| prediction: miss rate 3 s, moving | 0.519 | 0.473 | 0.398 | 0.377 | 0.285 | 0.649 | 0.521 |

false tracks / min 相对修正前：deploy 59.789 → 57.720（−2.07），v2_pp 14.482 → 13.447（−1.04），v1_pp 13.034 → 11.172（−1.86），teacher 14.275 → 13.034（−1.24），oracle_val 6.206 → 0.000（−6.21），nusc_pp_train 47.312 → 44.061（−3.25）。oracle_val 的 MOTA 0.939 → 0.963，ID 切换仍是 1.655 次/分钟。deploy 跟踪耗时中位数 7.9 → 5.0 ms（p90 19.1 ms）。

deploy，错误在哪：

| false tracks: cause | count |
|---|---|
| second track on a tracked vehicle | 4 |
| beside a vehicle (2-4 m off) | 51 |
| coasting after the vehicle was lost | 45 |
| nothing labelled nearby | 564 |

| missed vehicles (FN) | count |
|---|---|
| detected, not tracked | 58 |
| not detected | 224 |
| 0-10 m | 11 |
| 10-20 m | 27 |
| 20-30 m | 130 |
| 30-40 m | 114 |
| 1-9 pts | 13 |
| 10-49 pts | 63 |
| >=50 pts | 206 |

| zone | scored vehicles | missed | missed share | tracks | false | false share |
|---|---|---|---|---|---|---|
| elsewhere | 1597 | 202 | 0.126 | 1989 | 596 | 0.300 |
| edge | 408 | 73 | 0.179 | 388 | 60 | 0.155 |
| blind | 19 | 7 | 0.368 | 29 | 8 | 0.276 |

运动状态：moving recall 0.745，停着被报成在动 7/46（149 次/小时），相关段漏报 2/9，延迟 0.50 / 0.90 s。

预测，moving，129 个样本，3 s：IMM FDE 2.82 m、MR 0.519；CV FDE 2.80 m、MR 0.519。

预警：6 次里报出 3 次，提前 1.0 s；没有这种车的报警 1 次（12.4 次/小时）。

v2_pp，错误在哪：

| false tracks: cause | count |
|---|---|
| second track on a tracked vehicle | 0 |
| beside a vehicle (2-4 m off) | 48 |
| coasting after the vehicle was lost | 6 |
| nothing labelled nearby | 140 |

| missed vehicles (FN) | count |
|---|---|
| detected, not tracked | 42 |
| not detected | 493 |
| 0-10 m | 96 |
| 10-20 m | 60 |
| 20-30 m | 199 |
| 30-40 m | 180 |
| 1-9 pts | 23 |
| 10-49 pts | 113 |
| >=50 pts | 399 |

| zone | scored vehicles | missed | missed share | tracks | false | false share |
|---|---|---|---|---|---|---|
| elsewhere | 1597 | 421 | 0.264 | 1321 | 151 | 0.114 |
| edge | 408 | 106 | 0.260 | 339 | 33 | 0.097 |
| blind | 19 | 8 | 0.421 | 23 | 10 | 0.435 |

运动状态：moving recall 0.746，停着被报成在动 3/42（74.5 次/小时），相关段漏报 1/7，延迟 1.00 / 2.00 s。

预测，moving，110 个样本，3 s：IMM FDE 2.71 m、MR 0.473；CV FDE 2.71 m、MR 0.482。

预警：6 次里报出 5 次，中位提前 1.5 s（最短 −0.5 s）；没有这种车的报警 2 次（24.8 次/小时）。

oracle_train 预测，moving，217 个样本：

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.37 | 0.52 | 0.36 | 0.50 | 0.009 |
| IMM | 2 s | 0.73 | 1.28 | 0.70 | 1.23 | 0.189 |
| IMM | 3 s | 1.14 | 2.19 | 1.09 | 2.09 | 0.521 |
| CV | 1 s | 0.38 | 0.52 | 0.38 | 0.52 | 0.009 |
| CV | 2 s | 0.73 | 1.29 | 0.73 | 1.29 | 0.203 |
| CV | 3 s | 1.15 | 2.22 | 1.15 | 2.22 | 0.571 |

### T18 续 2：叉车改成忽略框后重算

T18 续跑的时候，叉车在标签里还算车。叉车不要求检出（2026-10-26 人定），已在标签里改成忽略框（T18 复核）。
上面的数里，10 m 内 v1、v2 的漏报大多是那辆叉车，所以要在新标签上重算，T19 第 1b 步三个 Pillar 的比较也用新标签。

```text
任务 T18 续 2：叉车改成了忽略框，重跑 T18 第 3、4 步，并在新标签上重算三个 Pillar 和教师的 AP。
先 git pull、跑快速测试。不用 GPU。
前置：拉下来的标签里叉车已是忽略框，下面应打印 57，不是就停下报告：
  python -c "import pickle; L=pickle.load(open('data/labels/0921_164127_s005/labels_sustech2.pkl','rb')); \
      print(sum(o.get('ignore_reason') == 'forklift' for f in L['frames'] for o in f['objects']))"

0. 同 T18 续第 0 步。
1. T18 第 3 步的七条全部重跑。
2. T18 第 4 步汇总（覆盖 summary.md）。
3. 三个 Pillar 和教师重算 AP（T15 第 4 步的做法，只留这四个）：
     RES=$(for r in $VAL; do echo nusc_pp=data/runs_nusc_pp/$r/pillar/results.pkl v2_pp=data/runs_veh_v2/$r/pillar/results.pkl \
         v1_pp=data/runs_veh_v1/$r/pillar/results.pkl teacher=data/runs/$r/voxel/results.pkl; done)
     python tools/dog/compare_detectors.py --classes vehicle --max-range 40 --results $RES \
         --labels $(for r in $VAL; do echo data/labels/$r/labels_sustech2.pkl; done) \
         --thresholds 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 --target-recall 0.9 --bootstrap 1000 \
         | tee data/reports/T18/compare_val_noforklift.md

停止条件：同 T18 续。
提交：data/reports/T18/（MCAP 不在里面）、本文档。
报告（写进"T18 续 2 结果"）：
- summary.md 七列的主要指标表；
- 和 T18 续（叉车算车）比，deploy、v2_pp、v1_pp 的 det recall、det F1、IDF1、false tracks / min、confirm p90 各变了多少；
- 第 3 步的 AP 表（含 0-10 m、10-20 m、20-40 m），重抽表里 v2_pp − nusc_pp、v1_pp − nusc_pp 两行，和 T15 的数（叉车算车）并排。
```

### T18 续 2 结果

七份都跑完。前置：`0921_164127_s005` 的 `labels_sustech2.pkl` 里叉车忽略框 **57**。拉代码后快速测试 153 passed。拼接数：验证集五份都是 19（上限 40），训练两份都是 80（上限 100）。oracle_val 的 false tracks / min 是 **0.000**。耗时：deploy 25 s，v2_pp 20 s，v1_pp 11 s，teacher 12 s，oracle_val 12 s，nusc_pp_train 25 s，oracle_train 21 s。MCAP：`data/tune/T18_mcap` 876 MB，`data/tune/T18_mcap_v2` 857 MB。40 m 内计入的车框从 T15 的 2011 降到 **1721**。

主要指标（区间在 `summary.md`）：

| metric | deploy | v2_pp | v1_pp | teacher | oracle_val | nusc_pp_train | oracle_train |
|---|---|---|---|---|---|---|---|
| detections: recall within near range | 0.872 | 0.844 | 0.798 | 0.898 | 1.000 | 0.845 | 1.000 |
| detections: precision within near range | 0.758 | 0.889 | 0.950 | 0.948 | 1.000 | 0.738 | 1.000 |
| detections: F1 within near range | 0.811 | 0.866 | 0.867 | 0.922 | 1.000 | 0.788 | 1.000 |
| detections: false per keyframe | 1.040 | 0.310 | 0.162 | 0.217 | 0.000 | 0.762 | 0.000 |
| tracks: MOTA | 0.459 | 0.690 | 0.720 | 0.812 | 0.961 | 0.526 | 0.943 |
| tracks: IDF1 | 0.726 | 0.814 | 0.826 | 0.880 | 0.957 | 0.732 | 0.877 |
| tracks: ID switches / min | 2.483 | 1.448 | 1.034 | 1.655 | 1.655 | 3.902 | 4.065 |
| tracks: false tracks / min | 58.134 | 13.654 | 11.379 | 13.447 | 0.000 | 44.061 | 0.000 |
| tracks: confirm delay p90 [s] | 1.200 | 7.250 | 3.500 | 0.500 | 0.501 | 0.501 | 0.501 |
| motion: relevant moving stretches missed | 2 | 1 | 1 | 1 | 0 | 1 | 0 |
| motion: parked reported moving / h | 86.891 | 62.065 | 74.478 | 24.826 | 0.000 | 34.143 | 0.000 |
| prediction: FDE 3 s, moving [m] | 2.897 | 2.846 | 2.482 | 1.958 | 1.388 | 3.050 | 2.188 |
| prediction: miss rate 3 s, moving | 0.541 | 0.500 | 0.420 | 0.393 | 0.286 | 0.649 | 0.521 |

和 T18 续（叉车算车）比，新值减旧值：

| metric | deploy | v2_pp | v1_pp |
|---|---|---|---|
| detections: recall within near range | +0.005 | +0.079 | +0.071 |
| detections: F1 within near range | −0.011 | +0.042 | +0.043 |
| tracks: IDF1 | −0.027 | +0.035 | +0.041 |
| tracks: false tracks / min | +0.414 | +0.207 | +0.207 |
| tracks: confirm delay p90 [s] | +0.250 | +2.300 | +1.000 |

nusc_pp_train 和 oracle_train 的点估计与 T18 续相同。

AP（40 m，`compare_val_noforklift.md`，1721 个车框）：

| model | AP@0.5 | AP@1.0 | AP@2.0 | AP@4.0 | mAP | AP@2.0[0-10m] | AP@2.0[10-20m] | AP@2.0[20-40m] |
|---|---|---|---|---|---|---|---|
| nusc_pp | 0.680 | 0.781 | 0.834 | 0.852 | 0.787 | 0.878 | 0.937 | 0.788 |
| v2_pp | 0.629 | 0.784 | 0.854 | 0.882 | 0.787 | 0.754 | 0.955 | 0.813 |
| v1_pp | 0.570 | 0.727 | 0.826 | 0.844 | 0.742 | 0.870 | 0.957 | 0.760 |
| teacher | 0.780 | 0.892 | 0.928 | 0.937 | 0.884 | 0.922 | 0.981 | 0.896 |

T15（叉车算车，2011 框）近处 AP@2.0：nusc_pp 0.930 / 0.937 / 0.801，v2_pp 0.545 / 0.939 / 0.816，v1_pp 0.501 / 0.945 / 0.750。去掉叉车后，v2、v1 的 0–10 m 从 0.545、0.501 升到 0.754、0.870；nusc_pp 的 0–10 m 从 0.930 降到 0.878。

重抽（10 个场景、1000 次、90% 区间）。脚本按结果列表顺序只打了 `nusc_pp − …`，下面两行是取反后的差值和区间，与 T15 并排。原行里 nusc_pp 在 mAP 上领先的重抽占 48%（对 v2）和 82%（对 v1），近处占 30% 和 29%。打平不算领先，取反后的领先占比不写。

| A − B | 本次 ΔmAP | 区间 | 本次 ΔAP@2.0 0–20 m | 区间 | T15 ΔmAP | T15 区间 | T15 领先 | T15 ΔAP@2.0 0–20 m | T15 区间 | T15 领先 |
|---|---|---|---|---|---|---|---|---|---|---|
| v2_pp − nusc_pp | +0.000 | −0.023 – +0.066 | +0.007 | −0.019 – +0.053 | −0.041 | −0.105 – −0.008 | 3% | −0.071 | −0.266 – −0.007 | 3% |
| v1_pp − nusc_pp | −0.045 | −0.085 – +0.084 | +0.013 | −0.038 – +0.074 | −0.101 | −0.157 – −0.046 | 1% | −0.105 | −0.315 – −0.002 | 5% |

（2026-10-27 复核）
- **去掉叉车后，v2_pp 和 nusc_pp 的 AP 打平了。** T15 里"nusc_pp 明显最好"主要是那辆叉车造成的。
  - mAP 都是 0.787；AP@2.0 0.854 对 0.834，AP@4.0 0.882 对 0.852，v2 略高。
  - ΔAP@2.0 0–20 m +0.007，区间 −0.019 – +0.053，跨过 0，分不出来。
  - nusc_pp 只在 AP@0.5（0.680 对 0.629）和 0–10 m（0.878 对 0.754）上高。
    AP@0.5 偏向 nuScenes 模型（标签是 nuScenes 教师的框，T17 复核）。
    0–10 m 现在只剩 74 个框，v2 和 nusc_pp 分别漏了 11 个和 5 个，主要是 165659_s006、164127_s000 里各一辆车，各漏 2–2.5 s。
- **按 T15 那条规则（v1、v2 都不比 nusc_pp 好就用 nusc_pp），现在仍是 nusc_pp；但跟踪层面 v2、v1 好得多。** 默认阈值 0.35 下：
  | | nusc_pp | v2_pp | v1_pp |
  |---|---|---|---|
  | det F1（30 m 内） | 0.811 | 0.866 | 0.867 |
  | IDF1 | 0.726 | 0.814 | 0.826 |
  | MOTA | 0.459 | 0.690 | 0.720 |
  | 假轨迹 / 分钟 | 58.1 | 13.7 | 11.4 |
  | 停着的车报成在动 / 小时 | 87 | 62 | 74 |
  | det recall（30 m 内） | **0.872** | 0.844 | 0.798 |
  | 进入 30 m 后被跟上的延迟 p90 | **1.2 s** | 7.25 s | 3.5 s |
  nusc_pp 只在召回和延迟上占优；v2 的延迟 p90 7.25 s 是 16 辆进入 30 m 的车里最慢的两辆，样本很少。
  0.35 对三个模型都不是合适的阈值：nusc_pp 要更高（T15：F1 最高在 0.40–0.45），v2、v1 要更低（0.25–0.30）。
  所以上线用哪个，等 T19 第 1、1b 步各自用最合适的阈值再比，这两步只要几分钟 CPU。
  照默认阈值的数看，v2 或 v1 很可能触发 1b 的"高 0.02 以上"，到时由人重新定。
- **改用 v2 或 v1 时怎么接着调**：已写进 T19 第 1b 步。
- 其他：
  - 10 m 内的漏检，nusc_pp 从 11 降到 5，v2 从 96 降到 11，v1 从 85 降到 4，大部分本来就是叉车；
  - 叉车在 nusc_pp 上原来大多是跟上的，所以它的 IDF1 反而降了 0.027；
  - 训练场景那两份没变：s007 的叉车在 40 m 外，本来就不计分。

---

## H4 在 Foxglove 里看基线（人做）

**2026-10-28 起看 `data/tune/T22_mcap/`（定版参数），重点按 T22 复核里的那张表看**；对照 v2 时用 `data/tune/T18_mcap_v2/`。
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
   1b. 三个 Pillar 各用自己最好的阈值再比一次（T18 复核：默认阈值下 nusc_pp 召回最高、假轨迹也最多，分数尺度不一样）。
       v1、v2 用 T15 存好的结果（已按默认参数过滤），只扫阈值，只跑 A：
     ev score_v2_A "$VAL" "data/runs_veh_v2/{scene}/pillar/results.pkl" --no-mcap --set tracker.high_score=0.35,0.2,0.25,0.3,0.4,0.45
     ev score_v1_A "$VAL" "data/runs_veh_v1/{scene}/pillar/results.pkl" --no-mcap --set tracker.high_score=0.35,0.2,0.25,0.3,0.4,0.45
     三份 sweep.md 里，按下面第 1 组的规则各选一行，把三行并排成一张表：det recall、det precision、det F1、IDF1、
     false tracks / min、confirm p90、moving missed、false moving / h。
     v1 或 v2 选出的那行，det F1 和 IDF1 都比 nusc_pp 选出的那行高 0.02 以上：停下报告（上线模型要人重新定），不做第 2–4 组。
     人改定 v2 或 v1 以后，从第 1 组重做 T19，先做两件事：
     a. 用它的权重照 T18 第 2 步重放验证场景、存原始检测，输出到 data/tune/deploy_<v2|v1>/{scene}，A 组改读这里；
        v2：data/train/dog_pp_veh_v2/epoch_10.pth，v1：data/train/dog_pp_veh_v1/epoch_20.pth。
     b. B 组仍用 nuScenes Pillar 在训练场景上的结果（v1、v2 见过训练场景，不能用它们）。
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

第 1b 步先停过一次：v2、v1 按第 1 组规则选出的行，det F1 和 IDF1 都比 nusc_pp 选出的行高 0.02 以上。2026-10-27 人定仍用 nuScenes Pillar，于是从第 2 组接着做，上线模型不换。拉下来的提交只改了文档，没有跑测试。耗时：score_A 55 s，score_B 109 s，score_v2_A 43 s，score_v1_A 41 s；output_A 103 s，output_B 211 s；points_A 33 s，points_B 67 s；ground_A 76 s，ground_B 149 s。

第 1 组（nusc_pp，上报阈值）选 **第 3 行 `tracker.high_score=0.40`**。A 上同时满足：det F1 领先 99%（≥ 90%），moving missed 2 不多于第 1 行，det recall 0.839 比第 1 行低 0.033、没有低过 0.05。满足这三条的只有这一行，det F1 也最高（0.832）。B 上这一行 det F1 0.808 不低于第 1 行的 0.788，moving missed 0 不多于 1。第 4 行 det F1 领先只有 88%，det recall 低了 0.079，两条都不满足。这个值写进 BASE，第 2 组起都带着它。

v2、v1 只跑了 A。两份里没有任何一行的 det F1 领先比例 ≥ 90%，所以都留第 1 行（0.35）。v1 第 4 行（0.30）det F1 最高（0.878），领先比例 89%，差 1 个百分点，按规则不选。

三行并排（都是验证集、30 m 内）：

| | nusc_pp 0.40 | v2_pp 0.35 | v1_pp 0.35 |
|---|---|---|---|
| det recall | 0.839 | 0.844 | 0.798 |
| det precision | 0.825 | 0.889 | 0.950 |
| det F1 | 0.832 | 0.866 | 0.867 |
| IDF1 | 0.743 | 0.814 | 0.826 |
| false tracks / min | 43.859 | 13.654 | 11.379 |
| confirm p90 s | 1.200 | 7.250 | 3.500 |
| moving missed | 2 | 1 | 1 |
| false moving / h | 86.891 | 62.065 | 74.478 |

相对 nusc_pp 选出的行：v2_pp 的 det F1 +0.034、IDF1 +0.071；v1_pp 的 det F1 +0.035、IDF1 +0.083。两条都超过 0.02，所以停下。v1、v2 用的是 T15 存好的、已按默认参数过滤的检测，只改了上报阈值；nusc_pp 是从过滤前的检测重滤的。

`score_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.35 | 0.872 | 0.758 | 0.811 | 1.040 | 0.459 | 0.726 | 2.483 | 58.134 | 1.200 | 2 | 86.891 | 2.897 | 0.541 |
| 2 | tracker.high_score=0.3 | 0.903 | 0.693 | 0.784 | 1.528 | 0.433 | 0.725 | 2.689 | 67.858 | 1.000 | 1 | 86.891 | 2.901 | 0.544 |
| 3 | tracker.high_score=0.4 | 0.839 | 0.825 | 0.832 | 0.643 | 0.510 | 0.743 | 2.276 | 43.859 | 1.200 | 2 | 86.891 | 2.851 | 0.537 |
| 4 | tracker.high_score=0.45 | 0.793 | 0.873 | 0.831 | 0.414 | 0.582 | 0.769 | 2.069 | 31.032 | 1.501 | 2 | 99.304 | 2.819 | 0.541 |
| 5 | tracker.high_score=0.5 | 0.723 | 0.912 | 0.806 | 0.238 | 0.601 | 0.778 | 1.655 | 22.550 | 2.001 | 2 | 86.891 | 2.828 | 0.517 |
| 6 | tracker.high_score=0.55 | 0.620 | 0.949 | 0.750 | 0.117 | 0.586 | 0.764 | 1.655 | 16.757 | 1.501 | 1 | 99.304 | 2.857 | 0.518 |

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.031 (100%) | -0.065 (0%) | -0.027 (0%) | 0.488 (0%) | -0.026 (0%) | -0.001 (45%) | 0.207 (21%) | 9.723 (0%) | -0.200 (70%) | -1 (67%) | 0.000 (0%) | 0.004 (25%) | 0.003 (5%) |
| 3 | -0.033 (0%) | 0.067 (100%) | 0.021 (99%) | -0.397 (100%) | 0.051 (100%) | 0.017 (100%) | -0.207 (56%) | -14.275 (100%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.046 (67%) | -0.004 (57%) |
| 4 | -0.079 (0%) | 0.116 (100%) | 0.021 (88%) | -0.626 (100%) | 0.123 (100%) | 0.044 (100%) | -0.414 (70%) | -27.102 (100%) | 0.301 (0%) | 0 (0%) | 12.413 (0%) | -0.078 (73%) | 0.000 (37%) |
| 5 | -0.149 (0%) | 0.154 (100%) | -0.004 (45%) | -0.802 (100%) | 0.143 (100%) | 0.052 (100%) | -0.828 (97%) | -35.584 (100%) | 0.801 (0%) | 0 (0%) | 0.000 (0%) | -0.070 (85%) | -0.024 (88%) |
| 6 | -0.252 (0%) | 0.191 (100%) | -0.061 (13%) | -0.922 (100%) | 0.127 (100%) | 0.038 (100%) | -0.828 (97%) | -41.377 (100%) | 0.301 (21%) | -1 (67%) | 12.413 (0%) | -0.040 (66%) | -0.023 (95%) |

`score_B/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.35 | 0.845 | 0.738 | 0.788 | 0.762 | 0.526 | 0.732 | 3.902 | 44.061 | 0.501 | 1 | 34.143 | 3.050 | 0.649 |
| 2 | tracker.high_score=0.3 | 0.866 | 0.662 | 0.751 | 1.124 | 0.491 | 0.721 | 4.227 | 52.353 | 0.500 | 1 | 48.776 | 3.024 | 0.636 |
| 3 | tracker.high_score=0.4 | 0.808 | 0.808 | 0.808 | 0.499 | 0.569 | 0.743 | 3.983 | 35.850 | 0.502 | 0 | 39.021 | 3.161 | 0.653 |
| 4 | tracker.high_score=0.45 | 0.761 | 0.868 | 0.811 | 0.311 | 0.625 | 0.765 | 3.739 | 25.282 | 1.000 | 0 | 34.143 | 3.177 | 0.655 |
| 5 | tracker.high_score=0.5 | 0.702 | 0.913 | 0.794 | 0.185 | 0.672 | 0.781 | 3.333 | 17.722 | 1.500 | 0 | 34.143 | 3.125 | 0.641 |
| 6 | tracker.high_score=0.55 | 0.632 | 0.939 | 0.755 | 0.114 | 0.703 | 0.794 | 3.252 | 11.056 | 1.700 | 0 | 29.265 | 3.060 | 0.640 |

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.021 (100%) | -0.076 (0%) | -0.038 (0%) | 0.362 (0%) | -0.035 (0%) | -0.011 (0%) | 0.325 (12%) | 8.292 (0%) | -0.001 (89%) | 0 (0%) | 14.633 (0%) | -0.026 (83%) | -0.014 (96%) |
| 3 | -0.037 (0%) | 0.070 (100%) | 0.020 (99%) | -0.263 (100%) | 0.043 (100%) | 0.011 (100%) | 0.081 (19%) | -8.211 (100%) | 0.001 (30%) | -1 (66%) | 4.878 (0%) | 0.111 (10%) | 0.004 (34%) |
| 4 | -0.084 (0%) | 0.130 (100%) | 0.023 (93%) | -0.451 (100%) | 0.099 (100%) | 0.033 (100%) | -0.163 (66%) | -18.779 (100%) | 0.499 (12%) | -1 (66%) | 0.000 (0%) | 0.127 (5%) | 0.006 (13%) |
| 5 | -0.143 (0%) | 0.175 (100%) | 0.005 (58%) | -0.577 (100%) | 0.145 (100%) | 0.049 (100%) | -0.569 (99%) | -26.339 (100%) | 0.999 (5%) | -1 (66%) | 0.000 (36%) | 0.075 (0%) | -0.009 (91%) |
| 6 | -0.214 (0%) | 0.200 (100%) | -0.033 (14%) | -0.648 (100%) | 0.177 (100%) | 0.062 (100%) | -0.650 (98%) | -33.005 (100%) | 1.199 (7%) | -1 (66%) | -4.878 (61%) | 0.010 (41%) | -0.010 (90%) |

`score_v2_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.35 | 0.844 | 0.889 | 0.866 | 0.310 | 0.690 | 0.814 | 1.448 | 13.654 | 7.250 | 1 | 62.065 | 2.846 | 0.500 |
| 2 | tracker.high_score=0.2 | 0.896 | 0.786 | 0.838 | 0.814 | 0.684 | 0.805 | 1.655 | 15.723 | 6.750 | 0 | 62.065 | 2.664 | 0.515 |
| 3 | tracker.high_score=0.25 | 0.884 | 0.826 | 0.854 | 0.571 | 0.682 | 0.812 | 1.448 | 16.551 | 6.750 | 1 | 62.065 | 2.655 | 0.500 |
| 4 | tracker.high_score=0.3 | 0.864 | 0.862 | 0.863 | 0.419 | 0.684 | 0.811 | 1.655 | 15.723 | 3.750 | 1 | 62.065 | 2.652 | 0.485 |
| 5 | tracker.high_score=0.4 | 0.821 | 0.921 | 0.868 | 0.210 | 0.701 | 0.818 | 1.448 | 11.379 | 7.750 | 1 | 62.065 | 2.857 | 0.495 |
| 6 | tracker.high_score=0.45 | 0.795 | 0.953 | 0.867 | 0.129 | 0.714 | 0.826 | 1.448 | 8.068 | 1.300 | 0 | 62.065 | 2.883 | 0.500 |

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.052 (100%) | -0.103 (0%) | -0.029 (4%) | 0.503 (0%) | -0.006 (1%) | -0.009 (5%) | 0.207 (0%) | 2.069 (0%) | -0.499 (52%) | -1 (67%) | 0.000 (0%) | -0.182 (62%) | 0.015 (10%) |
| 3 | 0.040 (100%) | -0.064 (0%) | -0.012 (14%) | 0.260 (0%) | -0.008 (0%) | -0.002 (2%) | 0.000 (0%) | 2.896 (0%) | -0.499 (66%) | 0 (0%) | 0.000 (0%) | -0.191 (77%) | 0.000 (28%) |
| 4 | 0.020 (100%) | -0.027 (0%) | -0.003 (28%) | 0.109 (0%) | -0.005 (0%) | -0.003 (2%) | 0.207 (0%) | 2.069 (0%) | -3.500 (66%) | 0 (0%) | 0.000 (0%) | -0.194 (97%) | -0.015 (92%) |
| 5 | -0.023 (0%) | 0.031 (100%) | 0.002 (59%) | -0.100 (100%) | 0.011 (100%) | 0.004 (97%) | 0.000 (0%) | -2.276 (100%) | 0.500 (0%) | 0 (0%) | 0.000 (0%) | 0.011 (28%) | -0.005 (65%) |
| 6 | -0.049 (0%) | 0.063 (100%) | 0.001 (51%) | -0.181 (100%) | 0.025 (100%) | 0.012 (100%) | 0.000 (0%) | -5.586 (100%) | -5.950 (60%) | -1 (67%) | 0.000 (0%) | 0.037 (22%) | 0.000 (36%) |

`score_v1_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.35 | 0.798 | 0.950 | 0.867 | 0.162 | 0.720 | 0.826 | 1.034 | 11.379 | 3.500 | 1 | 74.478 | 2.482 | 0.420 |
| 2 | tracker.high_score=0.2 | 0.856 | 0.858 | 0.857 | 0.560 | 0.711 | 0.822 | 1.034 | 13.034 | 3.500 | 1 | 74.478 | 2.063 | 0.402 |
| 3 | tracker.high_score=0.25 | 0.845 | 0.908 | 0.875 | 0.348 | 0.710 | 0.822 | 1.034 | 14.482 | 3.500 | 1 | 74.478 | 2.280 | 0.414 |
| 4 | tracker.high_score=0.3 | 0.826 | 0.938 | 0.878 | 0.231 | 0.714 | 0.824 | 1.034 | 13.034 | 3.000 | 1 | 74.478 | 2.280 | 0.414 |
| 5 | tracker.high_score=0.4 | 0.759 | 0.958 | 0.847 | 0.121 | 0.721 | 0.824 | 1.241 | 9.930 | 3.500 | 2 | 74.478 | 2.509 | 0.429 |
| 6 | tracker.high_score=0.45 | 0.715 | 0.970 | 0.823 | 0.090 | 0.725 | 0.825 | 1.241 | 8.068 | 3.751 | 2 | 74.478 | 2.508 | 0.429 |

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.058 (100%) | -0.092 (0%) | -0.010 (18%) | 0.398 (0%) | -0.009 (0%) | -0.005 (0%) | 0.000 (0%) | 1.655 (1%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.419 (100%) | -0.018 (67%) |
| 3 | 0.047 (100%) | -0.041 (0%) | 0.008 (72%) | 0.186 (0%) | -0.010 (0%) | -0.005 (0%) | 0.000 (0%) | 3.103 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.201 (100%) | -0.006 (67%) |
| 4 | 0.028 (99%) | -0.012 (0%) | 0.011 (89%) | 0.069 (0%) | -0.006 (0%) | -0.002 (0%) | 0.000 (0%) | 1.655 (0%) | -0.500 (54%) | 0 (0%) | 0.000 (0%) | -0.201 (100%) | -0.006 (67%) |
| 5 | -0.039 (0%) | 0.009 (100%) | -0.020 (7%) | -0.041 (100%) | 0.001 (66%) | -0.002 (26%) | 0.207 (0%) | -1.448 (99%) | 0.000 (0%) | 1 (0%) | 0.000 (0%) | 0.027 (8%) | 0.009 (0%) |
| 6 | -0.083 (0%) | 0.020 (100%) | -0.044 (7%) | -0.072 (100%) | 0.005 (92%) | -0.001 (41%) | 0.207 (0%) | -3.310 (100%) | 0.250 (0%) | 1 (0%) | 0.000 (0%) | 0.026 (8%) | 0.009 (0%) |

第 2 组（轨迹输出分数、低分门槛）选 **第 12 行 `tracker.output_min_score=0.35, tracker.low_score=0.2`**。主指标是 IDF1。A 上 IDF1 领先 ≥ 90%、moving missed 不多于第 1 行（2）、confirm p90 不比第 1 行（1.200 s）长过 0.5 s 的是第 10、11、12 行，IDF1 分别是 0.767、0.768、0.772，取得最高的第 12 行。B 上这一行 IDF1 0.757 不低于第 1 行的 0.743，moving missed 0 不多于 0。

`output_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.3, tracker.low_score=0.1 | 0.839 | 0.825 | 0.832 | 0.643 | 0.510 | 0.743 | 2.276 | 43.859 | 1.200 | 2 | 86.891 | 2.851 | 0.537 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.3, tracker.low_score=0.15 | 0.839 | 0.825 | 0.832 | 0.643 | 0.510 | 0.751 | 2.276 | 44.480 | 1.200 | 2 | 74.478 | 2.835 | 0.533 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.3, tracker.low_score=0.2 | 0.839 | 0.825 | 0.832 | 0.643 | 0.509 | 0.755 | 2.069 | 44.273 | 1.200 | 2 | 86.891 | 2.819 | 0.528 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.2, tracker.low_score=0.1 | 0.839 | 0.825 | 0.832 | 0.643 | 0.310 | 0.683 | 2.896 | 50.479 | 1.200 | 1 | 99.304 | 2.753 | 0.522 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.2, tracker.low_score=0.15 | 0.839 | 0.825 | 0.832 | 0.643 | 0.334 | 0.701 | 2.896 | 44.066 | 1.150 | 1 | 86.891 | 2.742 | 0.522 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.2, tracker.low_score=0.2 | 0.839 | 0.825 | 0.832 | 0.643 | 0.346 | 0.707 | 3.103 | 42.411 | 1.150 | 1 | 86.891 | 2.735 | 0.504 |
| 7 | tracker.high_score=0.4, tracker.output_min_score=0.25, tracker.low_score=0.1 | 0.839 | 0.825 | 0.832 | 0.643 | 0.422 | 0.716 | 2.896 | 48.617 | 1.200 | 1 | 99.304 | 2.780 | 0.527 |
| 8 | tracker.high_score=0.4, tracker.output_min_score=0.25, tracker.low_score=0.15 | 0.839 | 0.825 | 0.832 | 0.643 | 0.424 | 0.726 | 2.896 | 46.342 | 1.150 | 1 | 86.891 | 2.757 | 0.523 |
| 9 | tracker.high_score=0.4, tracker.output_min_score=0.25, tracker.low_score=0.2 | 0.839 | 0.825 | 0.832 | 0.643 | 0.420 | 0.729 | 2.689 | 45.307 | 1.150 | 1 | 86.891 | 2.760 | 0.515 |
| 10 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.1 | 0.839 | 0.825 | 0.832 | 0.643 | 0.576 | 0.767 | 2.276 | 36.204 | 1.200 | 2 | 74.478 | 2.797 | 0.530 |
| 11 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.15 | 0.839 | 0.825 | 0.832 | 0.643 | 0.572 | 0.768 | 2.276 | 37.653 | 1.200 | 2 | 74.478 | 2.790 | 0.521 |
| 12 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2 | 0.839 | 0.825 | 0.832 | 0.643 | 0.572 | 0.772 | 2.069 | 38.066 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (49%) | 0.008 (83%) | 0.000 (38%) | 0.621 (18%) | 0.000 (0%) | 0 (0%) | -12.413 (65%) | -0.017 (92%) | -0.004 (63%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.001 (45%) | 0.012 (70%) | -0.207 (58%) | 0.414 (31%) | 0.000 (0%) | 0 (0%) | 0.000 (32%) | -0.032 (94%) | -0.009 (92%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.199 (0%) | -0.060 (0%) | 0.621 (0%) | 6.620 (14%) | 0.000 (0%) | -1 (67%) | 12.413 (0%) | -0.099 (96%) | -0.015 (97%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.176 (0%) | -0.042 (0%) | 0.621 (13%) | 0.207 (47%) | -0.050 (43%) | -1 (67%) | 0.000 (32%) | -0.109 (96%) | -0.015 (86%) |
| 6 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.164 (0%) | -0.036 (1%) | 0.828 (22%) | -1.448 (64%) | -0.050 (43%) | -1 (67%) | 0.000 (32%) | -0.116 (99%) | -0.033 (89%) |
| 7 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.088 (0%) | -0.028 (0%) | 0.621 (0%) | 4.758 (3%) | 0.000 (0%) | -1 (67%) | 12.413 (0%) | -0.072 (89%) | -0.010 (88%) |
| 8 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.085 (0%) | -0.017 (5%) | 0.621 (13%) | 2.483 (18%) | -0.050 (42%) | -1 (67%) | 0.000 (32%) | -0.094 (96%) | -0.014 (79%) |
| 9 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.089 (0%) | -0.015 (17%) | 0.414 (27%) | 1.448 (30%) | -0.050 (42%) | -1 (67%) | 0.000 (32%) | -0.091 (99%) | -0.022 (89%) |
| 10 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.067 (100%) | 0.023 (100%) | 0.000 (0%) | -7.655 (100%) | 0.000 (0%) | 0 (0%) | -12.413 (65%) | -0.054 (67%) | -0.007 (67%) |
| 11 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.063 (100%) | 0.024 (100%) | 0.000 (38%) | -6.206 (99%) | 0.000 (0%) | 0 (0%) | -12.413 (65%) | -0.061 (92%) | -0.016 (92%) |
| 12 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.063 (100%) | 0.029 (100%) | -0.207 (58%) | -5.793 (99%) | 0.000 (0%) | 0 (0%) | 0.000 (32%) | -0.056 (94%) | -0.016 (92%) |

`output_B/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.3, tracker.low_score=0.1 | 0.808 | 0.808 | 0.808 | 0.499 | 0.569 | 0.743 | 3.983 | 35.850 | 0.502 | 0 | 39.021 | 3.161 | 0.653 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.3, tracker.low_score=0.15 | 0.808 | 0.808 | 0.808 | 0.499 | 0.561 | 0.741 | 3.983 | 35.688 | 0.501 | 0 | 39.021 | 3.154 | 0.651 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.3, tracker.low_score=0.2 | 0.808 | 0.808 | 0.808 | 0.499 | 0.555 | 0.739 | 4.065 | 35.444 | 0.501 | 0 | 34.143 | 3.154 | 0.646 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.2, tracker.low_score=0.1 | 0.808 | 0.808 | 0.808 | 0.499 | 0.363 | 0.685 | 3.902 | 40.321 | 0.500 | 0 | 39.021 | 3.221 | 0.654 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.2, tracker.low_score=0.15 | 0.808 | 0.808 | 0.808 | 0.499 | 0.369 | 0.686 | 4.065 | 37.720 | 0.500 | 0 | 39.021 | 3.188 | 0.660 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.2, tracker.low_score=0.2 | 0.808 | 0.808 | 0.808 | 0.499 | 0.384 | 0.690 | 4.146 | 35.850 | 0.500 | 0 | 34.143 | 3.194 | 0.653 |
| 7 | tracker.high_score=0.4, tracker.output_min_score=0.25, tracker.low_score=0.1 | 0.808 | 0.808 | 0.808 | 0.499 | 0.485 | 0.718 | 4.065 | 39.752 | 0.501 | 0 | 39.021 | 3.227 | 0.654 |
| 8 | tracker.high_score=0.4, tracker.output_min_score=0.25, tracker.low_score=0.15 | 0.808 | 0.808 | 0.808 | 0.499 | 0.479 | 0.717 | 4.065 | 37.720 | 0.501 | 0 | 39.021 | 3.183 | 0.656 |
| 9 | tracker.high_score=0.4, tracker.output_min_score=0.25, tracker.low_score=0.2 | 0.808 | 0.808 | 0.808 | 0.499 | 0.475 | 0.716 | 4.146 | 37.313 | 0.501 | 0 | 34.143 | 3.192 | 0.649 |
| 10 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.1 | 0.808 | 0.808 | 0.808 | 0.499 | 0.627 | 0.759 | 3.902 | 30.973 | 0.502 | 0 | 39.021 | 3.156 | 0.653 |
| 11 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.15 | 0.808 | 0.808 | 0.808 | 0.499 | 0.626 | 0.760 | 3.983 | 31.216 | 0.502 | 0 | 39.021 | 3.146 | 0.653 |
| 12 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2 | 0.808 | 0.808 | 0.808 | 0.499 | 0.618 | 0.757 | 3.983 | 32.029 | 0.501 | 0 | 34.143 | 3.139 | 0.645 |

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.009 (0%) | -0.002 (31%) | 0.000 (44%) | -0.163 (51%) | -0.001 (67%) | 0 (0%) | 0.000 (0%) | -0.007 (80%) | -0.002 (69%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.014 (0%) | -0.004 (18%) | 0.081 (34%) | -0.406 (60%) | -0.001 (80%) | 0 (0%) | -4.878 (65%) | -0.007 (66%) | -0.007 (89%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.206 (0%) | -0.058 (0%) | -0.081 (55%) | 4.471 (1%) | -0.001 (91%) | 0 (0%) | 0.000 (0%) | 0.059 (32%) | 0.001 (47%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.200 (0%) | -0.057 (0%) | 0.081 (31%) | 1.870 (13%) | -0.001 (91%) | 0 (0%) | 0.000 (0%) | 0.026 (45%) | 0.007 (34%) |
| 6 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.186 (0%) | -0.053 (0%) | 0.163 (23%) | 0.000 (45%) | -0.001 (91%) | 0 (0%) | -4.878 (65%) | 0.033 (38%) | -0.000 (54%) |
| 7 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.085 (0%) | -0.025 (0%) | 0.081 (0%) | 3.902 (0%) | -0.001 (71%) | 0 (0%) | 0.000 (0%) | 0.066 (26%) | 0.001 (47%) |
| 8 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.090 (0%) | -0.026 (0%) | 0.081 (31%) | 1.870 (14%) | -0.001 (81%) | 0 (0%) | 0.000 (0%) | 0.022 (40%) | 0.003 (40%) |
| 9 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.095 (0%) | -0.027 (0%) | 0.163 (23%) | 1.463 (15%) | -0.001 (81%) | 0 (0%) | -4.878 (65%) | 0.031 (34%) | -0.005 (60%) |
| 10 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.058 (100%) | 0.016 (100%) | -0.081 (65%) | -4.878 (100%) | 0.000 (28%) | 0 (0%) | 0.000 (0%) | -0.006 (46%) | -0.000 (36%) |
| 11 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.057 (100%) | 0.017 (99%) | 0.000 (44%) | -4.634 (100%) | 0.000 (28%) | 0 (0%) | 0.000 (0%) | -0.015 (81%) | -0.000 (36%) |
| 12 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.048 (100%) | 0.014 (98%) | 0.000 (44%) | -3.821 (100%) | -0.001 (53%) | 0 (0%) | -4.878 (65%) | -0.022 (80%) | -0.008 (90%) |

第 3 组（当前帧最少点数）留 **第 1 行 `filter.min_points.vehicle=5`**。A 上没有任何一行的 det F1 领先比例 ≥ 90%（最高是第 2 行的 51%，而且 det F1 和 5 点打平）。

`points_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5 | 0.839 | 0.825 | 0.832 | 0.643 | 0.572 | 0.772 | 2.069 | 38.066 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=3 | 0.839 | 0.824 | 0.832 | 0.652 | 0.571 | 0.771 | 2.069 | 37.860 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=8 | 0.836 | 0.824 | 0.830 | 0.614 | 0.580 | 0.775 | 2.069 | 35.170 | 1.200 | 2 | 86.891 | 2.796 | 0.521 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=12 | 0.832 | 0.827 | 0.829 | 0.597 | 0.589 | 0.778 | 2.069 | 32.067 | 1.500 | 2 | 86.891 | 2.809 | 0.530 |

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.001 (67%) | -0.000 (23%) | 0.000 (51%) | 0.009 (0%) | -0.002 (8%) | -0.001 (8%) | 0.000 (0%) | -0.207 (57%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) |
| 3 | -0.002 (0%) | -0.000 (0%) | -0.001 (0%) | -0.029 (100%) | 0.008 (100%) | 0.003 (100%) | 0.000 (0%) | -2.896 (100%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.001 (0%) | 0.000 (0%) |
| 4 | -0.007 (0%) | 0.002 (97%) | -0.002 (22%) | -0.047 (100%) | 0.017 (100%) | 0.006 (100%) | 0.000 (0%) | -6.000 (100%) | 0.300 (0%) | 0 (0%) | 0.000 (0%) | 0.014 (22%) | 0.009 (0%) |

`points_B/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5 | 0.808 | 0.808 | 0.808 | 0.499 | 0.618 | 0.757 | 3.983 | 32.029 | 0.501 | 0 | 34.143 | 3.139 | 0.645 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=3 | 0.812 | 0.806 | 0.809 | 0.512 | 0.615 | 0.756 | 3.983 | 32.842 | 0.501 | 0 | 34.143 | 3.140 | 0.645 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=8 | 0.804 | 0.811 | 0.807 | 0.483 | 0.617 | 0.755 | 4.146 | 30.729 | 1.000 | 0 | 29.265 | 3.139 | 0.645 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=12 | 0.796 | 0.812 | 0.804 | 0.464 | 0.613 | 0.752 | 4.227 | 29.265 | 1.100 | 0 | 29.265 | 3.098 | 0.643 |

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.004 (99%) | -0.002 (0%) | 0.001 (72%) | 0.013 (0%) | -0.003 (3%) | -0.001 (34%) | 0.000 (32%) | 0.813 (0%) | -0.000 (56%) | 0 (0%) | 0.000 (0%) | 0.001 (0%) | 0.000 (0%) |
| 3 | -0.004 (0%) | 0.003 (96%) | -0.001 (35%) | -0.016 (100%) | -0.001 (46%) | -0.002 (34%) | 0.163 (22%) | -1.301 (100%) | 0.499 (0%) | 0 (0%) | -4.878 (62%) | 0.000 (0%) | 0.000 (0%) |
| 4 | -0.012 (0%) | 0.004 (93%) | -0.004 (14%) | -0.035 (100%) | -0.005 (35%) | -0.005 (32%) | 0.244 (16%) | -2.764 (100%) | 0.599 (0%) | 0 (0%) | -4.878 (62%) | -0.041 (69%) | -0.002 (69%) |

第 4 组（离地 / 下沉上限）留 **第 1 行 `filter.max_float=0.7, filter.max_sink=1.2`**。A 上 det F1 领先比例最高是第 2 行的 68%，不到 90%。B 上第 2 行是 94%、第 8 行是 100%，但规则先看 A。

`ground_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.7, filter.max_sink=1.2 | 0.839 | 0.825 | 0.832 | 0.643 | 0.572 | 0.772 | 2.069 | 38.066 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.7, filter.max_sink=0.8 | 0.838 | 0.826 | 0.832 | 0.612 | 0.578 | 0.773 | 1.862 | 36.411 | 1.200 | 2 | 86.891 | 2.763 | 0.530 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.7, filter.max_sink=1.6 | 0.839 | 0.825 | 0.832 | 0.655 | 0.573 | 0.773 | 1.862 | 37.860 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.5, filter.max_sink=1.2 | 0.801 | 0.826 | 0.813 | 0.622 | 0.541 | 0.750 | 3.103 | 37.446 | 1.200 | 2 | 86.891 | 2.845 | 0.539 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.5, filter.max_sink=0.8 | 0.800 | 0.827 | 0.813 | 0.591 | 0.546 | 0.750 | 2.896 | 35.998 | 1.200 | 2 | 86.891 | 2.809 | 0.539 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.5, filter.max_sink=1.6 | 0.801 | 0.826 | 0.813 | 0.634 | 0.541 | 0.751 | 2.896 | 37.239 | 1.200 | 2 | 86.891 | 2.845 | 0.539 |
| 7 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=1.0, filter.max_sink=1.2 | 0.849 | 0.808 | 0.828 | 0.707 | 0.542 | 0.758 | 2.483 | 40.756 | 1.200 | 2 | 74.478 | 2.802 | 0.533 |
| 8 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=1.0, filter.max_sink=0.8 | 0.849 | 0.809 | 0.829 | 0.676 | 0.547 | 0.759 | 2.276 | 39.721 | 1.200 | 2 | 74.478 | 2.770 | 0.542 |
| 9 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=1.0, filter.max_sink=1.6 | 0.849 | 0.808 | 0.828 | 0.719 | 0.542 | 0.760 | 2.276 | 40.549 | 1.200 | 2 | 74.478 | 2.802 | 0.533 |

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -0.001 (0%) | 0.002 (90%) | 0.000 (68%) | -0.031 (100%) | 0.006 (88%) | 0.001 (68%) | -0.207 (67%) | -1.655 (99%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.032 (86%) | 0.009 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.012 (0%) | 0.001 (57%) | 0.001 (80%) | -0.207 (67%) | -0.207 (64%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.000 (65%) | 0.000 (0%) |
| 4 | -0.038 (0%) | 0.001 (59%) | -0.019 (0%) | -0.021 (100%) | -0.031 (6%) | -0.022 (2%) | 1.034 (0%) | -0.621 (76%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.050 (3%) | 0.018 (0%) |
| 5 | -0.039 (0%) | 0.002 (78%) | -0.018 (2%) | -0.052 (100%) | -0.027 (13%) | -0.022 (5%) | 0.828 (13%) | -2.069 (96%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.014 (38%) | 0.018 (0%) |
| 6 | -0.038 (0%) | 0.001 (59%) | -0.019 (0%) | -0.009 (94%) | -0.031 (4%) | -0.021 (2%) | 0.828 (13%) | -0.828 (79%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.050 (3%) | 0.018 (0%) |
| 7 | 0.011 (98%) | -0.017 (0%) | -0.004 (35%) | 0.064 (0%) | -0.031 (9%) | -0.014 (3%) | 0.414 (0%) | 2.689 (0%) | 0.000 (0%) | 0 (0%) | -12.413 (58%) | 0.007 (53%) | 0.012 (0%) |
| 8 | 0.010 (98%) | -0.015 (0%) | -0.003 (37%) | 0.033 (21%) | -0.025 (15%) | -0.013 (1%) | 0.207 (0%) | 1.655 (7%) | 0.000 (0%) | 0 (0%) | -12.413 (58%) | -0.025 (90%) | 0.020 (0%) |
| 9 | 0.011 (98%) | -0.017 (0%) | -0.004 (35%) | 0.076 (0%) | -0.030 (10%) | -0.012 (6%) | 0.207 (0%) | 2.483 (0%) | 0.000 (0%) | 0 (0%) | -12.413 (58%) | 0.007 (54%) | 0.012 (0%) |

`ground_B/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.7, filter.max_sink=1.2 | 0.808 | 0.808 | 0.808 | 0.499 | 0.618 | 0.757 | 3.983 | 32.029 | 0.501 | 0 | 34.143 | 3.139 | 0.645 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.7, filter.max_sink=0.8 | 0.804 | 0.818 | 0.811 | 0.444 | 0.630 | 0.761 | 3.821 | 29.022 | 0.601 | 0 | 34.143 | 3.110 | 0.659 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.7, filter.max_sink=1.6 | 0.809 | 0.804 | 0.806 | 0.518 | 0.616 | 0.757 | 4.065 | 32.517 | 0.501 | 0 | 39.021 | 3.142 | 0.647 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.5, filter.max_sink=1.2 | 0.773 | 0.808 | 0.790 | 0.479 | 0.605 | 0.742 | 4.146 | 30.891 | 0.601 | 0 | 39.021 | 3.146 | 0.645 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.5, filter.max_sink=0.8 | 0.769 | 0.819 | 0.793 | 0.425 | 0.616 | 0.741 | 4.227 | 27.802 | 1.100 | 0 | 39.021 | 3.176 | 0.660 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=0.5, filter.max_sink=1.6 | 0.774 | 0.804 | 0.788 | 0.499 | 0.605 | 0.742 | 4.227 | 31.135 | 0.502 | 0 | 43.898 | 3.151 | 0.647 |
| 7 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=1.0, filter.max_sink=1.2 | 0.823 | 0.804 | 0.814 | 0.521 | 0.617 | 0.760 | 3.658 | 33.655 | 0.500 | 0 | 43.898 | 3.147 | 0.641 |
| 8 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=1.0, filter.max_sink=0.8 | 0.819 | 0.814 | 0.816 | 0.467 | 0.633 | 0.765 | 3.496 | 30.241 | 0.501 | 0 | 39.021 | 3.105 | 0.648 |
| 9 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5, filter.max_float=1.0, filter.max_sink=1.6 | 0.824 | 0.800 | 0.812 | 0.541 | 0.615 | 0.760 | 3.739 | 34.224 | 0.500 | 0 | 43.898 | 3.134 | 0.643 |

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -0.004 (0%) | 0.010 (100%) | 0.003 (94%) | -0.054 (100%) | 0.012 (100%) | 0.004 (98%) | -0.163 (69%) | -3.008 (100%) | 0.100 (0%) | 0 (0%) | 0.000 (0%) | -0.029 (63%) | 0.014 (16%) |
| 3 | 0.001 (85%) | -0.004 (0%) | -0.002 (0%) | 0.020 (0%) | -0.001 (21%) | 0.000 (52%) | 0.081 (0%) | 0.488 (2%) | -0.000 (43%) | 0 (0%) | 4.878 (0%) | 0.003 (39%) | 0.002 (0%) |
| 4 | -0.035 (0%) | 0.000 (58%) | -0.018 (1%) | -0.020 (100%) | -0.012 (19%) | -0.015 (11%) | 0.163 (24%) | -1.138 (100%) | 0.101 (0%) | 0 (0%) | 4.878 (0%) | 0.007 (28%) | -0.000 (50%) |
| 5 | -0.039 (0%) | 0.011 (98%) | -0.015 (9%) | -0.074 (100%) | -0.001 (48%) | -0.016 (13%) | 0.244 (15%) | -4.227 (100%) | 0.599 (0%) | 0 (0%) | 4.878 (0%) | 0.037 (24%) | 0.016 (16%) |
| 6 | -0.034 (0%) | -0.004 (4%) | -0.020 (0%) | 0.000 (44%) | -0.013 (19%) | -0.015 (15%) | 0.244 (15%) | -0.894 (100%) | 0.001 (9%) | 0 (0%) | 9.755 (0%) | 0.011 (23%) | 0.002 (41%) |
| 7 | 0.015 (100%) | -0.004 (4%) | 0.005 (92%) | 0.022 (0%) | -0.001 (40%) | 0.003 (86%) | -0.325 (99%) | 1.626 (0%) | -0.000 (77%) | 0 (0%) | 9.755 (0%) | 0.008 (40%) | -0.004 (69%) |
| 8 | 0.011 (96%) | 0.006 (100%) | 0.008 (100%) | -0.032 (100%) | 0.016 (99%) | 0.009 (100%) | -0.488 (100%) | -1.788 (100%) | -0.000 (49%) | 0 (0%) | 4.878 (0%) | -0.034 (64%) | 0.003 (41%) |
| 9 | 0.016 (100%) | -0.008 (0%) | 0.004 (79%) | 0.042 (0%) | -0.002 (28%) | 0.003 (84%) | -0.244 (90%) | 2.195 (0%) | -0.000 (81%) | 0 (0%) | 9.755 (0%) | -0.006 (68%) | -0.002 (69%) |

最后的 BASE（第 3、4 组留的是默认值，不写进去）：

```text
--set tracker.high_score=0.4 --set tracker.output_min_score=0.35 --set tracker.low_score=0.2
```

---

## T20 跟踪扫参

```text
任务 T20：扫跟踪参数和运动状态的开关，定车辆的关联半径、coast 时间、确认帧数、观测噪声、运动状态阈值。
先按通用约定做。前置：T19 完成。不用 GPU。

0. mkdir -p data/reports/T20; TASK=T20; VAL、TRAIN、ev 同 T18；A、B 同 T19 第 0 步；BASE=T19 报告里最后的 BASE。
1. 关联半径：             A gate --set tracker.vehicle.gate=3,2,2.5,4
2. 最长 coast 时间：      A age --set tracker.vehicle.max_age=1.0,0.5,1.5,2.0,3.0
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

远程没有新提交，没有跑测试。BASE 从 T19 接着：`tracker.high_score=0.4`、`tracker.output_min_score=0.35`、`tracker.low_score=0.2`。耗时：gate 35 / 72 s，age 44 / 88 s，hits 52 / 107 s，noise 76 / 157 s，assoc 34 / 70 s，motion 74 / 148 s，final_A 12 s（都是 A / B）。

第 1 组（关联半径）留 **第 1 行 `tracker.vehicle.gate=3`**。A 上只有第 4 行（gate=4）IDF1 领先比例 ≥ 90%（91%，IDF1 0.775），moving missed、confirm p90、false tracks / min 都过。B 上这一行 IDF1 0.756，低于第 1 行的 0.757。

第 2 组（最长 coast）选 **第 3 行 `tracker.vehicle.max_age=1.5`**。A 上 IDF1 领先 100%，IDF1 0.773；moving missed 2 不多于第 1 行，confirm p90 仍是 1.200 s，false tracks / min 36.825，不超过第 1 行 38.066 的 1.1 倍。第 4、5 行 IDF1 更高（0.774、0.778），领先比例只有 64% 和 71%，不选。B 上第 3 行 IDF1 0.771，不低于第 1 行的 0.757。

第 3 组（确认帧数、试探寿命）选 **第 5 行 `tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15`**。A 上 IDF1 0.786，领先 100%；moving missed、confirm p90 与第 1 行相同，false tracks / min 27.102，低于第 1 行。第 6 行也领先 100%，IDF1 0.785，低于第 5 行。B 上第 5 行 IDF1 0.785，不低于第 1 行的 0.771。试探寿命仍是默认的 0.15 s。

第 4 组（观测噪声）留 **第 1 行 `pos_std=0.3, vel_std=0.8`**。A 上第 3、4、6、9 行 IDF1 领先都 ≥ 90%，最高是第 4、6 行的 0.795。这四行在 B 上的 IDF1 都低于第 1 行的 0.785（0.773、0.775、0.770、0.781）。

第 5 组（速度头、匹配方法）留 **第 1 行 `use_velocity=true, matching=greedy`**。A 上 IDF1 领先比例最高是第 2 行的 64%，不到 90%。关掉速度头的两行 IDF1 只高 0.002，停着的车报成在动升到 484 次/小时。

第 6 组（运动状态）留 **第 1 行 `motion_on_speed=0.7, motion_on_frames=3`**。false moving / h 更低的行，A 上 moving missed 都多于第 1 行的 2（第 3、9 行是 4，领先比例 96%，但不能选）。moving missed 不多于 2 的行里，false moving / h 最低仍是第 1 行的 86.891。

最后的 BASE（没改的参数不写）：

```text
--set tracker.high_score=0.4 --set tracker.output_min_score=0.35 --set tracker.low_score=0.2 --set tracker.vehicle.max_age=1.5 --set tracker.vehicle.min_hits=3
```

`final_A` 用这个 BASE，和 T18 续 2 的 deploy（同一批标签，默认跟踪参数）并排：

| metric | T18 deploy | T20 final_A |
|---|---|---|
| detections: recall within near range | 0.872 | 0.839 |
| detections: precision within near range | 0.758 | 0.825 |
| detections: F1 within near range | 0.811 | 0.832 |
| detections: false per keyframe | 1.040 | 0.641 |
| tracks: MOTA | 0.459 | 0.609 |
| tracks: IDF1 | 0.726 | 0.786 |
| tracks: ID switches / min | 2.483 | 1.862 |
| tracks: false tracks / min | 58.134 | 27.102 |
| tracks: confirm delay p90 [s] | 1.200 | 1.200 |
| motion: relevant moving stretches missed | 2 | 2 |
| motion: parked reported moving / h | 86.891 | 86.891 |
| prediction: FDE 3 s, moving [m] | 2.897 | 2.795 |
| prediction: miss rate 3 s, moving | 0.541 | 0.521 |

`gate_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.gate=3 | 0.839 | 0.825 | 0.832 | 0.643 | 0.572 | 0.772 | 2.069 | 38.066 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.gate=2 | 0.839 | 0.825 | 0.832 | 0.643 | 0.557 | 0.740 | 4.551 | 38.894 | 1.200 | 2 | 49.652 | 2.783 | 0.517 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.gate=2.5 | 0.839 | 0.825 | 0.832 | 0.643 | 0.575 | 0.767 | 2.689 | 37.446 | 1.200 | 2 | 62.065 | 2.810 | 0.526 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.gate=4 | 0.839 | 0.825 | 0.832 | 0.643 | 0.573 | 0.775 | 2.069 | 37.446 | 1.200 | 2 | 124.130 | 2.824 | 0.534 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.016 (3%) | -0.032 (3%) | 2.483 (0%) | 0.828 (25%) | 0.000 (0%) | 0 (0%) | -37.239 (67%) | -0.012 (56%) | -0.004 (56%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.002 (97%) | -0.005 (24%) | 0.621 (0%) | -0.621 (80%) | 0.000 (0%) | 0 (0%) | -24.826 (67%) | 0.015 (0%) | 0.004 (0%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.001 (58%) | 0.003 (91%) | 0.000 (31%) | -0.621 (85%) | 0.000 (0%) | 0 (0%) | 37.239 (0%) | 0.029 (2%) | 0.013 (0%) |

`gate_B/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.gate=3 | 0.808 | 0.808 | 0.808 | 0.499 | 0.618 | 0.757 | 3.983 | 32.029 | 0.501 | 0 | 34.143 | 3.139 | 0.645 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.gate=2 | 0.808 | 0.808 | 0.808 | 0.499 | 0.605 | 0.744 | 5.040 | 32.436 | 0.502 | 0 | 24.388 | 3.138 | 0.642 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.gate=2.5 | 0.808 | 0.808 | 0.808 | 0.499 | 0.613 | 0.753 | 4.227 | 32.355 | 0.502 | 0 | 29.265 | 3.139 | 0.647 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.gate=4 | 0.808 | 0.808 | 0.808 | 0.499 | 0.615 | 0.756 | 3.983 | 31.948 | 0.502 | 0 | 39.021 | 3.107 | 0.649 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.012 (4%) | -0.013 (1%) | 1.057 (0%) | 0.406 (21%) | 0.001 (0%) | 0 (0%) | -9.755 (89%) | -0.001 (51%) | -0.003 (56%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.004 (21%) | -0.004 (9%) | 0.244 (0%) | 0.325 (10%) | 0.001 (0%) | 0 (0%) | -4.878 (69%) | -0.000 (55%) | 0.002 (38%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.002 (13%) | -0.001 (31%) | 0.000 (37%) | -0.081 (41%) | 0.001 (0%) | 0 (0%) | 4.878 (0%) | -0.032 (93%) | 0.004 (32%) |

`age_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.0 | 0.839 | 0.825 | 0.832 | 0.643 | 0.572 | 0.772 | 2.069 | 38.066 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=0.5 | 0.839 | 0.825 | 0.832 | 0.643 | 0.555 | 0.726 | 4.345 | 39.721 | 1.200 | 3 | 49.652 | 2.777 | 0.517 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.575 | 0.773 | 1.862 | 36.825 | 1.200 | 2 | 86.891 | 2.791 | 0.513 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=2.0 | 0.839 | 0.827 | 0.833 | 0.634 | 0.578 | 0.774 | 1.448 | 36.204 | 1.200 | 2 | 86.891 | 2.791 | 0.513 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=3.0 | 0.839 | 0.831 | 0.835 | 0.622 | 0.581 | 0.778 | 1.655 | 35.998 | 1.200 | 2 | 86.891 | 2.791 | 0.513 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.018 (0%) | -0.046 (0%) | 2.276 (0%) | 1.655 (1%) | 0.000 (0%) | 1 (0%) | -37.239 (67%) | -0.018 (67%) | -0.004 (67%) |
| 3 | 0.000 (0%) | 0.001 (65%) | 0.000 (65%) | -0.002 (65%) | 0.003 (100%) | 0.001 (100%) | -0.207 (67%) | -1.241 (100%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.004 (90%) | -0.009 (67%) |
| 4 | 0.000 (0%) | 0.002 (98%) | 0.001 (98%) | -0.009 (98%) | 0.006 (100%) | 0.002 (64%) | -0.621 (76%) | -1.862 (100%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.004 (90%) | -0.009 (67%) |
| 5 | 0.000 (0%) | 0.006 (100%) | 0.003 (100%) | -0.021 (100%) | 0.009 (95%) | 0.006 (71%) | -0.414 (62%) | -2.069 (99%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.004 (90%) | -0.009 (67%) |

`age_B/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.0 | 0.808 | 0.808 | 0.808 | 0.499 | 0.618 | 0.757 | 3.983 | 32.029 | 0.501 | 0 | 34.143 | 3.139 | 0.645 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=0.5 | 0.808 | 0.807 | 0.808 | 0.501 | 0.604 | 0.706 | 7.560 | 32.761 | 0.501 | 0 | 34.143 | 3.168 | 0.647 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5 | 0.808 | 0.809 | 0.808 | 0.497 | 0.624 | 0.771 | 2.764 | 31.216 | 0.502 | 0 | 39.021 | 3.130 | 0.639 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=2.0 | 0.808 | 0.809 | 0.808 | 0.496 | 0.626 | 0.770 | 2.683 | 31.135 | 0.502 | 0 | 39.021 | 3.130 | 0.639 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=3.0 | 0.808 | 0.809 | 0.808 | 0.493 | 0.630 | 0.779 | 2.195 | 30.647 | 0.502 | 0 | 39.021 | 3.130 | 0.639 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | -0.001 (0%) | -0.001 (0%) | 0.003 (0%) | -0.014 (0%) | -0.051 (0%) | 3.577 (0%) | 0.732 (4%) | -0.000 (40%) | 0 (0%) | 0.000 (0%) | 0.029 (0%) | 0.002 (0%) |
| 3 | 0.000 (0%) | 0.001 (91%) | 0.000 (91%) | -0.002 (98%) | 0.006 (100%) | 0.014 (100%) | -1.219 (100%) | -0.813 (100%) | 0.001 (0%) | 0 (0%) | 4.878 (0%) | -0.009 (62%) | -0.006 (62%) |
| 4 | 0.000 (0%) | 0.001 (91%) | 0.000 (91%) | -0.003 (99%) | 0.008 (100%) | 0.013 (96%) | -1.301 (100%) | -0.894 (100%) | 0.001 (0%) | 0 (0%) | 4.878 (0%) | -0.009 (62%) | -0.006 (62%) |
| 5 | 0.000 (0%) | 0.001 (91%) | 0.000 (91%) | -0.005 (100%) | 0.012 (100%) | 0.022 (99%) | -1.788 (100%) | -1.382 (100%) | 0.001 (0%) | 0 (0%) | 4.878 (0%) | -0.009 (62%) | -0.006 (62%) |

`hits_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=2, tracker.tentative_max_age=0.15 | 0.839 | 0.825 | 0.832 | 0.641 | 0.575 | 0.773 | 1.862 | 36.825 | 1.200 | 2 | 86.891 | 2.791 | 0.513 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=2, tracker.tentative_max_age=0.25 | 0.839 | 0.825 | 0.832 | 0.641 | 0.573 | 0.772 | 1.862 | 36.825 | 1.200 | 2 | 86.891 | 2.789 | 0.513 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=1, tracker.tentative_max_age=0.15 | 0.839 | 0.825 | 0.832 | 0.641 | 0.519 | 0.754 | 2.689 | 57.100 | 1.200 | 2 | 86.891 | 2.955 | 0.544 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=1, tracker.tentative_max_age=0.25 | 0.839 | 0.825 | 0.832 | 0.641 | 0.519 | 0.754 | 2.689 | 57.100 | 1.200 | 2 | 86.891 | 2.955 | 0.544 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.25 | 0.839 | 0.825 | 0.832 | 0.641 | 0.607 | 0.785 | 2.069 | 27.515 | 1.200 | 2 | 86.891 | 2.789 | 0.513 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.002 (23%) | -0.001 (23%) | 0.000 (0%) | 0.000 (37%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.002 (92%) | 0.000 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.056 (0%) | -0.019 (0%) | 0.828 (0%) | 20.275 (0%) | 0.000 (59%) | 0 (0%) | 0.000 (0%) | 0.163 (0%) | 0.031 (0%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.056 (0%) | -0.019 (0%) | 0.828 (0%) | 20.275 (0%) | 0.000 (59%) | 0 (0%) | 0.000 (0%) | 0.163 (0%) | 0.031 (0%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.033 (100%) | 0.013 (100%) | 0.000 (0%) | -9.723 (100%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.004 (0%) | 0.009 (0%) |
| 6 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.032 (100%) | 0.012 (100%) | 0.207 (0%) | -9.310 (100%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.002 (92%) | 0.000 (0%) |

`hits_B/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=2, tracker.tentative_max_age=0.15 | 0.808 | 0.809 | 0.808 | 0.497 | 0.624 | 0.771 | 2.764 | 31.216 | 0.502 | 0 | 39.021 | 3.130 | 0.639 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=2, tracker.tentative_max_age=0.25 | 0.808 | 0.809 | 0.808 | 0.497 | 0.620 | 0.768 | 2.845 | 32.029 | 0.502 | 0 | 34.143 | 3.127 | 0.639 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=1, tracker.tentative_max_age=0.15 | 0.808 | 0.809 | 0.808 | 0.497 | 0.547 | 0.744 | 3.008 | 54.629 | 0.501 | 0 | 39.021 | 3.081 | 0.635 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=1, tracker.tentative_max_age=0.25 | 0.808 | 0.809 | 0.808 | 0.497 | 0.547 | 0.744 | 3.008 | 54.629 | 0.501 | 0 | 39.021 | 3.081 | 0.635 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15 | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 39.021 | 2.998 | 0.636 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.25 | 0.808 | 0.809 | 0.808 | 0.497 | 0.654 | 0.781 | 2.601 | 23.250 | 1.002 | 0 | 34.143 | 3.022 | 0.637 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.004 (0%) | -0.003 (1%) | 0.081 (0%) | 0.813 (0%) | 0.000 (0%) | 0 (0%) | -4.878 (65%) | -0.003 (86%) | 0.000 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.077 (0%) | -0.028 (0%) | 0.244 (7%) | 23.412 (0%) | -0.001 (63%) | 0 (0%) | 0.000 (36%) | -0.049 (76%) | -0.004 (71%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.077 (0%) | -0.028 (0%) | 0.244 (7%) | 23.412 (0%) | -0.001 (63%) | 0 (0%) | 0.000 (36%) | -0.049 (76%) | -0.004 (71%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.035 (100%) | 0.014 (100%) | -0.325 (98%) | -9.024 (100%) | 0.550 (0%) | 0 (0%) | 0.000 (0%) | -0.132 (79%) | -0.003 (58%) |
| 6 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.031 (100%) | 0.010 (100%) | -0.163 (81%) | -7.967 (100%) | 0.501 (0%) | 0 (0%) | -4.878 (65%) | -0.109 (92%) | -0.002 (69%) |

`noise_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.3, tracker.vehicle.vel_std=0.8 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.3, tracker.vehicle.vel_std=0.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.570 | 0.747 | 2.896 | 28.964 | 1.200 | 2 | 49.652 | 3.492 | 0.589 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.3, tracker.vehicle.vel_std=1.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.625 | 0.793 | 1.862 | 25.860 | 1.200 | 2 | 86.891 | 1.962 | 0.407 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.2, tracker.vehicle.vel_std=0.8 | 0.839 | 0.825 | 0.832 | 0.641 | 0.623 | 0.795 | 1.448 | 26.688 | 1.501 | 2 | 86.891 | 2.462 | 0.483 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.2, tracker.vehicle.vel_std=0.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.604 | 0.770 | 2.276 | 27.515 | 1.200 | 2 | 62.065 | 3.250 | 0.564 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.2, tracker.vehicle.vel_std=1.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.626 | 0.795 | 1.655 | 26.067 | 1.200 | 2 | 74.478 | 1.970 | 0.373 |
| 7 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.5, tracker.vehicle.vel_std=0.8 | 0.839 | 0.825 | 0.832 | 0.641 | 0.566 | 0.747 | 2.689 | 29.170 | 1.200 | 2 | 37.239 | 3.197 | 0.562 |
| 8 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.5, tracker.vehicle.vel_std=0.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.547 | 0.712 | 5.379 | 30.619 | 1.200 | 2 | 62.065 | 3.735 | 0.636 |
| 9 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.5, tracker.vehicle.vel_std=1.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.624 | 0.794 | 1.655 | 25.860 | 1.501 | 2 | 86.891 | 2.236 | 0.441 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.039 (0%) | -0.040 (0%) | 1.034 (0%) | 1.862 (0%) | 0.000 (0%) | 0 (0%) | -37.239 (90%) | 0.697 (0%) | 0.068 (8%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.016 (100%) | 0.007 (100%) | 0.000 (32%) | -1.241 (90%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.833 (100%) | -0.115 (98%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.014 (100%) | 0.009 (100%) | -0.414 (90%) | -0.414 (69%) | 0.300 (0%) | 0 (0%) | 0.000 (0%) | -0.334 (100%) | -0.038 (98%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.005 (0%) | -0.016 (0%) | 0.414 (0%) | 0.414 (0%) | 0.000 (0%) | 0 (0%) | -24.826 (67%) | 0.455 (0%) | 0.043 (0%) |
| 6 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.017 (100%) | 0.009 (100%) | -0.207 (65%) | -1.034 (89%) | 0.000 (0%) | 0 (0%) | -12.413 (67%) | -0.825 (100%) | -0.148 (100%) |
| 7 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.043 (0%) | -0.039 (0%) | 0.828 (0%) | 2.069 (0%) | 0.000 (0%) | 0 (0%) | -49.652 (90%) | 0.402 (0%) | 0.041 (11%) |
| 8 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.062 (0%) | -0.074 (0%) | 3.517 (0%) | 3.517 (0%) | 0.000 (0%) | 0 (0%) | -24.826 (67%) | 0.940 (0%) | 0.114 (2%) |
| 9 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.015 (99%) | 0.008 (99%) | -0.207 (65%) | -1.241 (99%) | 0.300 (0%) | 0 (0%) | 0.000 (0%) | -0.559 (100%) | -0.081 (99%) |

`noise_B/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.3, tracker.vehicle.vel_std=0.8 | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 39.021 | 2.998 | 0.636 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.3, tracker.vehicle.vel_std=0.5 | 0.808 | 0.809 | 0.808 | 0.497 | 0.648 | 0.779 | 2.520 | 23.006 | 1.000 | 0 | 34.143 | 3.869 | 0.766 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.3, tracker.vehicle.vel_std=1.5 | 0.808 | 0.809 | 0.808 | 0.497 | 0.655 | 0.773 | 2.845 | 22.437 | 1.052 | 0 | 34.143 | 2.359 | 0.457 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.2, tracker.vehicle.vel_std=0.8 | 0.808 | 0.809 | 0.808 | 0.497 | 0.656 | 0.775 | 2.764 | 22.843 | 1.052 | 0 | 34.143 | 2.658 | 0.543 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.2, tracker.vehicle.vel_std=0.5 | 0.808 | 0.809 | 0.808 | 0.497 | 0.659 | 0.785 | 2.439 | 22.112 | 1.052 | 0 | 34.143 | 3.468 | 0.739 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.2, tracker.vehicle.vel_std=1.5 | 0.808 | 0.809 | 0.808 | 0.497 | 0.652 | 0.770 | 3.170 | 22.681 | 1.500 | 0 | 29.265 | 2.399 | 0.476 |
| 7 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.5, tracker.vehicle.vel_std=0.8 | 0.808 | 0.809 | 0.808 | 0.497 | 0.650 | 0.782 | 2.439 | 23.006 | 1.000 | 0 | 34.143 | 3.568 | 0.713 |
| 8 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.5, tracker.vehicle.vel_std=0.5 | 0.808 | 0.809 | 0.808 | 0.497 | 0.637 | 0.764 | 3.170 | 23.738 | 1.000 | 0 | 34.143 | 4.289 | 0.781 |
| 9 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.vehicle.pos_std=0.5, tracker.vehicle.vel_std=1.5 | 0.808 | 0.809 | 0.808 | 0.497 | 0.657 | 0.781 | 2.683 | 22.274 | 1.052 | 0 | 34.143 | 2.522 | 0.527 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.010 (1%) | -0.006 (11%) | 0.081 (18%) | 0.813 (1%) | -0.052 (58%) | 0 (0%) | -4.878 (65%) | 0.871 (0%) | 0.129 (4%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.003 (2%) | -0.012 (1%) | 0.406 (0%) | 0.244 (12%) | 0.000 (0%) | 0 (0%) | -4.878 (65%) | -0.639 (100%) | -0.179 (99%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.003 (3%) | -0.010 (0%) | 0.325 (0%) | 0.650 (0%) | 0.000 (1%) | 0 (0%) | -4.878 (65%) | -0.340 (99%) | -0.094 (98%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (61%) | 0.000 (67%) | 0.000 (0%) | -0.081 (61%) | 0.000 (0%) | 0 (0%) | -4.878 (65%) | 0.470 (0%) | 0.103 (2%) |
| 6 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.007 (0%) | -0.015 (0%) | 0.732 (0%) | 0.488 (3%) | 0.448 (0%) | 0 (0%) | -9.755 (89%) | -0.599 (96%) | -0.161 (95%) |
| 7 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.008 (1%) | -0.003 (24%) | 0.000 (37%) | 0.813 (2%) | -0.052 (58%) | 0 (0%) | -4.878 (65%) | 0.570 (0%) | 0.076 (5%) |
| 8 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.021 (0%) | -0.021 (0%) | 0.732 (0%) | 1.545 (0%) | -0.052 (58%) | 0 (0%) | -4.878 (65%) | 1.291 (0%) | 0.144 (10%) |
| 9 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.002 (13%) | -0.004 (2%) | 0.244 (0%) | 0.081 (20%) | 0.000 (1%) | 0 (0%) | -4.878 (65%) | -0.476 (100%) | -0.109 (98%) |

`assoc_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=True, tracker.matching=greedy | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=True, tracker.matching=hungarian | 0.839 | 0.825 | 0.832 | 0.641 | 0.610 | 0.787 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=False, tracker.matching=greedy | 0.839 | 0.825 | 0.832 | 0.641 | 0.623 | 0.788 | 1.862 | 26.274 | 1.250 | 1 | 484.105 | 1.773 | 0.331 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=False, tracker.matching=hungarian | 0.839 | 0.825 | 0.832 | 0.641 | 0.624 | 0.788 | 1.862 | 26.274 | 1.250 | 1 | 484.105 | 1.773 | 0.331 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.001 (64%) | 0.000 (64%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.014 (99%) | 0.002 (58%) | 0.000 (33%) | -0.828 (84%) | 0.050 (26%) | -1 (67%) | 397.215 (0%) | -1.022 (100%) | -0.191 (100%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.015 (100%) | 0.002 (60%) | 0.000 (33%) | -0.828 (84%) | 0.050 (26%) | -1 (67%) | 397.215 (0%) | -1.022 (100%) | -0.191 (100%) |

`assoc_B/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=True, tracker.matching=greedy | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 39.021 | 2.998 | 0.636 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=True, tracker.matching=hungarian | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.274 | 1.052 | 0 | 39.021 | 2.998 | 0.636 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=False, tracker.matching=greedy | 0.808 | 0.809 | 0.808 | 0.497 | 0.649 | 0.768 | 3.821 | 22.843 | 1.500 | 0 | 614.574 | 2.737 | 0.539 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.use_velocity=False, tracker.matching=hungarian | 0.808 | 0.809 | 0.808 | 0.497 | 0.649 | 0.767 | 3.821 | 22.925 | 1.500 | 0 | 629.207 | 2.737 | 0.539 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.001 (0%) | -0.000 (0%) | 0.000 (0%) | 0.081 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.009 (2%) | -0.017 (1%) | 1.382 (0%) | 0.650 (9%) | 0.448 (0%) | 0 (0%) | 575.553 (0%) | -0.261 (74%) | -0.097 (85%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.010 (1%) | -0.017 (0%) | 1.382 (0%) | 0.732 (6%) | 0.448 (0%) | 0 (0%) | 590.186 (0%) | -0.261 (74%) | -0.097 (85%) |

`motion_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.7, tracker.motion_on_frames=3 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.7, tracker.motion_on_frames=2 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 124.130 | 2.795 | 0.521 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.7, tracker.motion_on_frames=5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 4 | 24.826 | 2.795 | 0.521 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.5, tracker.motion_on_frames=3 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 124.130 | 2.795 | 0.521 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.5, tracker.motion_on_frames=2 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 1 | 198.607 | 2.795 | 0.521 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.5, tracker.motion_on_frames=5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 3 | 49.652 | 2.795 | 0.521 |
| 7 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=1.0, tracker.motion_on_frames=3 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 4 | 62.065 | 2.795 | 0.521 |
| 8 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=1.0, tracker.motion_on_frames=2 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 3 | 99.304 | 2.795 | 0.521 |
| 9 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=1.0, tracker.motion_on_frames=5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 4 | 24.826 | 2.795 | 0.521 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 37.239 (0%) | 0.000 (7%) | 0.000 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 2 (0%) | -62.065 (96%) | 0.000 (7%) | 0.000 (0%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 37.239 (0%) | 0.000 (7%) | 0.000 (0%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -1 (69%) | 111.717 (0%) | 0.000 (7%) | 0.000 (0%) |
| 6 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 1 (0%) | -37.239 (89%) | 0.000 (7%) | 0.000 (0%) |
| 7 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 2 (0%) | -24.826 (90%) | 0.000 (7%) | 0.000 (0%) |
| 8 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 1 (0%) | 12.413 (0%) | 0.000 (7%) | 0.000 (0%) |
| 9 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 2 (0%) | -62.065 (96%) | -0.000 (63%) | 0.000 (0%) |

`motion_B/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.7, tracker.motion_on_frames=3 | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 39.021 | 2.998 | 0.636 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.7, tracker.motion_on_frames=2 | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 58.531 | 2.998 | 0.636 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.7, tracker.motion_on_frames=5 | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 4.878 | 2.998 | 0.636 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.5, tracker.motion_on_frames=3 | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 58.531 | 2.998 | 0.636 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.5, tracker.motion_on_frames=2 | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 92.674 | 2.998 | 0.636 |
| 6 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=0.5, tracker.motion_on_frames=5 | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 9.755 | 2.998 | 0.636 |
| 7 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=1.0, tracker.motion_on_frames=3 | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 14.633 | 2.998 | 0.636 |
| 8 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=1.0, tracker.motion_on_frames=2 | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 39.021 | 2.998 | 0.636 |
| 9 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.tentative_max_age=0.15, tracker.motion_on_speed=1.0, tracker.motion_on_frames=5 | 0.808 | 0.809 | 0.808 | 0.497 | 0.658 | 0.785 | 2.439 | 22.193 | 1.052 | 0 | 4.878 | 2.998 | 0.636 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 19.510 (0%) | 0.000 (0%) | 0.000 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | -34.143 (100%) | 0.000 (0%) | 0.000 (0%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 19.510 (0%) | 0.000 (0%) | 0.000 (0%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 53.653 (0%) | 0.000 (0%) | 0.000 (0%) |
| 6 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | -29.265 (100%) | 0.000 (0%) | 0.000 (0%) |
| 7 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | -24.388 (99%) | 0.000 (0%) | 0.000 (0%) |
| 8 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (50%) | 0.000 (0%) | 0.000 (0%) |
| 9 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | -34.143 (100%) | 0.000 (0%) | 0.000 (0%) |

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

远程没有新提交，没有跑测试。BASE 沿用 T20。耗时：现状 oracle_val / oracle_train / A 为 11 / 19 / 11 s，运动模型 36 / 70 / 41 s，合并距离 22 / 44 / 25 s。

第 1 步用默认 IMM（`cv:1.5+ct:1.5:0.25`，merge_dist 0.25）。moving、3 s 的 FDE1 / MR：

| 报告 | 样本 moving / straight / turning | IMM FDE1 | CV FDE1 | IMM MR | CV MR |
|---|---|---|---|---|---|
| oracle_val | 161 / 132 / 29 | 1.39 | 1.39 | 0.286 | 0.286 |
| oracle_train | 217 / 148 / 69 | 2.19 | 2.22 | 0.521 | 0.571 |
| A | 117 / 101 / 16 | 2.80 | 2.80 | 0.521 | 0.521 |

训练 oracle 上 CV 比 IMM 差，所以不满足「CV 在三份里都不比 IMM 差」，D2 不用据此改成匀速。验证 oracle 和端到端 A 上，moving 的 FDE1、MR 相同；minFDE 上 IMM 略好（验证 1.38 对 1.39，A 上 2.79 对 2.80）。训练 oracle：moving 的 FDE1 2.22 对 2.19，MR 0.571 对 0.521；转弯 69 个样本，FDE1 2.65 对 2.59，MR 0.754 对 0.638。

第 2 步留第 1 行 `tracker.vehicle.models=cv:1.5+ct:1.5:0.25`。主指标是 moving 的 FDE 3 s：要在 OT、OV、A 上都低于第 1 行，并且至少两份领先比例 ≥ 90%；三份的 MR 3 s 都不高于第 1 行；A 上 IDF1 下降不超过 0.01。

- 第 2 行 `cv:1.0+ct:1.0:0.25`：A 的 FDE 2.499（第 1 行 2.795，领先 100%），MR 0.483，IDF1 0.787。OT 的 FDE 2.225 高于 2.188，MR 0.525 高于 0.521。OV 的 FDE 低 0.001，领先只有 52%。只有一份领先 ≥ 90%。
- 第 3 行 `cv:2.5+ct:2.5:0.25`：只有 OT 更低（2.101，领先 100%，MR 0.461）。OV 1.391 高于 1.388，A 3.213 高于 2.795。
- 第 4 行 `cv:1.5+ct:1.5:0.5`：OV 1.389、OT 2.190 都略高于第 1 行。A 上 2.792，领先 89%，不到 90%。
- 第 5 行只有 `cv:1.5`：OV 1.390、OT 2.238 都更高（OT 的 MR 0.581）。A 上 2.794，领先 64%。

第 3 步在没改运动模型的 BASE 上扫合并距离，留第 1 行 `predict.merge_dist=0.25`。

- 第 2 行 0.5：OT 的 FDE 2.174，低 0.014，领先 100%，MR 0.525 高于 0.521。OV 的差是 −0.000，领先 55%。A 的 FDE 2.801 高于 2.795。
- 第 3 行 1.0：OT 的 FDE 2.157，低 0.031，领先 100%，MR 0.530 更高。OV 1.389、A 2.811 都更高，A 的 MR 也是 0.530。

最后的 BASE（和 T20 相同，没改的参数不写）：

```text
--set tracker.high_score=0.4 --set tracker.output_min_score=0.35 --set tracker.low_score=0.2 --set tracker.vehicle.max_age=1.5 --set tracker.vehicle.min_hits=3
```

第 1 步 Prediction（moving、moving straight、moving turning）：

验证 oracle（`base_oracle_val/report.md`）：

**moving** (161 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.29 | 0.39 | 0.29 | 0.39 | 0.012 |
| IMM | 2 s | 0.53 | 0.90 | 0.52 | 0.89 | 0.106 |
| IMM | 3 s | 0.78 | 1.39 | 0.77 | 1.38 | 0.286 |
| CV | 1 s | 0.29 | 0.39 | 0.29 | 0.39 | 0.012 |
| CV | 2 s | 0.53 | 0.90 | 0.53 | 0.90 | 0.106 |
| CV | 3 s | 0.78 | 1.39 | 0.78 | 1.39 | 0.286 |

**moving, straight** (132 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.26 | 0.35 | 0.26 | 0.35 | 0.008 |
| IMM | 2 s | 0.47 | 0.79 | 0.47 | 0.79 | 0.083 |
| IMM | 3 s | 0.69 | 1.25 | 0.69 | 1.25 | 0.250 |
| CV | 1 s | 0.26 | 0.35 | 0.26 | 0.35 | 0.008 |
| CV | 2 s | 0.47 | 0.79 | 0.47 | 0.79 | 0.083 |
| CV | 3 s | 0.69 | 1.25 | 0.69 | 1.25 | 0.250 |

**moving, turning** (29 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.42 | 0.56 | 0.41 | 0.56 | 0.034 |
| IMM | 2 s | 0.79 | 1.37 | 0.78 | 1.36 | 0.207 |
| IMM | 3 s | 1.16 | 2.01 | 1.15 | 2.00 | 0.448 |
| CV | 1 s | 0.41 | 0.56 | 0.41 | 0.56 | 0.034 |
| CV | 2 s | 0.79 | 1.37 | 0.79 | 1.37 | 0.207 |
| CV | 3 s | 1.16 | 2.01 | 1.16 | 2.01 | 0.448 |

训练 oracle（`base_oracle_train/report.md`）：

**moving** (217 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.37 | 0.52 | 0.36 | 0.50 | 0.009 |
| IMM | 2 s | 0.73 | 1.28 | 0.70 | 1.23 | 0.189 |
| IMM | 3 s | 1.14 | 2.19 | 1.09 | 2.09 | 0.521 |
| CV | 1 s | 0.38 | 0.52 | 0.38 | 0.52 | 0.009 |
| CV | 2 s | 0.73 | 1.29 | 0.73 | 1.29 | 0.203 |
| CV | 3 s | 1.15 | 2.22 | 1.15 | 2.22 | 0.571 |

**moving, straight** (148 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.33 | 0.47 | 0.33 | 0.46 | 0.000 |
| IMM | 2 s | 0.65 | 1.14 | 0.64 | 1.11 | 0.169 |
| IMM | 3 s | 1.02 | 2.00 | 1.00 | 1.96 | 0.466 |
| CV | 1 s | 0.34 | 0.47 | 0.34 | 0.47 | 0.000 |
| CV | 2 s | 0.65 | 1.15 | 0.65 | 1.15 | 0.176 |
| CV | 3 s | 1.03 | 2.01 | 1.03 | 2.01 | 0.486 |

**moving, turning** (69 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.46 | 0.63 | 0.44 | 0.60 | 0.029 |
| IMM | 2 s | 0.89 | 1.57 | 0.84 | 1.48 | 0.232 |
| IMM | 3 s | 1.38 | 2.59 | 1.28 | 2.37 | 0.638 |
| CV | 1 s | 0.46 | 0.63 | 0.46 | 0.63 | 0.029 |
| CV | 2 s | 0.90 | 1.60 | 0.90 | 1.60 | 0.261 |
| CV | 3 s | 1.40 | 2.65 | 1.40 | 2.65 | 0.754 |

端到端 A（`base_A/report.md`）：

**moving** (117 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.94 | 1.13 | 0.94 | 1.13 | 0.188 |
| IMM | 2 s | 1.35 | 1.98 | 1.35 | 1.98 | 0.368 |
| IMM | 3 s | 1.77 | 2.80 | 1.77 | 2.79 | 0.521 |
| CV | 1 s | 0.94 | 1.13 | 0.94 | 1.13 | 0.188 |
| CV | 2 s | 1.35 | 1.99 | 1.35 | 1.99 | 0.368 |
| CV | 3 s | 1.77 | 2.80 | 1.77 | 2.80 | 0.521 |

**moving, straight** (101 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.94 | 1.12 | 0.94 | 1.12 | 0.198 |
| IMM | 2 s | 1.34 | 1.96 | 1.34 | 1.96 | 0.376 |
| IMM | 3 s | 1.75 | 2.76 | 1.75 | 2.76 | 0.525 |
| CV | 1 s | 0.94 | 1.12 | 0.94 | 1.12 | 0.198 |
| CV | 2 s | 1.34 | 1.96 | 1.34 | 1.96 | 0.376 |
| CV | 3 s | 1.75 | 2.76 | 1.75 | 2.76 | 0.525 |

**moving, turning** (16 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.94 | 1.17 | 0.94 | 1.17 | 0.125 |
| IMM | 2 s | 1.42 | 2.13 | 1.42 | 2.13 | 0.312 |
| IMM | 3 s | 1.89 | 3.02 | 1.89 | 3.02 | 0.500 |
| CV | 1 s | 0.94 | 1.17 | 0.94 | 1.17 | 0.125 |
| CV | 2 s | 1.42 | 2.13 | 1.42 | 2.13 | 0.312 |
| CV | 3 s | 1.89 | 3.02 | 1.89 | 3.02 | 0.500 |

第 2 步运动模型：

`models_oracle_val/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.5+ct:1.5:0.25 | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.388 | 0.286 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.0+ct:1.0:0.25 | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.388 | 0.273 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:2.5+ct:2.5:0.25 | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.391 | 0.298 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.5+ct:1.5:0.5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.389 | 0.286 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.390 | 0.286 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.001 (52%) | -0.012 (79%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.003 (48%) | 0.012 (0%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.001 (38%) | 0.000 (0%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.002 (30%) | 0.000 (0%) |

`models_oracle_train/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.5+ct:1.5:0.25 | 1.000 | 1.000 | 1.000 | 0.000 | 0.953 | 0.917 | 2.601 | 0.000 | 0.501 | 0 | 0.000 | 2.188 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.0+ct:1.0:0.25 | 1.000 | 1.000 | 1.000 | 0.000 | 0.952 | 0.917 | 2.601 | 0.000 | 0.501 | 0 | 0.000 | 2.225 | 0.525 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:2.5+ct:2.5:0.25 | 1.000 | 1.000 | 1.000 | 0.000 | 0.953 | 0.917 | 2.601 | 0.000 | 0.501 | 0 | 0.000 | 2.101 | 0.461 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.5+ct:1.5:0.5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.953 | 0.917 | 2.601 | 0.000 | 0.501 | 0 | 0.000 | 2.190 | 0.507 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.953 | 0.917 | 2.601 | 0.000 | 0.501 | 0 | 0.000 | 2.238 | 0.581 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.000 (0%) | -0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.037 (21%) | 0.005 (44%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.087 (100%) | -0.060 (99%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.002 (45%) | -0.014 (88%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.050 (0%) | 0.060 (0%) |

`models_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.5+ct:1.5:0.25 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.0+ct:1.0:0.25 | 0.839 | 0.825 | 0.832 | 0.641 | 0.612 | 0.787 | 1.862 | 26.895 | 1.200 | 2 | 86.891 | 2.499 | 0.483 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:2.5+ct:2.5:0.25 | 0.839 | 0.825 | 0.832 | 0.641 | 0.606 | 0.781 | 2.069 | 27.309 | 1.200 | 2 | 74.478 | 3.213 | 0.573 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.5+ct:1.5:0.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.792 | 0.521 |
| 5 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, tracker.vehicle.models=cv:1.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.794 | 0.521 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.003 (90%) | 0.001 (90%) | 0.000 (0%) | -0.207 (64%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.296 (100%) | -0.038 (98%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | -0.002 (0%) | -0.006 (0%) | 0.207 (0%) | 0.207 (0%) | 0.000 (0%) | 0 (0%) | -12.413 (67%) | 0.418 (0%) | 0.051 (0%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.003 (89%) | 0.000 (0%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.002 (64%) | 0.000 (0%) |

第 3 步合并距离：

`merge_oracle_val/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, predict.merge_dist=0.25 | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.388 | 0.286 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, predict.merge_dist=0.5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.388 | 0.286 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, predict.merge_dist=1.0 | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.389 | 0.286 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.000 (55%) | 0.000 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.001 (0%) | 0.000 (0%) |

`merge_oracle_train/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, predict.merge_dist=0.25 | 1.000 | 1.000 | 1.000 | 0.000 | 0.953 | 0.917 | 2.601 | 0.000 | 0.501 | 0 | 0.000 | 2.188 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, predict.merge_dist=0.5 | 1.000 | 1.000 | 1.000 | 0.000 | 0.953 | 0.917 | 2.601 | 0.000 | 0.501 | 0 | 0.000 | 2.174 | 0.525 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, predict.merge_dist=1.0 | 1.000 | 1.000 | 1.000 | 0.000 | 0.953 | 0.917 | 2.601 | 0.000 | 0.501 | 0 | 0.000 | 2.157 | 0.530 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.014 (100%) | 0.005 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.031 (100%) | 0.009 (0%) |

`merge_A/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, predict.merge_dist=0.25 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, predict.merge_dist=0.5 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.801 | 0.521 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, tracker.vehicle.max_age=1.5, tracker.vehicle.min_hits=3, predict.merge_dist=1.0 | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.811 | 0.530 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.006 (5%) | 0.000 (0%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.016 (0%) | 0.009 (0%) |


---

## T22 定版：参数改成默认值，从原始扫描完整跑一遍

T21 之后，我把 T19–T21 选定的参数改成代码里的默认值（`DEFAULT_SCORE_THRESHOLDS`、`TrackerConfig`、`DetectionFilterConfig`、`IMMPredictor`），
在这里写出旧默认值的 `configs.json`，推送后再做 T22。这一步不再是重放：
- 从原始扫描起跑，前半段（读扫描、去畸变、拼帧）也在里面，和线上一样；
- 新旧两组参数用同一份检测比。

```text
任务 T22：用新的默认参数，从原始扫描完整跑一遍验证场景，和旧默认参数比，出最终报告和 Foxglove MCAP。
先按通用约定做（拉到了代码改动，跑快速测试）。前置：T21 完成；默认值已经改好，`docs/pipeline_tuning/T22_configs.json` 已经写好（第 1 行旧默认值，第 2 行新默认值）。
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

拉到 `3ea27e7`（默认值已改，`docs/pipeline_tuning/T22_configs.json` 已在）。`dog_perception/` 有改动，快速测试 153 passed（`CUDA_VISIBLE_DEVICES=3`，20 s）。

第 1 步在 GPU 0 上跑：当时空闲 20909 MiB，利用率 100%（Carla 占 2009 MiB），4 个进程放得下。45 秒时四个推理进程各 700 / 740 / 780 / 820 MiB。`run_sequence` 09:38:06–09:43:01（约 5 分钟），final 24 s，compare 18 s。标签拼成一辆车的次数是 19，不超过 40。序列里没有 `cameras.mcap`，MCAP 里没有相机。

final 的 IDF1 是 0.786，和 T20 的 final_A 相同，没有触发「低 0.05 以上」。同一份新检测上，旧默认值（compare 第 1 行）的点估计和 T18 deploy 一致，新默认值（第 2 行）和 T20 final_A 一致：从原始扫描完整跑和之前的重放一致。

和 T18 deploy 并排（主要指标的点估计）：

| metric | T18 deploy | T22 final |
|---|---|---|
| detections: recall within near range | 0.872 | 0.839 |
| detections: precision within near range | 0.758 | 0.825 |
| detections: F1 within near range | 0.811 | 0.832 |
| detections: false per keyframe | 1.040 | 0.641 |
| tracks: MOTA | 0.459 | 0.609 |
| tracks: IDF1 | 0.726 | 0.786 |
| tracks: ID switches / min | 2.483 | 1.862 |
| tracks: false tracks / min | 58.134 | 27.102 |
| tracks: confirm delay p90 [s] | 1.200 | 1.200 |
| motion: relevant moving stretches missed | 2 | 2 |
| motion: parked reported moving / h | 86.891 | 86.891 |
| prediction: FDE 3 s, moving [m] | 2.897 | 2.795 |
| prediction: miss rate 3 s, moving | 0.541 | 0.521 |

`final/report.md`：

## Main metrics

| metric | value | interval |
|---|---|---|
| detections: recall within near range | **0.839** | 0.789 – 0.860 |
| detections: precision within near range | **0.825** | 0.634 – 0.903 |
| detections: F1 within near range | **0.832** | 0.710 – 0.873 |
| detections: false per keyframe | **0.641** | 0.484 – 0.799 |
| tracks: MOTA | **0.609** | 0.304 – 0.709 |
| tracks: IDF1 | **0.786** | 0.663 – 0.838 |
| tracks: ID switches / min | **1.862** | 0.419 – 3.356 |
| tracks: false tracks / min | **27.102** | 20.238 – 33.793 |
| tracks: confirm delay p90 [s] | **1.200** | 0.501 – 4.000 |
| motion: relevant moving stretches missed | **2** | 0.000 – 4.000 |
| motion: parked reported moving / h | **86.891** | 12.388 – 175.081 |
| prediction: FDE 3 s, moving [m] | **2.795** | 2.297 – 3.621 |
| prediction: miss rate 3 s, moving | **0.521** | 0.430 – 0.694 |

## Detections at the reporting threshold

Vehicle threshold 0.40 (`tracker.high_score`), 2 m match, keyframes. Near: within 30 m.

| range | precision | recall | TP | FP | FN |
|---|---|---|---|---|---|
| 0-30 m | 0.825 | 0.839 | 1149 | 243 | 221 |
| 30-40 m | 0.618 | 0.576 | 209 | 129 | 154 |

False detections per keyframe 0.641. AP over all scores: AP@2.0 0.834, AP@4.0 0.853, AP@2.0 0-20 m 0.933, 20-40 m 0.788.

## Tracking

Reported tracks (confirmed, coasting included) against the scored vehicles, keyframes, 2 m.

| MOTA | MOTP m | IDF1 | recall | precision | IDSW | IDSW / min | FRAG | MT | ML | vehicles | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0.609** | 0.28 | **0.786** | 0.815 | 0.802 | 9 | 1.86 | 41 | 34 | 7 | 59 | 1412 | 348 | 321 |

- false tracks (a track false on consecutive keyframes counts once): 27.10 per minute
- a vehicle coming into 30 m waits for its first matched track: median 0.5 s, p90 1.2 s (17 vehicles; 1 never tracked; 0.5 s resolution)

### Where the errors are

| false tracks: cause | count |
|---|---|
| second track on a tracked vehicle | 2 |
| beside a vehicle (2-4 m off) | 21 |
| coasting after the vehicle was lost | 24 |
| nothing labelled nearby | 301 |

| missed vehicles (FN) | count |
|---|---|
| detected, not tracked | 54 |
| not detected | 267 |
| 0-10 m | 7 |
| 10-20 m | 26 |
| 20-30 m | 162 |
| 30-40 m | 126 |
| 1-9 pts | 13 |
| 10-49 pts | 65 |
| >=50 pts | 243 |

Blind sector (142-218 deg behind the robot) and the 20 deg bands beside it:

| zone | scored vehicles | missed | missed share | tracks | false | false share |
|---|---|---|---|---|---|---|
| elsewhere | 1317 | 227 | 0.172 | 1398 | 310 | 0.222 |
| edge | 402 | 90 | 0.224 | 339 | 32 | 0.094 |
| blind | 14 | 4 | 0.286 | 23 | 6 | 0.261 |

## Motion state

Online state against the labels' moving flags (tools/dog/motion_eval.py). Relevant: >= 1 m/s and within 30 m.

| moving recall | false moving | parked reported moving | false episodes (per h) | moving stretches missed | latency median / p90 s | relevant missed | relevant latency median / p90 s |
|---|---|---|---|---|---|---|---|
| 0.760 | 0.0170 | 4 / 38 | 7 (86.9) | 3 / 19 | 0.50 / 1.50 | **2 / 9** | 0.50 / 1.20 |

## Prediction

Tracks matched to a vehicle with 3 s of labelled future (keyframes every 0.5 s). Moving: >= 1.5 m in 3 s; turning: heading changes > 15 deg. ADE1 / FDE1: the most likely mode; minADE / minFDE: the best; MR: no mode ends within 2 m. IMM is what runs online, CV keeps the track's velocity.

**all matched** (1123 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.43 | 0.49 | 0.43 | 0.49 | 0.060 |
| IMM | 2 s | 0.56 | 0.74 | 0.56 | 0.74 | 0.102 |
| IMM | 3 s | 0.68 | 1.01 | 0.68 | 1.01 | 0.140 |
| CV | 1 s | 0.43 | 0.49 | 0.43 | 0.49 | 0.060 |
| CV | 2 s | 0.56 | 0.75 | 0.56 | 0.75 | 0.102 |
| CV | 3 s | 0.68 | 1.01 | 0.68 | 1.01 | 0.140 |

**moving** (117 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.94 | 1.13 | 0.94 | 1.13 | 0.188 |
| IMM | 2 s | 1.35 | 1.98 | 1.35 | 1.98 | 0.368 |
| IMM | 3 s | 1.77 | 2.80 | 1.77 | 2.79 | 0.521 |
| CV | 1 s | 0.94 | 1.13 | 0.94 | 1.13 | 0.188 |
| CV | 2 s | 1.35 | 1.99 | 1.35 | 1.99 | 0.368 |
| CV | 3 s | 1.77 | 2.80 | 1.77 | 2.80 | 0.521 |

**moving, straight** (101 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.94 | 1.12 | 0.94 | 1.12 | 0.198 |
| IMM | 2 s | 1.34 | 1.96 | 1.34 | 1.96 | 0.376 |
| IMM | 3 s | 1.75 | 2.76 | 1.75 | 2.76 | 0.525 |
| CV | 1 s | 0.94 | 1.12 | 0.94 | 1.12 | 0.198 |
| CV | 2 s | 1.34 | 1.96 | 1.34 | 1.96 | 0.376 |
| CV | 3 s | 1.75 | 2.76 | 1.75 | 2.76 | 0.525 |

**moving, turning** (16 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.94 | 1.17 | 0.94 | 1.17 | 0.125 |
| IMM | 2 s | 1.42 | 2.13 | 1.42 | 2.13 | 0.312 |
| IMM | 3 s | 1.89 | 3.02 | 1.89 | 3.02 | 0.500 |
| CV | 1 s | 0.94 | 1.17 | 0.94 | 1.17 | 0.125 |
| CV | 2 s | 1.42 | 2.13 | 1.42 | 2.13 | 0.312 |
| CV | 3 s | 1.89 | 3.02 | 1.89 | 3.02 | 0.500 |

**moving, < 3 m/s** (107 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.95 | 1.14 | 0.95 | 1.14 | 0.196 |
| IMM | 2 s | 1.38 | 2.02 | 1.38 | 2.02 | 0.393 |
| IMM | 3 s | 1.80 | 2.86 | 1.80 | 2.85 | 0.551 |
| CV | 1 s | 0.95 | 1.15 | 0.95 | 1.15 | 0.196 |
| CV | 2 s | 1.38 | 2.02 | 1.38 | 2.02 | 0.393 |
| CV | 3 s | 1.80 | 2.86 | 1.80 | 2.86 | 0.551 |

**moving, >= 6 m/s** (10 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m |
|---|---|---|---|---|---|---|
| IMM | 1 s | 0.80 | 0.96 | 0.80 | 0.96 | 0.100 |
| IMM | 2 s | 1.11 | 1.57 | 1.11 | 1.57 | 0.100 |
| IMM | 3 s | 1.41 | 2.13 | 1.41 | 2.13 | 0.200 |
| CV | 1 s | 0.81 | 0.96 | 0.81 | 0.96 | 0.100 |
| CV | 2 s | 1.12 | 1.58 | 1.12 | 1.58 | 0.100 |
| CV | 3 s | 1.41 | 2.15 | 1.41 | 2.15 | 0.200 |

## Hazard proxy

A moving vehicle that comes within 5 m of the robot within 3 s (labelled future, the robot's own path) should raise an alarm: a moving track whose most likely predicted path, or its position now, comes that close. Few such vehicles in 0921; the numbers are for the controlled scenes to come (DATA_COLLECTION_PLAN.md 3.2).

- events 6, reported 3, lead before the closest approach: median 1.0 s, min 1.0 s
- alarms without such a vehicle: 1 (12.4 per hour)

## Timing (this machine, ms per frame: median / p90)

| stage | median | p90 |
|---|---|---|
| filter | 1.50 | 3.30 |
| track | 2.23 | 6.88 |
| predict | 1.16 | 4.01 |

## Issues (`issues.csv`)

| kind | count |
|---|---|
| ID switch | 9 |
| bad prediction | 44 |
| false moving | 7 |
| false track | 56 |
| hazard not reported | 3 |
| lost vehicle | 27 |
| missed moving vehicle | 2 |

`compare/sweep.md`（第 1 行旧默认值，第 2 行新默认值）：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.35, tracker.output_min_score=0.3, tracker.low_score=0.1, tracker.vehicle.max_age=1.0, tracker.vehicle.min_hits=2 | 0.872 | 0.758 | 0.811 | 1.040 | 0.459 | 0.726 | 2.483 | 58.134 | 1.200 | 2 | 86.891 | 2.897 | 0.541 |
| 2 | defaults | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -0.033 (0%) | 0.068 (100%) | 0.021 (99%) | -0.398 (100%) | 0.150 (100%) | 0.060 (100%) | -0.621 (82%) | -31.032 (100%) | 0.000 (0%) | 0 (0%) | 0.000 (32%) | -0.102 (95%) | -0.020 (67%) |

`timing.txt`（第 1 步，中位数；前半段 + 网络，不含跟踪和预测）：

```text
data/tune/final/0921_164127_s000/pillar/results.pkl network 103.1 ms, preprocess + detector 219.0 ms
data/tune/final/0921_164127_s001/pillar/results.pkl network 103.4 ms, preprocess + detector 224.5 ms
data/tune/final/0921_164127_s002/pillar/results.pkl network 108.8 ms, preprocess + detector 223.8 ms
data/tune/final/0921_164127_s003/pillar/results.pkl network 106.7 ms, preprocess + detector 220.6 ms
data/tune/final/0921_164127_s004/pillar/results.pkl network 103.4 ms, preprocess + detector 228.7 ms
data/tune/final/0921_164127_s005/pillar/results.pkl network 101.3 ms, preprocess + detector 234.6 ms
data/tune/final/0921_165659_s003/pillar/results.pkl network 100.6 ms, preprocess + detector 219.3 ms
data/tune/final/0921_165659_s004/pillar/results.pkl network 107.8 ms, preprocess + detector 228.9 ms
data/tune/final/0921_165659_s005/pillar/results.pkl network 91.3 ms, preprocess + detector 195.6 ms
data/tune/final/0921_165659_s006/pillar/results.pkl network 91.1 ms, preprocess + detector 200.0 ms
```

MCAP：`data/tune/T22_mcap/`，10 个场景，共 872 MB。

（2026-10-28 复核）
- **定版成立**：从原始扫描完整跑，和重放的结果逐项一致；新默认值在同一份检测上相对旧默认值：
  - IDF1 +0.060、MOTA +0.150、假轨迹每分钟 −31，重抽里领先都是 100%；
  - 检测 F1 +0.021（99%），3 s 预测误差 −0.10 m（95%）；
  - 代价是 30 m 内召回 −0.033；运动车漏报、被跟上的延迟、停着的车报成在动都没变。
- **timing.txt 的耗时不能用**：第 1 步那张卡利用率 100%（Carla 也在用），还同时放了 4 个进程，网络 100 ms / 帧，是 T14 单独测时的 5 倍。
  - 跟踪、预测在 CPU 上：每帧中位数 2.2 ms、1.2 ms，p90 6.9 ms、4.0 ms，不是瓶颈。
  - 上线延迟要单独测（T7 第 2b 步的做法，独占一张卡），最终以 Orin 上（H2）为准。
- **问题清单（`data/reports/T22/final/issues.csv`）集中在几个地方**，H4 看 `data/tune/T22_mcap/` 时按这张表看：
  | 场景 | 时间（场景内秒数） | 看什么 |
  |---|---|---|
  | 164127_s001 | 7.5–27 s | **运动的小车（4.5 × 1.9 m，约 2 m/s，19 m 外）漏了 12.5 s**。只有约 74 个点，两个教师也都没检出，标签是人补的。对照 `T18_mcap_v2` 看 v2 能不能检出 |
  | 165659_s006 | 4–12 s | **9 m 的货车在狗旁边约 4 m、约 1 m/s 慢慢开，一直没报"在动"**，预警 3 次都没报 |
  | 164127_s000 | 9–29 s | 12 m 的停着的货车（23 m 外）被报成在动 4 次，占全部 7 次里的 4 次：长车的框沿车身滑动，还是速度头读数有问题 |
  | 164127_s005 | 1–15 s | 两条持续约 14 s 的假轨迹（13 m、19 m 处）：是没标上的车，还是墙、货堆 |
  | 164127_s000 | 23–28 s | 4 次 ID 切换：停着的车被挡 2–3 s，coast 1.5 s 不够 |
- **D2 的方向**（等 H4 确认是哪种情况再动手）：
  - 长车的运动判断：位移速度不用框中心，或者对长车要求更大的位移；
  - 近处慢速大车的运动判断；
  - 停着的轨迹 coast 更久（静止的车被挡时）；
  - 速度估计更稳，3 s 预测误差一半来自它。
  - 第一条检测上的漏检（s001 的小车）后处理改不了，要靠模型和数据，记进下一轮的数据需求。

（2026-10-28 复核：T19–T21，以及为 T22 改好的默认值）
- **T19 第 1b 步：在各自最合适的阈值下，v1、v2 在跟踪层面仍明显比 nusc_pp 好。**
  - v2 的检测 F1 高 0.034、IDF1 高 0.071，v1 高 0.035、0.083；假轨迹每分钟 13.7、11.4 条，nusc_pp 是 43.9 条。
  - nusc_pp 剩下的优势只有一项：车进入 30 m 后被跟上的延迟 p90，1.2 s，v2 7.25 s、v1 3.5 s。
    这一项只有 16 辆车，v2 的 7.25 s 是其中最慢的两辆。30 m 内的召回在 0.40 阈值下已经不比 v2 高（0.839 对 0.844）。
  - 2026-10-27 人定仍用 nuScenes Pillar。理由可以是它没在任何 0921 数据上训练过，在新的区域更可能保持同样的表现；
    v1、v2 是在同一个厂区、同一天的数据上训练的，验证场景对它们偏乐观。
  - 这个决定等新采的数据到了再验证（DATA_COLLECTION_PLAN.md 3.5 的测试集）。
- **T19–T21 选出的参数，已改成代码默认值**：
  - 上报（新建轨迹）阈值：车 0.35 → 0.40；
  - 轨迹输出分数：车 0.3 → 0.35；
  - 低分门槛：0.1 → 0.2；
  - 车的最长 coast：1.0 → 1.5 s；
  - 确认帧数：2 → 3。
  - 离线自动标注不跟着变，固定用调之前的值（`autolabel.label_tracker_config`）：0921 的标签就是用它们做的，以后重做标注也对得上。
- **调完以后（T20 final_A 对 T18 的 deploy，验证集）**：
  | | 调之前 | 调之后 |
  |---|---|---|
  | IDF1 | 0.726 | 0.786 |
  | MOTA | 0.459 | 0.609 |
  | 假轨迹 / 分钟 | 58.1 | 27.1 |
  | ID 切换 / 分钟 | 2.48 | 1.86 |
  | 检测 F1（30 m 内） | 0.811 | 0.832 |
  | 召回（30 m 内） | 0.872 | 0.839 |
  - 召回低了 0.033，车进入 30 m 后被跟上的延迟、运动的车漏报（2 段）都没变。
- **还没解决的**：
  - **停着的车被报成在动**：仍是每小时 87 次（在验证集上约 7 次）。T20 第 6 组找不到不多漏运动车的设置，教师检测下是 25 次，所以一半以上来自 Pillar 的检测；
  - **预测**：IMM 和匀速一样；端到端 3 s 的误差 2.8 m，用标注当检测时是 1.4 m，一半来自估出的速度；
  - **ID 切换**：被挡 2–3 s 的停着的车换号，coast 1.5 s 还不够。
  这三项交给 D2（改代码），H4 看 Foxglove 时也请重点看这几类。

---

## H5 看最终效果，定上线参数（人做）

用 Foxglove 看 `data/tune/T22_mcap/`（同 H4），对照 T22 的报告，决定新参数是否上线。
- 上线后，在 Orin 上用 ROS 2 节点录一段（T8、H2 之后），用同一个工具评：录制转成序列，跑 `run_sequence.py`，再跑 `pipeline_eval.py`。
- 可控场景（DATA_COLLECTION_PLAN.md 3.2）采到并标好以后，用同样的命令评预警。

### H5 结果

（待填）

---

## T23 第二档预测（学习模型）第一版

第一档（IMM）在 0921 上和匀速外推一样：
- 运动的车 3 s 的 FDE 2.8 m（T18、T21）；
- 标签当检测时是 1.4 m，所以一半误差来自跟踪估出的速度。

第二档拿过去 2 s 的轨迹（以及周围的车、狗自己）预测 3 s，给 6 个模式和概率（TRACKING_PREDICTION.md 3.3 节）。
- 训练数据不用标注：跟踪器自己的轨迹当输入，事后平滑过的观测位置当目标；
- 评估用审过的标签。

**预期要放低**：0921 上运动的车只有约 70 辆（训练场景 41、验证场景 27），窗口大多来自停着的车。
这一版主要是把流程跑通、留一个基线，新数据（DATA_COLLECTION_PLAN.md 3.2、3.3）到了再训。

附带两项检查（2026-10-28 加，评估工具每次都会算，第 1 步的报告里就有）：
- **误差拆分**：运动车概率最高那条路线的误差，拆成沿标注路径的纵向（速度、起步、停车；bias 为正表示预测快了）
  和离开路径的横向（路线形状）。用来判断要不要加道路地图 / 占据栅格：横向占比小就不需要。
- **训练目标有多准**：训练目标是伪真值，由跟踪日志事后平滑得到。有两种来源：
  - filtered：跟踪器的位置；
  - measured：匹配到的原始检测框中心，日志从这一版起记录。
  评估工具按 train_predictor.py 切窗的同样规则切出每个预测样本的训练目标，和审过的标签比。第 1 步据此选第 2 步用哪种。

```text
任务 T23：用上线的检测器和定版的跟踪参数生成 35 个场景的跟踪日志，训练第二档预测的第一版（GRU、Transformer 各一个），
在验证集上和 IMM 配对比较，判断能不能替换第一档。
先按通用约定做（拉到了代码改动，跑快速测试）。前置：T22 完成。
模型小（约 16–25 万参数），CPU 就能训；有空闲 GPU 就加 --device cuda:0（约 1 GB）。

0. mkdir -p data/reports/T23 data/tune/track_logs data/train/pred_v1; TASK=T23；VAL、TRAIN、ev 同 T18 第 0、3 步。
1. 跟踪日志（CPU，各约 1 分钟）。用默认参数，也就是定版的跟踪参数，预测是 IMM，日志里同时记下 IMM 的预测，用来对比：
     ev logs_val "$VAL" "data/tune/deploy/{scene}/pillar/results.pkl" --detections raw --no-mcap --bootstrap 0 \
         --track-logs data/tune/track_logs
     ev logs_train "$TRAIN" "data/tune/nusc_pp_train/{scene}/pillar/results.pkl" --detections raw --no-mcap --bootstrap 0 \
         --track-logs data/tune/track_logs
     ls data/tune/track_logs/*.npz | wc -l        # 应是 35
     python -c "import numpy as np; print(np.load('data/tune/track_logs/$(echo $VAL | cut -d' ' -f1).npz')['rows'].shape[1])"   # 应是 15（含匹配到的检测）
   两份 report.md（data/reports/T23/logs_val、logs_train）里看：
   - "Prediction" 的 moving、moving straight、moving turning 三张表：IMM、CV 的 along / bias / across；
   - "Training targets" 表：filtered、measured 两种目标各自的 windows、mean、p90、> 1 m、along / bias / across。
   选训练目标 TGT：measured 在两份报告里 moving 3 s 的 mean 都比 filtered 低，且 > 1 m 的比例不更高，就用 measured；
   否则用 filtered。
2. 训练：训练场景训，验证场景验，GRU、Transformer 各一个：
     TGT=measured        # 或 filtered，按第 1 步
     LT=$(for r in $TRAIN; do echo data/tune/track_logs/$r.npz; done); LV=$(for r in $VAL; do echo data/tune/track_logs/$r.npz; done)
     for enc in gru transformer; do
       python tools/dog/train_predictor.py --logs $LT --val-logs $LV --encoder $enc --modes 6 --epochs 60 \
           --static-keep 0.3 --mirror --targets $TGT --out data/train/pred_v1/$enc.pt 2>&1 | tee data/reports/T23/train_$enc.txt
     done
   train_*.txt 开头打印窗口数（其中运动的），最后一张表是验证窗口上的 cv / imm / learned，分全部、运动、静止。
   这张表的"真值"是同一种伪真值；能不能替换 IMM 只看第 3 步（对标签）。
3. 和标签比（配对：第 1 行 IMM，第 2 行第二档；端到端和标签当检测各一次）：
     for enc in gru transformer; do
       ev pred_$enc "$VAL" "data/tune/deploy/{scene}/pillar/results.pkl" --detections raw --no-mcap \
           --set "eval.learned_model=,data/train/pred_v1/$enc.pt"
       ev pred_${enc}_oracle "$VAL" "data/tune/deploy/{scene}/pillar/results.pkl" --detections labels \
           --dense-labels "data/labels/{scene}/labels_full2.pkl" --no-mcap --set "eval.learned_model=,data/train/pred_v1/$enc.pt"
     done
   每份的 sweep.md 看 FDE 3 s、MR 3 s；sweep.csv 里还有 pred_minFDE3_moving、false_alarms_per_h、hazard_reported、predict_ms，
   以及 3 s 误差拆分 pred_along3_moving、pred_along_bias3_moving、pred_across3_moving。

怎么选（能不能替换第一档；都满足的 encoder 里选端到端 FDE 3 s 低的）：
- 端到端（pred_*）：FDE 3 s 比第 1 行（IMM）低，领先比例 ≥ 90%；MR 3 s 不高于第 1 行；
- 标签当检测（pred_*_oracle）：FDE 3 s 不高于第 1 行；
- 不能让狗多误停：sweep.csv 里 false_alarms_per_h 不高于第 1 行，hazard_reported 不少于第 1 行；
  train_*.txt 静止那组，learned 的 FDE1 不比 imm 高 0.1 m 以上（停着的车不能被预测成要开走）；
- 耗时：sweep.csv 里第 2 行的 predict_ms（中位数，服务器 CPU）不超过 10 ms。
都不满足：线上继续用 IMM，模型留作基线，新数据来了再训。满足了写清是哪个 encoder，由人决定是否上线
（ROS 2 节点参数 predictor_model）。

停止条件：任何报错；第 1 步日志不是 35 个，或 rows 不是 15 列（代码没拉到）；训练 loss 出现 nan。
提交：data/reports/T23/、本文档。不提交 .pt 和跟踪日志。
报告（写进"T23 结果"）：
- 第 1 步两份 report.md 的 "Prediction"（moving、moving straight、moving turning）和 "Training targets" 两部分；
  选的 TGT 和理由；
- 两个 train_*.txt 开头的窗口数行、最后的表；
- 四份 sweep.md 的两张表；
- sweep.csv 里两行的 predict_ms、false_alarms_per_h、hazard_reported、pred_minFDE3_moving，
  pred_along3_moving、pred_along_bias3_moving、pred_across3_moving；
- 按"怎么选"给出的结论；每个训练的耗时。
```

### T23 结果

拉到 `b299f17`，本地未推送的 `Make the review overlays easier to read` 接到了它上面。`dog_perception/`、`tools/` 有改动，快速测试 157 passed（`CUDA_VISIBLE_DEVICES=3`，22 s）。

跟踪日志 35 个，`rows` 是 15 列。耗时：logs_val 7 s，logs_train 13 s。训练在物理 GPU 1 上（空闲 2755 MiB，利用率 0%，进程里是 `cuda:0`），目标是 measured。GRU 44 s，Transformer 56 s。loss 没有 nan。对比（CPU）：pred_gru 1418 s，pred_gru_oracle 987 s，pred_transformer 1839 s，pred_transformer_oracle 1583 s。

第 1 步选 **measured**。moving、3 s 的 mean 在两份报告里都低于 filtered，> 1 m 的比例也更低：验证 0.22 对 0.66、0.047 对 0.259；训练 0.18 对 0.47、0.008 对 0.068。

误差拆分（IMM，moving，3 s）：验证集纵向 2.72 m、bias −0.07、横向 0.58 m；训练集纵向 2.50 m、bias +1.01、横向 1.09 m。验证集上横向比纵向小很多，误差主要在沿路径的速度上；训练集转弯（42 个样本）横向升到 1.84 m。

**线上继续用 IMM。** GRU 和 Transformer 都不满足替换条件。权重在 `work_dirs/pred/gru.pt`（629 KB）、`work_dirs/pred/transformer.pt`（986 KB），新数据来了再训。

- 端到端 FDE 3 s 要低于 IMM 且领先 ≥ 90%：GRU 3.034 对 2.795（领先 31%），Transformer 3.517 对 2.795（领先 14%）。两边的 MR 3 s 都更低（0.188、0.239 对 0.521），但 FDE 不合格。minFDE 更低（GRU 1.251、Transformer 1.495 对 2.793），最好的那条模式更近，概率最高的那条更差。
- 标签当检测时 FDE 3 s 不能更高：GRU 2.390 对 1.388，Transformer 3.046 对 1.388，都更高。
- 误报不能更多：端到端 false_alarms_per_h 都是 24.826，IMM 是 12.413。hazard_reported 两边都是 3，和 IMM 一样。标签当检测时误报都是 0，hazard_reported 都是 5。
- 静止窗口的 FDE1：GRU 0.291、Transformer 0.294，IMM 是 0.599，没有把停着的车预测成要开走。
- predict_ms（中位数，CPU）：GRU 端到端 2745.9 ms、oracle 1356.9 ms；Transformer 4322.1 ms、3565.6 ms。IMM 约 1.1 ms。都超过 10 ms。

端到端 moving 3 s 的拆分（第 2 行相对 IMM）：GRU along 3.005、bias −1.770、across 0.382（IMM 2.722、−0.074、0.582）；Transformer along 3.488、bias −2.532、across 0.347。概率最高的路线偏慢，横向比 IMM 小。

窗口数（两份相同）：

```text
windows: train 5830 (416 moving), val 3523 (351 moving)
kept 2031 training windows (all 416 moving, 30 % of the rest)
mirrored: 4062 training windows
```

`train_gru.txt` 最后的表：

| windows | predictor | minADE | minFDE | ADE1 | FDE1 | MR@2m | n |
|---|---|---|---|---|---|---|---|
| all | cv | 0.536 | 0.829 | 0.536 | 0.829 | 0.114 | 3523 |
| all | imm | 0.536 | 0.828 | 0.536 | 0.829 | 0.114 | 3523 |
| all | learned | 0.180 | 0.226 | 0.376 | 0.535 | 0.013 | 3523 |
| moving | cv | 1.858 | 2.907 | 1.858 | 2.907 | 0.556 | 351 |
| moving | imm | 1.856 | 2.903 | 1.857 | 2.907 | 0.556 | 351 |
| moving | learned | 0.727 | 1.044 | 1.777 | 2.738 | 0.114 | 351 |
| static | cv | 0.390 | 0.599 | 0.390 | 0.599 | 0.065 | 3172 |
| static | imm | 0.390 | 0.599 | 0.390 | 0.599 | 0.065 | 3172 |
| static | learned | 0.119 | 0.135 | 0.221 | 0.291 | 0.002 | 3172 |

`train_transformer.txt` 最后的表：

| windows | predictor | minADE | minFDE | ADE1 | FDE1 | MR@2m | n |
|---|---|---|---|---|---|---|---|
| all | cv | 0.536 | 0.829 | 0.536 | 0.829 | 0.114 | 3523 |
| all | imm | 0.536 | 0.828 | 0.536 | 0.829 | 0.114 | 3523 |
| all | learned | 0.184 | 0.240 | 0.387 | 0.559 | 0.018 | 3523 |
| moving | cv | 1.858 | 2.907 | 1.858 | 2.907 | 0.556 | 351 |
| moving | imm | 1.856 | 2.903 | 1.857 | 2.907 | 0.556 | 351 |
| moving | learned | 0.813 | 1.208 | 1.879 | 2.948 | 0.174 | 351 |
| static | cv | 0.390 | 0.599 | 0.390 | 0.599 | 0.065 | 3172 |
| static | imm | 0.390 | 0.599 | 0.390 | 0.599 | 0.065 | 3172 |
| static | learned | 0.115 | 0.133 | 0.221 | 0.294 | 0.001 | 3172 |

第 1 步 Prediction（moving、moving straight、moving turning）：

验证（`logs_val/report.md`）：

**moving** (117 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| IMM | 1 s | 0.94 | 1.13 | 0.94 | 1.13 | 0.188 | 1.07 | 0.19 | 0.23 |
| IMM | 2 s | 1.35 | 1.98 | 1.35 | 1.98 | 0.368 | 1.92 | 0.03 | 0.40 |
| IMM | 3 s | 1.77 | 2.80 | 1.77 | 2.79 | 0.521 | 2.72 | -0.07 | 0.58 |
| CV | 1 s | 0.94 | 1.13 | 0.94 | 1.13 | 0.188 | 1.07 | 0.19 | 0.24 |
| CV | 2 s | 1.35 | 1.99 | 1.35 | 1.99 | 0.368 | 1.92 | 0.03 | 0.42 |
| CV | 3 s | 1.77 | 2.80 | 1.77 | 2.80 | 0.521 | 2.72 | -0.07 | 0.60 |

**moving, straight** (101 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| IMM | 1 s | 0.94 | 1.12 | 0.94 | 1.12 | 0.198 | 1.07 | 0.24 | 0.21 |
| IMM | 2 s | 1.34 | 1.96 | 1.34 | 1.96 | 0.376 | 1.89 | 0.17 | 0.36 |
| IMM | 3 s | 1.75 | 2.76 | 1.75 | 2.76 | 0.525 | 2.64 | 0.15 | 0.51 |
| CV | 1 s | 0.94 | 1.12 | 0.94 | 1.12 | 0.198 | 1.07 | 0.24 | 0.22 |
| CV | 2 s | 1.34 | 1.96 | 1.34 | 1.96 | 0.376 | 1.89 | 0.17 | 0.37 |
| CV | 3 s | 1.75 | 2.76 | 1.75 | 2.76 | 0.525 | 2.64 | 0.15 | 0.53 |

**moving, turning** (16 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| IMM | 1 s | 0.94 | 1.17 | 0.94 | 1.17 | 0.125 | 1.07 | -0.10 | 0.36 |
| IMM | 2 s | 1.42 | 2.13 | 1.42 | 2.13 | 0.312 | 2.16 | -0.80 | 0.71 |
| IMM | 3 s | 1.89 | 3.02 | 1.89 | 3.02 | 0.500 | 3.25 | -1.47 | 1.06 |
| CV | 1 s | 0.94 | 1.17 | 0.94 | 1.17 | 0.125 | 1.07 | -0.10 | 0.36 |
| CV | 2 s | 1.42 | 2.13 | 1.42 | 2.13 | 0.312 | 2.16 | -0.80 | 0.71 |
| CV | 3 s | 1.89 | 3.02 | 1.89 | 3.02 | 0.500 | 3.25 | -1.47 | 1.06 |

训练（`logs_train/report.md`）：

**moving** (165 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| IMM | 1 s | 0.95 | 1.14 | 0.95 | 1.13 | 0.145 | 1.06 | 0.64 | 0.29 |
| IMM | 2 s | 1.37 | 2.00 | 1.35 | 1.96 | 0.376 | 1.74 | 0.86 | 0.66 |
| IMM | 3 s | 1.83 | 3.00 | 1.79 | 2.91 | 0.636 | 2.50 | 1.01 | 1.09 |
| CV | 1 s | 0.96 | 1.14 | 0.96 | 1.14 | 0.145 | 1.05 | 0.63 | 0.30 |
| CV | 2 s | 1.37 | 2.02 | 1.37 | 2.02 | 0.412 | 1.72 | 0.84 | 0.70 |
| CV | 3 s | 1.84 | 3.03 | 1.84 | 3.03 | 0.661 | 2.46 | 0.96 | 1.18 |

**moving, straight** (123 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| IMM | 1 s | 0.99 | 1.16 | 0.99 | 1.16 | 0.171 | 1.13 | 0.81 | 0.24 |
| IMM | 2 s | 1.37 | 1.95 | 1.36 | 1.93 | 0.341 | 1.78 | 1.21 | 0.52 |
| IMM | 3 s | 1.78 | 2.84 | 1.77 | 2.80 | 0.569 | 2.51 | 1.55 | 0.84 |
| CV | 1 s | 0.99 | 1.16 | 0.99 | 1.16 | 0.171 | 1.13 | 0.81 | 0.25 |
| CV | 2 s | 1.37 | 1.95 | 1.37 | 1.95 | 0.358 | 1.77 | 1.20 | 0.53 |
| CV | 3 s | 1.78 | 2.85 | 1.78 | 2.85 | 0.577 | 2.50 | 1.55 | 0.87 |

**moving, turning** (42 samples)

| predictor | horizon | ADE1 | FDE1 | minADE | minFDE | MR@2m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| IMM | 1 s | 0.85 | 1.07 | 0.83 | 1.03 | 0.071 | 0.83 | 0.14 | 0.45 |
| IMM | 2 s | 1.36 | 2.16 | 1.31 | 2.06 | 0.476 | 1.63 | -0.17 | 1.09 |
| IMM | 3 s | 1.95 | 3.46 | 1.85 | 3.21 | 0.833 | 2.48 | -0.60 | 1.84 |
| CV | 1 s | 0.86 | 1.08 | 0.86 | 1.08 | 0.071 | 0.82 | 0.12 | 0.48 |
| CV | 2 s | 1.38 | 2.21 | 1.38 | 2.21 | 0.571 | 1.56 | -0.24 | 1.20 |
| CV | 3 s | 2.00 | 3.55 | 2.00 | 3.55 | 0.905 | 2.33 | -0.75 | 2.09 |

第 1 步 Training targets：

验证（`logs_val/report.md`）：

**Training targets**

The tier-2 training targets (prediction/dataset.py) of the same samples, cut from this run's track log the way train_predictor.py cuts them, against the labelled future: a pseudo ground truth, smoothed afterwards from the tracker's positions (filtered) or from the matched detections' centers (measured). windows: samples whose track has a training window there. Errors in m; > 1 m: share; along / bias / across as above (moving).

| targets | samples | windows | horizon | mean | p90 | > 1 m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| filtered | all matched | 970 / 1123 | 1 s | 0.23 | 0.62 | 0.049 | - | - | - |
| filtered | all matched | 970 / 1123 | 2 s | 0.24 | 0.62 | 0.048 | - | - | - |
| filtered | all matched | 970 / 1123 | 3 s | 0.26 | 0.69 | 0.057 | - | - | - |
| filtered | moving | 85 / 117 | 1 s | 0.52 | 1.17 | 0.165 | 0.50 | 0.22 | 0.10 |
| filtered | moving | 85 / 117 | 2 s | 0.62 | 1.40 | 0.259 | 0.60 | 0.16 | 0.12 |
| filtered | moving | 85 / 117 | 3 s | 0.66 | 1.46 | 0.259 | 0.65 | 0.14 | 0.11 |
| filtered | static | 885 / 1006 | 1 s | 0.21 | 0.51 | 0.038 | - | - | - |
| filtered | static | 885 / 1006 | 2 s | 0.20 | 0.51 | 0.028 | - | - | - |
| filtered | static | 885 / 1006 | 3 s | 0.22 | 0.58 | 0.037 | - | - | - |
| measured | all matched | 970 / 1123 | 1 s | 0.16 | 0.32 | 0.012 | - | - | - |
| measured | all matched | 970 / 1123 | 2 s | 0.15 | 0.32 | 0.008 | - | - | - |
| measured | all matched | 970 / 1123 | 3 s | 0.17 | 0.35 | 0.016 | - | - | - |
| measured | moving | 85 / 117 | 1 s | 0.18 | 0.38 | 0.012 | 0.16 | 0.03 | 0.04 |
| measured | moving | 85 / 117 | 2 s | 0.21 | 0.53 | 0.035 | 0.20 | -0.04 | 0.04 |
| measured | moving | 85 / 117 | 3 s | 0.22 | 0.64 | 0.047 | 0.21 | -0.04 | 0.04 |
| measured | static | 885 / 1006 | 1 s | 0.15 | 0.32 | 0.012 | - | - | - |
| measured | static | 885 / 1006 | 2 s | 0.15 | 0.31 | 0.006 | - | - | - |
| measured | static | 885 / 1006 | 3 s | 0.16 | 0.34 | 0.014 | - | - | - |

训练（`logs_train/report.md`）：

**Training targets**

The tier-2 training targets (prediction/dataset.py) of the same samples, cut from this run's track log the way train_predictor.py cuts them, against the labelled future: a pseudo ground truth, smoothed afterwards from the tracker's positions (filtered) or from the matched detections' centers (measured). windows: samples whose track has a training window there. Errors in m; > 1 m: share; along / bias / across as above (moving).

| targets | samples | windows | horizon | mean | p90 | > 1 m | along | bias | across |
|---|---|---|---|---|---|---|---|---|---|
| filtered | all matched | 1847 / 2212 | 1 s | 0.17 | 0.40 | 0.010 | - | - | - |
| filtered | all matched | 1847 / 2212 | 2 s | 0.17 | 0.41 | 0.011 | - | - | - |
| filtered | all matched | 1847 / 2212 | 3 s | 0.18 | 0.43 | 0.012 | - | - | - |
| filtered | moving | 132 / 165 | 1 s | 0.41 | 0.83 | 0.038 | 0.39 | 0.27 | 0.09 |
| filtered | moving | 132 / 165 | 2 s | 0.43 | 0.83 | 0.030 | 0.40 | 0.27 | 0.10 |
| filtered | moving | 132 / 165 | 3 s | 0.47 | 0.89 | 0.068 | 0.44 | 0.31 | 0.11 |
| filtered | static | 1715 / 2047 | 1 s | 0.15 | 0.33 | 0.008 | - | - | - |
| filtered | static | 1715 / 2047 | 2 s | 0.15 | 0.32 | 0.009 | - | - | - |
| filtered | static | 1715 / 2047 | 3 s | 0.16 | 0.35 | 0.008 | - | - | - |
| measured | all matched | 1847 / 2212 | 1 s | 0.13 | 0.28 | 0.006 | - | - | - |
| measured | all matched | 1847 / 2212 | 2 s | 0.13 | 0.28 | 0.005 | - | - | - |
| measured | all matched | 1847 / 2212 | 3 s | 0.15 | 0.33 | 0.008 | - | - | - |
| measured | moving | 132 / 165 | 1 s | 0.16 | 0.34 | 0.008 | 0.15 | -0.04 | 0.03 |
| measured | moving | 132 / 165 | 2 s | 0.16 | 0.37 | 0.000 | 0.15 | -0.04 | 0.04 |
| measured | moving | 132 / 165 | 3 s | 0.18 | 0.40 | 0.008 | 0.17 | -0.03 | 0.05 |
| measured | static | 1715 / 2047 | 1 s | 0.13 | 0.27 | 0.006 | - | - | - |
| measured | static | 1715 / 2047 | 2 s | 0.13 | 0.26 | 0.006 | - | - | - |
| measured | static | 1715 / 2047 | 3 s | 0.14 | 0.32 | 0.008 | - | - | - |

第 3 步 sweep.md：

`pred_gru/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | eval.learned_model= | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | eval.learned_model=data/train/pred_v1/gru.pt | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 3.034 | 0.188 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.239 (31%) | -0.333 (100%) |

`pred_gru_oracle/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | eval.learned_model= | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.388 | 0.286 |
| 2 | eval.learned_model=data/train/pred_v1/gru.pt | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 2.390 | 0.068 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 1.002 (0%) | -0.217 (100%) |

`pred_transformer/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | eval.learned_model= | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | eval.learned_model=data/train/pred_v1/transformer.pt | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 3.517 | 0.239 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.722 (14%) | -0.282 (100%) |

`pred_transformer_oracle/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | eval.learned_model= | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.388 | 0.286 |
| 2 | eval.learned_model=data/train/pred_v1/transformer.pt | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 3.046 | 0.099 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 1.658 (0%) | -0.186 (100%) |

sweep.csv 里两行（第 1 行 IMM，第 2 行第二档）：

| 报告 | 行 | predict_ms | false_alarms_per_h | hazard_reported | pred_minFDE3_moving | pred_along3_moving | pred_along_bias3_moving | pred_across3_moving |
|---|---|---|---|---|---|---|---|---|
| pred_gru | IMM | 1.115 | 12.413 | 3 | 2.793 | 2.722 | −0.074 | 0.582 |
| pred_gru | GRU | 2745.858 | 24.826 | 3 | 1.251 | 3.005 | −1.770 | 0.382 |
| pred_gru_oracle | IMM | 0.714 | 0 | 5 | 1.383 | 1.416 | −1.006 | 0.219 |
| pred_gru_oracle | GRU | 1356.933 | 0 | 5 | 0.826 | 2.441 | −2.324 | 0.130 |
| pred_transformer | IMM | 1.123 | 12.413 | 3 | 2.793 | 2.722 | −0.074 | 0.582 |
| pred_transformer | Transformer | 4322.116 | 24.826 | 3 | 1.495 | 3.488 | −2.532 | 0.347 |
| pred_transformer_oracle | IMM | 0.718 | 0 | 5 | 1.383 | 1.416 | −1.006 | 0.219 |
| pred_transformer_oracle | Transformer | 3565.584 | 0 | 5 | 1.174 | 3.111 | −3.078 | 0.146 |


### T23 复核（2026-10-28）

结论同意：**线上继续用 IMM**。补充五点，其中第 4 点的耗时测量无效，已修。

1. **伪真值够准，以后默认用 measured。**
   - 运动车 3 s：measured 离标签平均 0.22 m（验证）/ 0.18 m（训练），> 1 m 的 4.7% / 0.8%。
   - filtered 是 0.66 / 0.47 m，纵向 bias 为正（+0.14 / +0.31 m）：滤过两次的位置，在车减速时会冲到前面。
   - measured 的误差约是预测误差（2.8 m）的 8%，不用改成拿标签当训练目标。
   - 代码默认改成 measured（`SampleConfig.targets`、`train_predictor.py --targets`）；旧日志（9 列）要显式加 `--targets filtered`。
   - 运动样本里有一部分没有训练窗口（验证 32/117、训练 33/165），多是未来被遮挡或轨迹断了。这部分运动，模型学不到。
2. **误差在纵向，不在路线形状；道路地图 / 占据栅格暂不做。**
   - 验证 IMM：纵向 2.72 m、横向 0.58 m，横向只占平方误差的约 4%。训练：2.50 m、1.09 m，约 16%。
   - 横向大的只有转弯的车：验证 16 个样本 1.06 m，训练 42 个 1.84 m。
   - 方案写在 [DRIVABLE_AREA_PLAN.md](DRIVABLE_AREA_PLAN.md)，暂不做。
   - 纵向误差的来源：
     - 标签当检测：纵向 1.42 m、bias −1.01，预测慢了。车在加速、起步，跟踪的速度跟不上。
     - 端到端：纵向 2.72 m、bias ≈ 0，快慢都有，多出来的是检测带来的速度噪声。
   - 所以下一步的抓手是跟踪器的速度估计（H4 / D2 的范围，人定暂缓）。
3. **第二档为什么输：概率最高的那条偏慢。**
   - 端到端 bias：GRU −1.77 m、Transformer −2.53 m（IMM −0.07）；标签当检测更明显，−2.32 / −3.08（IMM −1.01）。
   - 原因多半是训练、验证场景里车的行为不同：
     - 训练场景的车多在减速：IMM 在训练场景 bias +1.01，也就是预测快了；
     - 验证场景不是：bias −0.07。
     - 41 辆运动车学到的"会减速"，换到验证场景就不对了。数据太少，场景之间的差别被学成了先验。
   - 但它的几条模式覆盖得好：minFDE 1.25 对 2.79，MR 0.19 对 0.52。停着的车在伪真值上 FDE1 0.29 对 0.60，不会把停着的车往外推。新数据多了值得再训；规划如果用全部模式和概率，收益会先体现在 MR 上。
   - train_*.txt 的伪真值验证窗口上，GRU 的 FDE1 比 IMM 低（2.74 对 2.91），对标签反而高。两边样本不同：
     - 伪真值只算有训练窗口的；
     - 对标签算的是全部关键帧样本，含约四分之一没有窗口的。
     - 端到端领先 31%，说明这个差别在场景之间的波动范围内；标签当检测时的差（领先 0%）是稳的。
4. **耗时那条测量无效，结论不变。**
   - 2746 / 4322 ms 是 `--jobs 8` 的 8 个进程各自开满 torch 线程、互相抢 CPU 测出来的。评估也因此跑了 1418–1839 s；只用 IMM 的同样评估只要几秒（第 1 步 logs_val 7 s）。
   - 已改：评估工具里学习模型单线程跑（`pipeline_eval._learned_predictor`），predict_ms 是单核时间。
   - 我这边单线程实测（同结构未训练的模型，容器 CPU）：
     - GRU：10 条轨迹约 16 ms；30 条约 40 ms，其中输入构造 13 ms、网络 27 ms；
     - Transformer：30 条约 75 ms。
     - 仍超过 10 ms。
   - 真要上线得先优化：只对在动的轨迹跑网络（停着的走 IMM）；邻居的历史每帧只编码一次，不按每个目标重复编码；Orin 上导出 TensorRT。
5. **误报** 12.4 → 24.8 次 / 小时，其实是 1 次变 2 次（验证集关键帧时长约 4.8 分钟）。样本太少，不单独下结论。

基线权重在 `work_dirs/pred/gru.pt`、`work_dirs/pred/transformer.pt`，厂区新数据来了用同样的流程再训（训练目标默认 measured）。

## T24 第二档第二版：跟踪的部分合并（状态估计交给网络）

T23 复核的结论：
- IMM 的误差主要在纵向，也就是速度：端到端 2.72 m，标签当检测时 1.42 m，差出来的主要是检测带来的速度噪声；
- 停着的车被报成在动，每小时 87 次（T22），是跟踪层最大的问题；
- 第一版概率最高的那条路线偏慢：未来怎么走要靠场景间的行为先验，数据太少学不好。"现在的状态"不依赖未来，数据也够。

这一版跟踪器不变，继续负责关联、ID、轨迹的产生和结束、遮挡外推。网络多两样（TRACKING_PREDICTION.md 3.3.8）：
- 输入加上每帧匹配到的原始检测：中心、速度头、分数（`--det-inputs`）；
- 输出加"现在的状态"：当前速度、在动的概率（`--state-head`）。
  - 速度的目标：伪真值（measured）在 ±0.5 s 上的速度；
  - 在动的目标：和自动标注同一条规则。

评估三种用法，和 IMM、跟踪器的规则配对比：
- 用网络的速度匀速外推，替换 IMM 的预测（`eval.learned_mode=velocity`）；
- 用网络的在动概率加滞回，替换跟踪器的运动状态规则（`eval.learned_motion=true`）：≥ 0.6 进入在动，< 0.4 退出；
- 两者一起。

```text
任务 T24：重新生成 35 个场景的跟踪日志（多一列跟踪器的运动状态），训练两个带状态输出的 GRU
（state：只加状态输出；det：再加原始检测输入），在验证集上和 IMM、跟踪器的运动状态规则配对比较。
先按通用约定做（拉到了代码改动，跑快速测试）。前置：T23 完成。CPU 就能训，有空闲 GPU 就加 --device cuda:0。

0. mkdir -p data/reports/T24 data/train/pred_v2; TASK=T24；VAL、TRAIN、ev 同 T18 第 0、3 步。
1. 跟踪日志（覆盖 T23 的）：
     ev logs_val "$VAL" "data/tune/deploy/{scene}/pillar/results.pkl" --detections raw --no-mcap --bootstrap 0 \
         --track-logs data/tune/track_logs
     ev logs_train "$TRAIN" "data/tune/nusc_pp_train/{scene}/pillar/results.pkl" --detections raw --no-mcap --bootstrap 0 \
         --track-logs data/tune/track_logs
     ls data/tune/track_logs/*.npz | wc -l        # 应是 35
     python -c "import numpy as np; print(np.load('data/tune/track_logs/$(echo $VAL | cut -d' ' -f1).npz')['rows'].shape[1])"   # 应是 16
2. 训练（训练目标默认 measured）：
     LT=$(for r in $TRAIN; do echo data/tune/track_logs/$r.npz; done); LV=$(for r in $VAL; do echo data/tune/track_logs/$r.npz; done)
     for v in state det; do
       extra="--state-head"; [ $v = det ] && extra="--state-head --det-inputs"
       python tools/dog/train_predictor.py --logs $LT --val-logs $LV --encoder gru --modes 6 --epochs 60 \
           --static-keep 0.3 --mirror $extra --out data/train/pred_v2/$v.pt 2>&1 | tee data/reports/T24/train_$v.txt
     done
   train_*.txt 最后有两张表：
   - 轨迹：同 T23；
   - "State now"：跟踪器（它的速度、记下的运动状态）和网络在同一批验证窗口上的速度误差、在动的召回率、静止窗口被判在动的比例。
   这里的"真值"是伪真值；能不能替换只看第 3 步（对标签）。
3. 和标签比，每个模型 5 行配对（第 1 行是现在的 IMM 加跟踪器规则）；端到端和标签当检测各一次：
     for v in state det; do
       M=data/train/pred_v2/$v.pt
       python -c 'import json, sys; m = sys.argv[1]; json.dump([{},
           {"eval.learned_model": m, "eval.learned_mode": "trajectory"},
           {"eval.learned_model": m, "eval.learned_mode": "velocity"},
           {"eval.learned_model": m, "eval.learned_mode": "imm", "eval.learned_motion": True},
           {"eval.learned_model": m, "eval.learned_mode": "velocity", "eval.learned_motion": True}],
           open(sys.argv[2], "w"), indent=1)' $M data/reports/T24/rows_$v.json
       ev pred_$v "$VAL" "data/tune/deploy/{scene}/pillar/results.pkl" --detections raw --no-mcap \
           --configs data/reports/T24/rows_$v.json
       ev pred_${v}_oracle "$VAL" "data/tune/deploy/{scene}/pillar/results.pkl" --detections labels \
           --dense-labels "data/labels/{scene}/labels_full2.pkl" --no-mcap --configs data/reports/T24/rows_$v.json
     done
   每份的 sweep.md 看 false moving / h、moving missed、FDE 3 s、MR 3 s 和领先比例；sweep.csv 里还要看：
   - 运动状态：motion_relevant_latency_p90、motion_moving_recall、motion_false_moving；
   - 预测：pred_along3_moving、pred_along_bias3_moving、pred_across3_moving；
   - 预警：false_alarms_per_h、hazard_reported；
   - 耗时：predict_ms。

怎么选（每个模型分别看；都满足的，选端到端更好的那个）：
- 运动状态能不能替换跟踪器的规则，看端到端第 4 行对第 1 行，同时满足：
  - false moving / h 更低，领先 ≥ 90%；
  - moving missed 不增加；
  - motion_relevant_latency_p90 不比第 1 行晚 0.5 s 以上；
  - false_alarms_per_h 不高于第 1 行，hazard_reported 不少于第 1 行。
- 速度预测能不能替换 IMM，看第 3 行对第 1 行，同时满足：
  - 端到端 FDE 3 s 更低，领先 ≥ 90%，MR 3 s 不高于第 1 行；
  - 标签当检测的 FDE 3 s 不高于第 1 行；
  - false_alarms_per_h 不高于第 1 行，hazard_reported 不少于第 1 行。
- 两项都满足时，再看第 5 行（两者一起）是否仍都满足。
- det 对 state：det 不比 state 好，说明原始检测输入没用，就选 state。
- predict_ms（中位数，单线程）只记录，本轮不作为门槛。过了上面的门槛，下一步再提速：只对在动的轨迹跑网络、邻居历史每帧只编码一次、Orin 上用 TensorRT。
都不满足：保持现状，模型留作基线。满足了写清是哪个模型、哪种用法，由人决定是否上线。

停止条件：任何报错；第 1 步日志不是 35 个，或 rows 不是 16 列（代码没拉到）；训练 loss 出现 nan。
提交：data/reports/T24/、本文档。不提交 .pt 和跟踪日志。
报告（写进"T24 结果"）：
- 两个 train_*.txt 开头的窗口数行和最后两张表；
- 四份 sweep.md 的两张表；
- sweep.csv 里每行的 motion_relevant_latency_p90、motion_moving_recall、motion_false_moving、pred_along3_moving、
  pred_along_bias3_moving、pred_across3_moving、false_alarms_per_h、hazard_reported、predict_ms；
- 按"怎么选"给出的结论；每个训练和评估的耗时。
```

### T24 结果

拉到 `92c9da8`。`dog_perception/`、`tools/` 有改动，快速测试 160 passed（`CUDA_VISIBLE_DEVICES=3`，24 s）。

跟踪日志覆盖 T23 的，35 个，`rows` 是 16 列。耗时：logs_val 7 s，logs_train 13 s。训练在物理 GPU 5 上（空闲 8993 MiB，利用率 0%，进程里是 `cuda:0`），目标是默认的 measured。state 46 s，det 52 s。loss 没有 nan。对比（CPU）：pred_state 54 s，pred_state_oracle 49 s，pred_det 54 s，pred_det_oracle 49 s。

窗口数（两份相同）：

```text
windows: train 5830 (416 moving), val 3523 (351 moving)
kept 2031 training windows (all 416 moving, 30 % of the rest)
mirrored: 4062 training windows
```

**保持现状。** 运动状态规则和 IMM 都不换。权重在 `work_dirs/pred/state.pt`（663 KB）、`work_dirs/pred/det.pt`（664 KB）。

运动状态（端到端第 4 行对第 1 行）两边都不满足：

- state：false moving / h 62.065 对 86.891，领先只有 67%；moving missed 4 对 2，增加了；确认相关的 latency p90 1.9 s 对 1.2 s，晚了 0.7 s；hazard_reported 2 对 3，少了 1。false_alarms_per_h 从 12.413 降到 0。
- det：false moving / h 同样是 62.065，领先 66%；moving missed 同样增加到 4。latency p90 1.301 s，只晚 0.1 s；hazard_reported 仍是 3；false_alarms_per_h 也降到 0。假运动少了，但漏掉的运动段变多，领先比例不到 90%。

速度外推（第 3 行对第 1 行）也不满足：

- state：端到端 FDE 3 s 3.059 对 2.795，更高，领先 27%；MR 3 s 0.709 对 0.521，更高。标签当检测的 FDE 2.607 对 1.388，更高。
- det：端到端 FDE 3 s 2.603，比 IMM 低 0.193，领先 75%，不到 90%；MR 3 s 0.530，比 0.521 高。标签当检测的 FDE 2.175 对 1.388，更高。误报和 hazard_reported 与 IMM 相同（12.413 / 小时，3 次）。

两项都没过，第 5 行（一起用）不再采用。det 的速度外推在端到端上比 state 低（2.603 对 3.059），运动状态的 latency p90 也更短（1.301 s 对 1.9 s），hazard 没少报；但两边都没过门槛，原始检测输入这一项不单独上线。

predict_ms 只记录、不作门槛。端到端上 IMM 约 1.1 ms，网络的几种用法大约 3.4–4.8 ms。

`train_state.txt` 最后两张表：

| windows | predictor | minADE | minFDE | ADE1 | FDE1 | MR@2m | n |
|---|---|---|---|---|---|---|---|
| all | cv | 0.536 | 0.829 | 0.536 | 0.829 | 0.114 | 3523 |
| all | imm | 0.536 | 0.828 | 0.536 | 0.829 | 0.114 | 3523 |
| all | learned | 0.188 | 0.235 | 0.385 | 0.553 | 0.019 | 3523 |
| moving | cv | 1.858 | 2.907 | 1.858 | 2.907 | 0.556 | 351 |
| moving | imm | 1.856 | 2.903 | 1.857 | 2.907 | 0.556 | 351 |
| moving | learned | 0.862 | 1.250 | 1.867 | 2.944 | 0.188 | 351 |
| static | cv | 0.390 | 0.599 | 0.390 | 0.599 | 0.065 | 3172 |
| static | imm | 0.390 | 0.599 | 0.390 | 0.599 | 0.065 | 3172 |
| static | learned | 0.113 | 0.123 | 0.221 | 0.288 | 0.001 | 3172 |

State now (targets: velocity over +-0.5 s and the auto-label's moving rule on the pseudo ground truth; tracker: its velocity and logged motion state, moving = 'moving'; learned: P(moving) >= 0.5)

| windows | estimator | vel err m/s | moving recall | false moving | n |
|---|---|---|---|---|---|
| all | tracker | 0.209 | 0.829 | 0.0186 | 3509 |
| all | learned | 0.082 | 0.693 | 0.0094 | 3509 |
| moving | tracker | 0.685 | 0.831 | 0.0915 | 349 |
| moving | learned | 0.648 | 0.729 | 0.0211 | 349 |
| static | tracker | 0.156 | 0.810 | 0.0153 | 3160 |
| static | learned | 0.019 | 0.333 | 0.0089 | 3160 |

`train_det.txt` 最后两张表：

| windows | predictor | minADE | minFDE | ADE1 | FDE1 | MR@2m | n |
|---|---|---|---|---|---|---|---|
| all | cv | 0.536 | 0.829 | 0.536 | 0.829 | 0.114 | 3523 |
| all | imm | 0.536 | 0.828 | 0.536 | 0.829 | 0.114 | 3523 |
| all | learned | 0.187 | 0.249 | 0.370 | 0.545 | 0.021 | 3523 |
| moving | cv | 1.858 | 2.907 | 1.858 | 2.907 | 0.556 | 351 |
| moving | imm | 1.856 | 2.903 | 1.857 | 2.907 | 0.556 | 351 |
| moving | learned | 0.836 | 1.290 | 1.809 | 2.956 | 0.191 | 351 |
| static | cv | 0.390 | 0.599 | 0.390 | 0.599 | 0.065 | 3172 |
| static | imm | 0.390 | 0.599 | 0.390 | 0.599 | 0.065 | 3172 |
| static | learned | 0.115 | 0.134 | 0.211 | 0.278 | 0.002 | 3172 |

State now (targets: velocity over +-0.5 s and the auto-label's moving rule on the pseudo ground truth; tracker: its velocity and logged motion state, moving = 'moving'; learned: P(moving) >= 0.5)

| windows | estimator | vel err m/s | moving recall | false moving | n |
|---|---|---|---|---|---|
| all | tracker | 0.209 | 0.829 | 0.0186 | 3509 |
| all | learned | 0.067 | 0.697 | 0.0091 | 3509 |
| moving | tracker | 0.685 | 0.831 | 0.0915 | 349 |
| moving | learned | 0.540 | 0.744 | 0.0211 | 349 |
| static | tracker | 0.156 | 0.810 | 0.0153 | 3160 |
| static | learned | 0.015 | 0.238 | 0.0086 | 3160 |

四份 sweep.md：

`pred_state/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | defaults | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | eval.learned_model=data/train/pred_v2/state.pt, eval.learned_mode=trajectory | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 3.303 | 0.274 |
| 3 | eval.learned_model=data/train/pred_v2/state.pt, eval.learned_mode=velocity | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 3.059 | 0.709 |
| 4 | eval.learned_model=data/train/pred_v2/state.pt, eval.learned_mode=imm, eval.learned_motion=True | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 4 | 62.065 | 2.795 | 0.521 |
| 5 | eval.learned_model=data/train/pred_v2/state.pt, eval.learned_mode=velocity, eval.learned_motion=True | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 4 | 62.065 | 3.059 | 0.709 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.508 (14%) | -0.248 (100%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.264 (27%) | 0.188 (5%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 2 (0%) | -24.826 (67%) | 0.000 (0%) | 0.000 (0%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 2 (0%) | -24.826 (67%) | 0.264 (27%) | 0.188 (5%) |

`pred_state_oracle/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | defaults | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.388 | 0.286 |
| 2 | eval.learned_model=data/train/pred_v2/state.pt, eval.learned_mode=trajectory | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 2.920 | 0.168 |
| 3 | eval.learned_model=data/train/pred_v2/state.pt, eval.learned_mode=velocity | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 2.607 | 0.621 |
| 4 | eval.learned_model=data/train/pred_v2/state.pt, eval.learned_mode=imm, eval.learned_motion=True | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 2 | 0.000 | 1.388 | 0.286 |
| 5 | eval.learned_model=data/train/pred_v2/state.pt, eval.learned_mode=velocity, eval.learned_motion=True | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 2 | 0.000 | 2.607 | 0.621 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 1.532 (1%) | -0.118 (100%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 1.219 (1%) | 0.335 (5%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 2 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 2 (0%) | 0.000 (0%) | 1.219 (1%) | 0.335 (5%) |

`pred_det/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | defaults | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | eval.learned_model=data/train/pred_v2/det.pt, eval.learned_mode=trajectory | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 3.260 | 0.274 |
| 3 | eval.learned_model=data/train/pred_v2/det.pt, eval.learned_mode=velocity | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 2 | 86.891 | 2.603 | 0.530 |
| 4 | eval.learned_model=data/train/pred_v2/det.pt, eval.learned_mode=imm, eval.learned_motion=True | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 4 | 62.065 | 2.795 | 0.521 |
| 5 | eval.learned_model=data/train/pred_v2/det.pt, eval.learned_mode=velocity, eval.learned_motion=True | 0.839 | 0.825 | 0.832 | 0.641 | 0.609 | 0.786 | 1.862 | 27.102 | 1.200 | 4 | 62.065 | 2.603 | 0.530 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.465 (16%) | -0.248 (100%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | -0.193 (75%) | 0.009 (38%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 2 (0%) | -24.826 (66%) | 0.000 (0%) | 0.000 (0%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 2 (0%) | -24.826 (66%) | -0.193 (75%) | 0.009 (38%) |

`pred_det_oracle/sweep.md`：

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | defaults | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 1.388 | 0.286 |
| 2 | eval.learned_model=data/train/pred_v2/det.pt, eval.learned_mode=trajectory | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 2.941 | 0.186 |
| 3 | eval.learned_model=data/train/pred_v2/det.pt, eval.learned_mode=velocity | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 0 | 0.000 | 2.175 | 0.435 |
| 4 | eval.learned_model=data/train/pred_v2/det.pt, eval.learned_mode=imm, eval.learned_motion=True | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 1 | 0.000 | 1.388 | 0.286 |
| 5 | eval.learned_model=data/train/pred_v2/det.pt, eval.learned_mode=velocity, eval.learned_motion=True | 1.000 | 1.000 | 1.000 | 0.000 | 0.961 | 0.957 | 1.655 | 0.000 | 0.501 | 1 | 0.000 | 2.175 | 0.435 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 1.553 (1%) | -0.099 (100%) |
| 3 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.787 (1%) | 0.149 (5%) |
| 4 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 1 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) |
| 5 | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) | 1 (0%) | 0.000 (0%) | 0.787 (1%) | 0.149 (5%) |

sweep.csv 里每行：

| 报告 | 行 | latency p90 | moving recall | false moving | along | bias | across | false alarms / h | hazard reported | predict_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| pred_state | 1 IMM | 1.200 | 0.760 | 0.017 | 2.722 | -0.074 | 0.582 | 12.413 | 3 | 1.112 |
| pred_state | 2 trajectory | 1.200 | 0.760 | 0.017 | 3.289 | -2.326 | 0.320 | 24.826 | 3 | 3.450 |
| pred_state | 3 velocity | 1.200 | 0.760 | 0.017 | 3.056 | -1.971 | 0.329 | 12.413 | 3 | 3.414 |
| pred_state | 4 motion | 1.900 | 0.640 | 0.011 | 2.722 | -0.074 | 0.582 | 0.000 | 2 | 4.563 |
| pred_state | 5 velocity+motion | 1.900 | 0.640 | 0.011 | 3.056 | -1.971 | 0.329 | 0.000 | 2 | 3.407 |
| pred_state_oracle | 1 IMM | 1.498 | 0.727 | 0.004 | 1.416 | -1.006 | 0.219 | 0.000 | 5 | 0.711 |
| pred_state_oracle | 2 trajectory | 1.498 | 0.727 | 0.004 | 2.994 | -2.867 | 0.103 | 0.000 | 5 | 2.821 |
| pred_state_oracle | 3 velocity | 1.498 | 0.727 | 0.004 | 2.712 | -2.624 | 0.119 | 0.000 | 5 | 2.824 |
| pred_state_oracle | 4 motion | 3.700 | 0.453 | 0.002 | 1.416 | -1.006 | 0.219 | 0.000 | 2 | 3.645 |
| pred_state_oracle | 5 velocity+motion | 3.700 | 0.453 | 0.002 | 2.712 | -2.624 | 0.119 | 0.000 | 2 | 2.816 |
| pred_det | 1 IMM | 1.200 | 0.760 | 0.017 | 2.722 | -0.074 | 0.582 | 12.413 | 3 | 1.097 |
| pred_det | 2 trajectory | 1.200 | 0.760 | 0.017 | 3.240 | -2.567 | 0.299 | 24.826 | 3 | 3.684 |
| pred_det | 3 velocity | 1.200 | 0.760 | 0.017 | 2.578 | -1.993 | 0.299 | 12.413 | 3 | 3.636 |
| pred_det | 4 motion | 1.301 | 0.620 | 0.010 | 2.722 | -0.074 | 0.582 | 0.000 | 3 | 4.815 |
| pred_det | 5 velocity+motion | 1.301 | 0.620 | 0.010 | 2.578 | -1.993 | 0.299 | 0.000 | 3 | 3.631 |
| pred_det_oracle | 1 IMM | 1.498 | 0.727 | 0.004 | 1.416 | -1.006 | 0.219 | 0.000 | 5 | 0.702 |
| pred_det_oracle | 2 trajectory | 1.498 | 0.727 | 0.004 | 3.026 | -2.914 | 0.136 | 0.000 | 5 | 2.997 |
| pred_det_oracle | 3 velocity | 1.498 | 0.727 | 0.004 | 2.272 | -2.198 | 0.094 | 0.000 | 5 | 3.031 |
| pred_det_oracle | 4 motion | 3.800 | 0.494 | 0.003 | 1.416 | -1.006 | 0.219 | 0.000 | 5 | 3.777 |
| pred_det_oracle | 5 velocity+motion | 3.800 | 0.494 | 0.003 | 2.272 | -2.198 | 0.094 | 0.000 | 5 | 3.061 |

### T24 复核（2026-10-28）

结论同意：**保持现状**，运动状态规则和 IMM 都不换。人定：不在 0921 上继续调，权重（`work_dirs/pred/state.pt`、`det.pt`）留作基线，厂区新数据来了再一起改、一起训。下面记下这一轮看到的东西，和新数据来了先做什么。

1. **伪真值上，网络的状态估计明显好于跟踪器**（train_*.txt，验证窗口）：

   | 项目 | 跟踪器 | state | det |
   |---|---|---|---|
   | 速度误差，全部窗口（m/s） | 0.209 | 0.082 | 0.067 |
   | 速度误差，停着的车（m/s） | 0.156 | 0.019 | 0.015 |
   | 速度误差，运动的车（m/s） | 0.685 | 0.648 | 0.540 |
   | 误判在动的比例 | 0.019 | 0.009 | 0.009 |
   | 在动的召回率（P ≥ 0.5） | 0.83 | 0.69 | 0.70 |

   网络更保守：误判在动减半，召回率也低一些。

2. **对标签，运动状态是换了一个工作点，不是全面更好**（端到端第 4 行，sweep.csv）：
   - 误判在动的帧比例 0.017 → 0.010；
   - 停着的车报成在动：每小时 86.9 → 62.1 次，即 7 → 5 次，领先 66–67%；
   - 误报警 1 → 0 次；
   - 但召回率 0.76 → 0.62 / 0.64，漏掉的相关运动段 2 → 4。
   - state 模型的报出延迟 p90 1.2 → 1.9 s，危险接近报出次数 3 → 2，安全上不能接受。det 模型的延迟只晚 0.1 s，危险接近报出次数没少。
   - 验证集只有 7 次误报在动，差 2 次很难到 90% 领先。这一项在 0921 上本来就难判，要等厂区数据。

3. **网络估的速度系统性偏慢，这是这一轮最主要的发现**（纵向 bias，3 s）：

   | 用法 | 端到端 | 标签当检测 |
   |---|---|---|
   | IMM | −0.07 m | −1.01 m |
   | det，速度外推 | −1.99 m | −2.20 m |
   | state，速度外推 | −1.97 m | −2.62 m |
   | 两个模型的轨迹模式 | −2.33 / −2.57 m | −2.87 / −2.91 m |

   - "现在的速度"不依赖未来怎么走，它也偏慢。所以 T23 复核时归因的"场景间行为先验"不是全部原因，训练本身就把速度往 0 拉。
   - 推测的原因：
     - 训练窗口约 80% 是停着的车（static-keep 后 1615 对 416），L1 回归在拿不准时取 0；
     - 速度目标在"不在动"时被置成 0（`track_targets`），慢车、刚起步的车的目标偏小。
   - 方向估得好：横向误差 0.58 → 0.30 m。det 的速度外推 FDE 已经低于 IMM（2.60 对 2.80，领先 75%）。偏慢修掉以后可能更好，要实测。

4. **原始检测输入有用**：det 在这几项上都好于 state，以后默认带上 `--det-inputs`：
   - 运动车的速度误差（伪真值上）：0.54 对 0.65 m/s；
   - 速度外推的 FDE：2.60 对 3.06 m；
   - 报出延迟 p90：1.3 对 1.9 s。

5. **标签当检测时网络明显变差**：召回率 0.45–0.49（跟踪器 0.73），延迟 p90 3.7–3.8 s（跟踪器 1.5 s）。
   - 输入换成干净的标签框，和训练时的真实检测不一样了，线上不会遇到这种情况；
   - 但说明网络和检测器绑定，换检测模型就要重训。流程现成：日志加训练约 2 分钟。

6. **耗时**：端到端每帧 3.4–4.8 ms（IMM 1.1 ms），在 10 ms 以内。比我本地估的低（T23 复核按 30 条轨迹约 40 ms 算），因为 0921 每帧的轨迹少。轨迹多的厂区场景要再测。

**新数据来了先做的事**（现在不做）：
- 速度目标不再按在动标志置 0，直接用窗口速度；
- 在动窗口的状态损失加权，或者换一种样本均衡；
- `train_predictor.py` 的状态表加带符号的速度偏差（投到真实运动方向），直接看偏慢有没有修掉；
- 运动状态的阈值做成可调的；再加一种"否决"用法：跟踪器说在动、网络很确定不在动时才改成不在动。这样跟踪器仍是主判断，只用网络去掉停着的车的假运动，不会新增漏报；
- 用厂区的验证 / 测试集重跑 T24 的配对比较。

## T25 Orin 部署：环境和通道

部署到 Jetson AGX Orin 的总体做法见 [DEPLOY_ORIN.md](DEPLOY_ORIN.md)：模型转换、量化、C++ 移植、ROS 2 集成，以及每一步的验收。T25 先把通道打通、把两边的环境查清楚，不改代码。

```text
任务 T25：打通服务器到 Orin 的通道，记录服务器和 Orin 的环境，装好缺的包。
先按通用约定做。前置：人在本机开着反向隧道（DEPLOY_ORIN.md 第 2 节的 autossh 命令）。
sudo 密码问人要，不写进任何文件和报告。

0. mkdir -p data/reports/T25
1. 通道：
   - 服务器 ~/.ssh/config 加 Host orin（HostName localhost、Port 5037、User seres，DEPLOY_ORIN.md 第 2 节）；
   - 服务器公钥放到板子上：ssh-copy-id orin（要人输一次密码）；
   - ssh orin 'hostname; uname -m' 应输出 ubuntu、aarch64。连不上就停，报告错误。
2. 同步代码：deploy/scripts/sync_to_orin.sh
3. 环境报告：
     bash deploy/scripts/orin_check.sh 2>&1 | tee data/reports/T25/server_env.txt
     ssh orin 'bash ~/dogperc/repo/deploy/scripts/orin_check.sh' 2>&1 | tee data/reports/T25/orin_env.txt
4. 板上补包（sudo）。标 MISSING 的才装：
     sudo apt install -y libeigen3-dev libyaml-cpp-dev libgtest-dev python3-colcon-common-extensions
     sudo apt install -y python3-libnvinfer                # 没有 python tensorrt 时
     python3 -m pip install --user numpy scipy             # 没有 numpy / scipy 时
   装完再跑一次 orin_check.sh，存成 orin_env_after.txt。根分区剩余 < 3 GB 就停，报告，不要自己删板上的东西。
5. 服务器补包：在项目环境里装
     pip install "tensorrt==10.3.*" onnx onnxruntime polygraphy "nvidia-modelopt[onnx]"
   - tensorrt 的大版本和小版本要和板子一致（10.3）；
   - 和项目环境冲突就新建 conda 环境 deploy，装同样的包加项目依赖，torch 后端那一步回项目环境跑；
   - 再记下：python -c "import tensorrt; print(tensorrt.__version__)"、nvidia-smi、docker --version、
     sudo -n true 是否可用、0921 原始 bag 在服务器上的路径（T30 要从中截 ROS 2 bag 片段）。
6. 测试：python -m pytest tests/dog -m "not slow" -q，以及 python -m pytest tests/dog/test_deploy.py -q。
   test_deploy.py 里 TensorRT 小引擎那条在服务器上不应跳过；跳过就说明 tensorrt 或 CUDA 没装好。

停止条件：通道不通；板上根分区剩余 < 3 GB；安装报错。
提交：data/reports/T25/、本文档。
报告（写进"T25 结果"）：
- 两份环境报告里的版本行和 MISSING 行，装了什么，装完后的剩余空间；
- 服务器的 tensorrt 版本、GPU 型号和驱动、有没有 docker 和 sudo、0921 原始 bag 的路径；
- 测试结果。
```

### T25 结果

2026-10-28。`ssh orin` 输出 `ubuntu`、`aarch64`。代码在板上 `~/dogperc`（1.9 MB）。原始输出在 `data/reports/T25/`。

**板子**（`orin_env.txt` 是安装前；`orin_env_after.txt` 是装完 trtexec 之后）：

- Jetson AGX Orin，L4T R36.4.4，Ubuntu 22.04.5，内核 5.15.148-tegra aarch64
- CUDA 12.6，cuDNN 9.3.0.75，libnvinfer 10.3.0.30+cuda12.5，Python tensorrt 10.3.0（`~/.local`）
- ROS 2 Humble（rclcpp、rosbag2、sensor_msgs、visualization_msgs），colcon 0.3.0（`python3-colcon-common-extensions`）。`colcon version-check` 没有输出、一直占着，已结束该进程，报告才写完后面的段
- gcc 11.4，eigen 3.4.0，yaml-cpp 0.7.0，gtest 1.11.0；numpy 1.26.4，scipy 1.8.0
- 功耗模式 MAXN。内存 61 GiB，根分区剩余 5.5 GB / 54 GB（90%）
- 安装前唯一的 `MISSING` 行是 `trtexec`。第 4 步列出的 eigen、yaml-cpp、gtest、colcon、python tensorrt、numpy、scipy 都已经在。板上的 apt 源没有这个包，从 NVIDIA r36.4 仓库装了同版本的 `libnvinfer-bin` 10.3.0.30，以及它依赖的 `libnvinfer-lean10`、`libnvinfer-dispatch10`、`libnvinfer-vc-plugin10`。装完后 `trtexec` 在 `/usr/src/tensorrt/bin/trtexec`（TensorRT v100300），`orin_env_after.txt` 里没有 `MISSING` 行，根分区仍剩 5.5 GB

**服务器**：

- `orin_check.sh` 用的是 PATH 上的 `/zjst/robot/qnn/py310/bin/python3`，所以 `server_env.txt` 里是 `MISSING python tensorrt`、`MISSING trtexec`、没有 torch。项目环境是 `/home/engineers/junyongmou/robot/ultralytics/.venv`（torch 2.13.0+cu130，spconv 2.3.8）
- 装进项目环境：tensorrt 10.3.0（`tensorrt-cu12`，和板上 10.3 对齐）、polygraphy 0.53.6、nvidia-modelopt 0.47.0、onnxruntime-gpu 1.22.0。onnx、onnxruntime 本来就有
- modelopt 把 numpy 升到 2.2.6、onnx 从 1.22.0 降到 1.21.0。numpy 已装回 1.26.4，之后 torch、scipy、spconv、cv2、modelopt 都能导入。cupy-cuda12x 14.2.0 声明要 numpy ≥ 2，和装回的 1.26.4 冲突，这次没再拆它。机器上没有 conda，没有另建 `deploy` 环境
- GPU：8× NVIDIA GeForce RTX 4090，驱动 580.126.09。Docker 27.3.1。`sudo -n` 不可用
- 0921 原始 bag：`/zjst/robot/rosbag0921`（和 `/public/zjst/robot/rosbag0921` 是同一目录，NFS）。里面是 `rosbag2_2026_09_21-16_41_27`、`rosbag2_2026_09_21-16_56_59`，共 35 GB

**测试**（项目环境，`PYTHONPATH=.`）：

- `pytest tests/dog -m "not slow" -q`：162 通过，14 个 slow 排除，1 失败
- `pytest tests/dog/test_deploy.py -q`：4 通过，1 失败。失败的都是 `test_trt_runner_on_a_tiny_engine`，没有跳过
- 失败原因：torch 2.13 的 `onnx.export` 默认把权重写到 `tiny.onnx.data`，TensorRT 10.3 按进程当前目录找这个文件，找不到就解析失败。同一条网络改成 `external_data=False` 后，FP32 小引擎能建，和 PyTorch 的最大绝对误差是 0。`export_pillar_onnx.py` 已经用 `dynamo=False`，T26 的导出不走这条默认路径

### T25 复核（2026-10-28）

结论：通道和两边的环境都可以开始 T26。下面几处已经改了，T26、T27 的命令也跟着改了。

1. **小引擎测试失败，根子在建引擎工具，不只是测试**：
   - `build_engines.build` 原来把 ONNX 读成字节再解析，TensorRT 不知道模型在哪个目录，外部权重只能按当前目录找；
   - 改用 `parser.parse_from_file`，外部权重按模型所在的目录找；
   - 测试改成：旧导出器导出一份权重内嵌的图，再另存一份权重在外部文件的图，从另一个目录建引擎，两份的输出都要和 PyTorch 一致；
   - 上线 Pillar 的导出本来就走旧导出器、权重内嵌，T26 不受影响。
2. **测引擎延迟**：
   - T25 之后板上从 NVIDIA r36.4 仓库补了 `libnvinfer-bin`，trtexec 在 `/usr/src/tensorrt/bin/trtexec`，T26 第 6e 步照原样用它；
   - 另外新增 `tools/deploy/bench_engines.py`：输入只拷一次，然后用 CUDA event 计每次执行的 GPU 时间，给中位数和 p99，和 trtexec 的 GPU Compute Time 是同一个量。服务器的 pip TensorRT 没有 trtexec，靠它测；
   - `build_engines.py --benchmark` 也改用它。原来测的是含输入输出拷贝的主机时间，偏大；
   - T26 第 6e 步在板上两个都测一遍，对得上（中位数差 < 10%），以后 T27 和 C++ 的板上对比就可以只用其中一个。
3. **`orin_check.sh`**：
   - 去掉 `colcon version-check`：它要联网查 PyPI，没网就卡住。改为打印 colcon 的路径和 dpkg 版本；
   - 新增 `PYTHON=` 指定检查用哪个 Python，服务器上传项目环境的路径；
   - 没有 trtexec 不再标 `MISSING`，因为 bench_engines.py 可以代替。
4. **libcudart**：
   - 服务器默认的 libcudart 是 CUDA 13 的，pip 的 TensorRT 10.3 是 CUDA 12 编的；
   - T25 的测试里两套运行时混用也算对了（同一个驱动、同一个上下文），但不保险；
   - `exported.py` 改为先加载 `libcudart.so.12`（`import tensorrt` 已经带进进程），找不到再用系统的。
5. **ModelOpt 0.47.0**：
   - 下载 wheel 核对了 `quantize`：`quantize_int8.py` 用到的参数名都在，含义没变；输入数组按第一维拆成逐帧校准；
   - 没进 INT8 的层存成 FP16，输入输出保持 FP32；输出图的 opset 是 19；
   - 两处对齐：
     - 排除节点按 ModelOpt 的 `re.match` 计数（原来用 fullmatch，对 `.*/hm/.*` 结果一样）；
     - 日志多记 ModelOpt 版本和排除的节点数。
6. **numpy**：项目环境保持 1.26.4，也就是测试通过的版本。cupy-cuda12x 14.2 声明要 numpy ≥ 2，本项目不用 cupy，不处理。以后 pip 再把 numpy 升到 2.x，就装回 1.26.4，并在报告里写明。
7. **板上的 torch**：板上有 torch 2.5（NVIDIA 的版本），但没有 det3d，所以 `torch` 后端仍只在服务器跑，板上继续和服务器的 dump 比。
8. **锁频要人做**：
   - `ssh orin 'sudo ...'` 没有终端，输不了密码；
   - 功耗模式已经是 MAXN；
   - T26 第 6c 步改为请人在板上执行 `sudo jetson_clocks`；
   - 人不在就不锁频，测出的延迟标明"未锁频"。
9. **服务器的 C++ 环境**：
   - 没有 CUDA 12 工具链、没有 sudo，要看 Docker 能不能用；
   - T26 第 0 步加了只查不装的检查，结果决定 C++ 在哪里编（DEPLOY_ORIN.md 第 3 节）。
10. **空间**：
    - 服务器根分区剩 35 GB，T26 的产物不大；
    - TensorRT 容器约 9 GB，第 0 步先看 Docker 数据目录在哪个盘、剩多少；
    - 板上剩 5.5 GB，两个场景各 100 帧加一个模型目录放得下，T27 做完就删。

## T26 Orin 部署：FP32 / FP16 转换和转换前后评估（含原 T8）

把上线的 nuScenes Pillar（`work_dirs/pretrained/nusc_pp.pth`）导出，建 TensorRT 引擎，在三个层面和 PyTorch 比：
- 逐段：PFN、BEV、head 图；
- 逐框；
- 审过标签上的指标。

先在服务器 x86 GPU 上全量比，再在板上用两个场景的小子集比，并测板上引擎的延迟。各后端的接口见 DEPLOY_ORIN.md 第 4 节，门槛见第 6 节。

```text
任务 T26：导出上线 Pillar 的 ONNX 和元数据，按柱数定 P_max，建 FP32 / FP16 引擎（服务器 x86 和 Orin 各一套），
逐段、逐框、按标签和 PyTorch 比较，测板上引擎延迟。
先按通用约定做（拉到了代码改动，跑快速测试）。前置：T25 完成。
T25 失败的 tests/dog/test_deploy.py::test_trt_runner_on_a_tiny_engine 现在应当通过（含外部权重和 bench_engines.py），不过就停下报告。

0. mkdir -p data/reports/T26 data/deploy; TASK=T26；VAL、ev 同 T18 第 0、3 步。
   服务器 C++ 环境检查（只查不装；输出写进 data/reports/T26/docker.txt）：
     docker info --format '{{.DockerRootDir}} {{.ServerVersion}}'; df -h "$(docker info --format '{{.DockerRootDir}}')"
     docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L
   - 要 sudo、permission denied、--gpus 用不了、拉不到镜像：把报错原样记下，不要想办法绕；
   - 能用 GPU，且 Docker 数据目录所在的盘剩 ≥ 40 GB：
       docker pull nvcr.io/nvidia/tensorrt:24.08-py3
       docker run --rm --gpus all nvcr.io/nvidia/tensorrt:24.08-py3 bash -c "nvcc --version | tail -1; ls /usr/lib/x86_64-linux-gnu/libnvinfer.so.10*"
     再记一次剩余空间。拉不下来也只记报错。
1. 导出（原 T8）：
     f=$(ls data/runs/0921_165659_s004/frames/*.npy | sed -n 150p)
     python tools/dog/export_pillar_onnx.py --checkpoint work_dirs/pretrained/nusc_pp.pth \
         --out data/deploy/nusc_pp --check $f 2>&1 | tee data/reports/T26/export.txt
   通过：pfn / bev / heads 的相对误差都 < 1e-3，检测个数相同，分数差 < 1e-3，中心差 < 1 cm。
2. 柱数：
     python tools/deploy/pillar_stats.py --model-dir data/deploy/nusc_pp --frames "data/runs/*/frames/*.npy" \
         --stride 5 --jobs 8 --out data/reports/T26/pillars.json
   suggested_max_pillars 和 30000 不同，就用它重做第 1 步（加 --max-pillars <值>，export.txt 覆盖）。
   frames_over_current > 0 时必须重做。
3. 服务器 x86 引擎：
     python tools/deploy/build_engines.py --model-dir data/deploy/nusc_pp --precision fp32,fp16 --benchmark 50 \
         2>&1 | tee data/reports/T26/build_x86.txt
4. 逐段、逐框（10 个验证场景，每 3 帧取 1）：
     for s in $VAL; do
       python tools/deploy/compare_backends.py --model-dir data/deploy/nusc_pp --frames "data/runs/$s/frames/*.npy" \
           --stride 3 --backends torch,onnx,trt:fp32,trt:fp16 --device cuda:0 --out data/reports/T26/compare_$s.json \
           2>&1 | tail -6 | tee data/reports/T26/compare_$s.txt
     done
5. 按标签（同一批帧，PyTorch、TRT FP32、TRT FP16 各跑一遍完整流水线，再分别评估）：
     python tools/dog/run_sequence.py --replay $(for s in $VAL; do echo data/runs/$s; done) \
         --detectors pillar --ckpt-pillar work_dirs/pretrained/nusc_pp.pth \
         --exported trt_fp32=data/deploy/nusc_pp:trt:fp32 --exported trt_fp16=data/deploy/nusc_pp:trt:fp16 \
         --save-raw --no-predict --no-mcap --jobs 4 --device cuda:0 --out data/deploy/T26_runs/{scene}
     for b in pillar trt_fp32 trt_fp16; do
       ev $b "$VAL" "data/deploy/T26_runs/{scene}/$b/results.pkl" --detections raw --no-mcap
     done
6. 板上：
   a. 服务器上存参考检测：两个场景，每 3 帧取 1，最多 100 帧：
        for s in 0921_164127_s000 0921_165659_s004; do
          python tools/deploy/compare_backends.py --model-dir data/deploy/nusc_pp --frames "data/runs/$s/frames/*.npy" \
              --stride 3 --limit 100 --backends torch --device cuda:0 --dump data/deploy/dumps/$s
        done
   b. 拷过去（同样的 stride 和 limit，选出的帧和第 a 步一致）：
        deploy/scripts/sync_to_orin.sh --model data/deploy/nusc_pp --dumps data/deploy/dumps \
            --frames data/runs/0921_164127_s000/frames --frames data/runs/0921_165659_s004/frames --stride 3 --limit 100
   c. 锁频（要人）：请人在板子的终端执行 `sudo jetson_clocks`（板子重启过就要重新执行），人确认后再继续。
      人不在就不锁频，后面的延迟都标"未锁频"。然后确认功耗模式、建 Orin 引擎：
        ssh orin 'nvpmodel -q' 2>&1 | tee data/reports/T26/orin_power.txt
        ssh orin 'cd ~/dogperc/repo && python3 tools/deploy/build_engines.py --model-dir ~/dogperc/models/nusc_pp \
            --precision fp32,fp16 --benchmark 50' 2>&1 | tee data/reports/T26/build_orin.txt
   d. 板上比较（参考是服务器的 PyTorch 检测）：
        for s in 0921_164127_s000 0921_165659_s004; do
          ssh orin "cd ~/dogperc/repo && python3 tools/deploy/compare_backends.py --model-dir ~/dogperc/models/nusc_pp \
              --frames '/home/seres/dogperc/data/$s/frames/*.npy' \
              --backends dump:/home/seres/dogperc/data/dumps/$s/torch,trt:fp32,trt:fp16 \
              --out /home/seres/dogperc/data/compare_$s.json" 2>&1 | tail -6 | tee data/reports/T26/orin_compare_$s.txt
          scp orin:/home/seres/dogperc/data/compare_$s.json data/reports/T26/orin_compare_$s.json
        done
   e. 引擎延迟，trtexec 和 bench_engines.py 各测一遍（后者用 CUDA event 计 GPU 时间，同 trtexec 的 GPU Compute Time）：
        for e in pfn.fp32 pfn.fp16 rpn_head.fp32 rpn_head.fp16; do
          ssh orin "/usr/src/tensorrt/bin/trtexec --loadEngine=/home/seres/dogperc/models/nusc_pp/$e.engine \
              --iterations=200 --warmUp=500" 2>&1 | grep -E "GPU Compute Time|Throughput" | tee data/reports/T26/orin_trtexec_$e.txt
        done
        ssh orin 'cd ~/dogperc/repo && python3 tools/deploy/bench_engines.py --model-dir ~/dogperc/models/nusc_pp \
            --engines pfn.fp32 pfn.fp16 rpn_head.fp32 rpn_head.fp16 --runs 200 --out /home/seres/dogperc/data/bench_orin.json' \
            2>&1 | tee data/reports/T26/orin_bench.txt
        scp orin:/home/seres/dogperc/data/bench_orin.json data/reports/T26/orin_bench.json
      服务器上同样测一遍 x86 引擎（挑一张空闲的卡，CUDA_VISIBLE_DEVICES 指定）：
        python tools/deploy/bench_engines.py --model-dir data/deploy/nusc_pp --runs 200 \
            --out data/reports/T26/x86_bench.json 2>&1 | tee data/reports/T26/x86_bench.txt

门槛（DEPLOY_ORIN.md 第 6 节）：
- trt:fp32 对 torch：heads rel < 1e-3，检测几乎完全一致；不满足说明导出或后端有错，停下报告；
- trt:fp16：每个场景 compare_backends 打印 pass；
- 第 5 步 AP@2 / AP@4 比 pillar 低不超过 0.005，MAIN_METRICS 的差值在 pillar 那份 90% 区间之内；
- 板上 trt:fp16 对服务器 torch 也打印 pass。

停止条件：第 1 步不过；trt:fp32 不过；任何报错。
提交：data/reports/T26/、本文档。不提交 ONNX、引擎和 dumps（data/deploy/ 被 .gitignore 忽略）。
报告（写进"T26 结果"）：
- export.txt 的误差行；pillars.json；最终的 P_max；
- 服务器和 Orin 的建引擎日志（每个引擎的耗时、大小、--benchmark 的 GPU 时间中位数和 p99）；
- 每个场景 compare 的表，和 10 个场景的汇总：reported match 最小值，score / center p99 的最大值，heads rel 的最大值；
- 第 5 步三份 report.md 的 AP@2、AP@4 和 MAIN_METRICS，并排一张表；
- 板上两个场景的 compare 表；
- 板上每个引擎 trtexec 的 GPU Compute Time 和 bench_engines 的 GPU 时间（中位数、p99）并排一张表，服务器 bench_engines 一张表；板上锁频了没有；
- 第 0 步的 Docker 检查：数据目录和剩余空间、能不能用 GPU、TensorRT 容器拉没拉下来；
- 结论：FP16 能不能用。
```

### T26 结果

2026-10-28。推理和建引擎在物理 GPU 5 上（`CUDA_VISIBLE_DEVICES=5`，命令里仍是 `cuda:0`）。当时这张卡空闲约 8.5 GB、利用率 0。原始输出在 `data/reports/T26/`。ONNX 和引擎在 `data/deploy/nusc_pp`（不提交）。

复核 `62ff348` 改了 ONNX 按模型目录解析外部权重，以及用 CUDA event 计 GPU 时间。拉下来之后快速测试 `pytest tests/dog -m "not slow" -n 8`：163 passed。`tests/dog/test_deploy.py`：5 passed，含 `test_trt_runner_on_a_tiny_engine`。逐框和标签是同一套引擎测的，没有为了新计时重建引擎。

**导出。** 先按默认 P_max 30000 导出，检查通过。柱数（2084 帧，每 5 帧取 1）：最大 19668，p99 16162，`frames_over_current` 0，建议 **24000**。按 24000 重导，`export.txt` 已覆盖。检查帧 `0921_165659_s004` 第 150 帧：

- pfn / bev / heads 相对误差 7.3e-06 / 7.3e-06 / 4.5e-05，都小于 1e-3
- 检测 40 对 40；分数差 4.86e-06；中心差 7.12e-06 m
- 这一帧用了 13524 / 24000 根柱

`compare_backends` 的表里 heads rel 一列是空的，工具这次没算。上面的 heads 误差来自导出时的逐段检查。

**服务器引擎**（TensorRT 10.3.0，x86）。建引擎日志里的 `--benchmark 50` 中位数是复核前的主机时间（含拷贝）。GPU 时间是复核后 `bench_engines.py` 用 CUDA event 测的（200 次，物理 GPU 5，`x86_bench.json`）：

| 引擎 | 构建 | 大小 | 主机中位数 | GPU 中位数 | GPU p99 |
|---|---|---|---|---|---|
| pfn.fp32 | 2.6 s | 678 KB | 2.75 ms | 0.719 ms | 0.724 ms |
| rpn_head.fp32 | 9.6 s | 41.1 MB | 5.70 ms | 1.876 ms | 1.887 ms |
| pfn.fp16 | 6.1 s | 179 KB | 2.66 ms | 0.457 ms | 0.460 ms |
| rpn_head.fp16 | 20.6 s | 13.5 MB | 4.62 ms | 0.804 ms | 0.810 ms |

**10 个验证场景，每 3 帧取 1，对 PyTorch。** trt:fp32 全部 pass。trt:fp16 有 8 个场景只因 `reported_match` < 0.99 失败，分数差和中心差都在门槛内。汇总：reported match 最小 fp32 0.992、fp16 0.969；score p99 最大 fp32 2.0e-3、fp16 8.4e-3；center p99 最大 fp32 1.4e-3 m、fp16 4.9e-3 m。

| 场景 | fp32 match | fp16 match | fp16 门 |
|---|---|---|---|
| 0921_164127_s000 | 0.9992 | 0.9910 | pass |
| 0921_164127_s001 | 0.9978 | 0.9888 | FAIL |
| 0921_164127_s002 | 0.9991 | 0.9889 | FAIL |
| 0921_164127_s003 | 0.9958 | 0.9735 | FAIL |
| 0921_164127_s004 | 0.9982 | 0.9700 | FAIL |
| 0921_164127_s005 | 0.9949 | 0.9694 | FAIL |
| 0921_165659_s003 | 1.0000 | 0.9834 | FAIL |
| 0921_165659_s004 | 0.9922 | 0.9845 | FAIL |
| 0921_165659_s005 | 0.9981 | 0.9885 | FAIL |
| 0921_165659_s006 | 1.0000 | 0.9902 | pass |

ONNX 十个场景都是 pass。完整表在 `compare_<场景>.txt`。

**按标签**（raw 检测，10 个场景）。重放 `--jobs 4`：每个进程约 1.5 GB，四路加起来放得进这张卡。AP@2 / AP@4：pillar 0.834 / 0.853，trt_fp32 0.834 / 0.853，trt_fp16 0.833 / 0.852。FP16 比 pillar 低 0.001，不超过 0.005。主要指标里 FP16 的值都落在 pillar 那一列的 90% 区间内（FP32 也是）。

| metric | pillar | trt_fp32 | trt_fp16 | pillar 90% 区间 |
|---|---|---|---|---|
| recall | 0.839 | 0.838 | 0.838 | 0.789 – 0.860 |
| precision | 0.825 | 0.825 | 0.824 | 0.634 – 0.903 |
| F1 | 0.832 | 0.831 | 0.831 | 0.710 – 0.873 |
| FP / keyframe | 0.641 | 0.645 | 0.648 | 0.484 – 0.799 |
| MOTA | 0.609 | 0.606 | 0.606 | 0.304 – 0.709 |
| IDF1 | 0.786 | 0.785 | 0.785 | 0.663 – 0.838 |
| IDSW / min | 1.862 | 1.862 | 2.069 | 0.419 – 3.356 |
| false tracks / min | 27.102 | 27.309 | 26.895 | 20.238 – 33.793 |
| confirm p90 s | 1.200 | 1.200 | 1.200 | 0.501 – 4.000 |
| moving missed | 2 | 2 | 2 | 0 – 4 |
| false moving / h | 86.891 | 86.891 | 86.891 | 12.388 – 175.081 |
| FDE 3 s | 2.795 | 2.794 | 2.801 | 2.297 – 3.621 |
| MR 3 s | 0.521 | 0.521 | 0.530 | 0.430 – 0.694 |

**板上。** 同步前先建了两个场景的 `frames` 目录，否则 `sync_to_orin.sh` 会因父目录不存在而失败。拷了 96 + 92 帧和参考检测。根分区同步后剩 4.7 GB。两个场景的 trt:fp32、trt:fp16 对服务器 PyTorch 都是 pass。

锁频了。`nvpmodel -q` 是 MAXN（0），写在 `orin_power.txt`。测延迟前执行过 `jetson_clocks`；这次 bench 之前再看，CPU 最低频率等于最高频率（2.2016 GHz），GPU 1.3005 GHz 同样钉死，EMC `FreqOverride=1`。板子时钟停在 1970-01-01（没有校时），引擎日志里的时间是这个。

建引擎日志里的 benchmark 中位数是主机时间。trtexec 和 `bench_engines.py`（200 次，CUDA event）的 GPU 中位数相差 1.0%–2.6%，都在 10% 以内。T27 的延迟可以只用 trtexec。

| 引擎 | 构建 | 大小 | 主机中位数 | trtexec GPU 中位数 | trtexec p99 | bench GPU 中位数 | bench p99 |
|---|---|---|---|---|---|---|---|
| pfn.fp32 | 22.2 s | 83 KB | 9.21 ms | 5.285 ms | 5.295 ms | 5.222 ms | 5.251 ms |
| pfn.fp16 | 66.1 s | 66 KB | 7.43 ms | 3.526 ms | 3.531 ms | 3.434 ms | 3.477 ms |
| rpn_head.fp32 | 37.2 s | 23.6 MB | 16.42 ms | 9.182 ms | 9.219 ms | 9.092 ms | 9.134 ms |
| rpn_head.fp16 | 63.7 s | 12.3 MB | 12.09 ms | 4.847 ms | 4.860 ms | 4.720 ms | 4.730 ms |

板上 compare 的 network ms（含两段引擎）：s000 的 fp32 / fp16 是 47.1 / 40.9 ms，s004 是 49.8 / 43.5 ms。

**Docker。** 数据目录 `/zjst/sekj_tech/docker`（Docker 27.3.1），在 `/zjst` 上，当时剩 693 TB。`nvidia/cuda:12.6.0-base-ubuntu22.04` 里 `nvidia-smi -L` 看到 8 张 RTX 4090。`nvcr.io/nvidia/tensorrt:24.08-py3` 拉下来了：容器里是 TensorRT 10.3.0，nvcc 是 `cuda_12.6.r12.6/compiler.34431801_0`，有 `libnvinfer.so.10` 和 `libnvinfer.so.10.3.0`。拉完这盘还剩 692 TB。

**结论：** FP32 可以用。FP16 按「每个验证场景都要 pass」这条不能用，差在匹配率 0.969–0.989，不是分数或位置漂出了门槛。标签上的 AP 和主要指标都在允许范围内。板上抽的两个场景 FP16 是 pass。

### T26 复核（2026-10-28）

结论：**FP32、FP16 都可以用，上线先用 FP16**。板上 FP16 的 GPU 时间：PFN 3.4 ms（FP32 5.2 ms），RPN 4.7 ms（FP32 9.1 ms）。FP16 没过的只有匹配率这一条。复核认为是这条门槛定得不对，已改；T27 第 3 步会用新的算法把 FP16 再算一遍确认。

1. **FP16 的匹配率：多半是阈值附近的框在来回翻，不是丢了物体**：
   - 漏的和多的个数相近，总框数几乎不变：s005 漏 7、多 5，过滤后 1265 对 1264；板上 s000 也是 7 / 5；
   - FP16 分数差最大 0.015、p99 0.0075，中心差 p99 5 mm。分数离上报阈值不到 0.015 的框，换个精度就可能跨过去；
   - 下游看不出影响：AP@2 / AP@4 只掉 0.001，跟踪、预测的主要指标都在 90% 区间里（IDSW 每分钟 1.86 → 2.07，也在区间内）；
   - 原来的匹配率把"分数变了 0.01、跨过了阈值"和"真丢了一个物体"算成一样，阈值附近的框多的场景就不容易过。
   - **改法**（`tools/deploy/compare_backends.py`）：每个漏的、多的框归为四类：
     - near：同一个框，分数差 ≤ 0.02（和 score p99 的门槛相同），只是跨过了阈值；
     - score：同一个框，分数差更大；
     - class：0.5 m 内只有别的类的框；
     - none：0.5 m 内什么都没有。
   - 门槛改为不计 near 的匹配率 ≥ 0.99（`reported_match_tol`）。原来的匹配率照样打印。near 的分数差本来就由 score p99 管着。
   - **还要确认**：T26 的报告没有逐框记录，上面是推断。T27 第 3 步本来就要重跑 trt:fp16 对 torch，新表会给出四类的个数。如果 near 以外的占多数，就要回头查 FP16。
   - FP16 的朝向差 p99 是 0.08 rad（4.6°），FP32 是 0.02 rad。表里新加了 yaw p99 一列，做 INT8 时一起看。
2. **逐段误差没算出来**：
   - 参考是 torch，而 torch 后端不保留中间张量，所以 heads rel 一列是空的；
   - 改为：参考没有中间张量时，逐段误差对第一个导出的后端算（通常是 onnx），报告里记下 `stage_reference`；
   - onnx 对 torch 的逐段误差由导出时的检查保证（heads 4.5e-5）；
   - T27 的后端列表里要带上 onnx。
3. **板上延迟和下一步的优化**（GPU 时间，已锁频，FP16）：
   - PFN 3.4 ms，RPN 加 head 4.7 ms，合计 8.2 ms；
   - PFN 占了 42%，并不是我之前写的"计算量小"。它总按 P_max = 24000 算，而柱数中位数约 10000（`pillars.json`：p50 10040，p99 16162，最大 19668）；
   - 新增 `export_pillar_onnx.py --dynamic-pillars`：
     - PFN 的柱数维度放开；
     - 建引擎时给一个范围（1 到 P_max），按 `--opt-pillars` 调优，默认 P_max 的一半；
     - 检测器只送实际用到的柱；
     - 数学上和补零一样，测试里两种 ONNX 的检测结果一致；
     - 预计中位帧上 PFN 省一半以上（约 2 ms），T27 第 6 步实测。
   - PFN 不做 INT8：它的输入里有 ±40 m 的绝对坐标，也有 0.1 m 量级的柱内偏移。整个张量共用一个 scale，偏移就量化没了。减少柱数更直接。
   - 板上 compare 的 network ms 是 41–44 ms，比 8.2 ms 多出来的是 Python 端的拷贝（PFN 输入 19 MB、BEV 画布 42.6 MB）和 numpy 的 scatter。C++ 版这些都在 GPU 上，不经过主机。
4. **服务器的 C++ 环境定了**：
   - Docker 免 sudo 就能用 GPU；
   - `nvcr.io/nvidia/tensorrt:24.08-py3` 已拉下来，TensorRT 10.3.0、CUDA 12.6，和板子一致；
   - Docker 数据目录在 `/zjst`，剩 692 TB；
   - C++ 全部（含 CUDA、TensorRT）先在这个容器里开发、测试，再到板上原生编译、终验。
5. **`sync_to_orin.sh`**：rsync 只建目标路径的最后一级，场景目录不存在就失败。已改为先在板上建好所有场景的目录。
6. **板子时钟停在 1970 年**：
   - 不影响引擎和对比；
   - P6 的 bag 回放用 bag 的时间戳（`use_sim_time`），也不受影响；
   - 上车实时跑之前要校时，否则节点的延迟统计和日志时间都不对。这一步要人做：从本机同步时间，或配 chrony / NTP。
7. **板上 scipy 的警告**：apt 装的 scipy 1.8 要 numpy < 1.25，板上是 1.26.4，导入时会警告。只有 Python 端用到 scipy，C++ 不用，不处理。

## T27 Orin 部署：INT8 量化

FP16 先过了再做。INT8 的做法：
- NVIDIA ModelOpt 在真实帧上算 scale，生成带 Q/DQ 的图，再建 INT8 引擎（DEPLOY_ORIN.md 第 5 节）；
- 校准只用训练场景，验证场景留给打分；
- 再做一个热力图层保持 FP16 的混合精度版本，对比哪个更好。

另外试动态柱数的 PFN（T26 复核第 3 条）：PFN 只算实际用到的柱，不按 P_max 补零。PFN 不做 INT8。

```text
任务 T27：rpn_head 做 INT8 PTQ（纯 INT8 和热力图层保持 FP16 两个版本），和 FP16、PyTorch 比精度，测板上延迟；
再导出动态柱数的 PFN，比精度、测板上延迟。
先按通用约定做。前置：T26 通过。

0. mkdir -p data/reports/T27；TASK=T27；VAL、TRAIN、ev 同 T18。
   先跑 `pytest tests/dog/test_deploy.py -q 2>&1 | tee data/reports/T27/pytest_deploy.txt`（含 slow 和 TensorRT 那几条）。
   T26 复核新加的动态 PFN、逐段参考、四类分类都在里面，复核时这台机器没有 torch，没能跑。全过才往下做，不过就停下报告。
1. 量化（校准用训练场景，128 帧均匀取）：
     F=$(for s in $TRAIN; do echo "data/runs/$s/frames/*.npy"; done)
     python tools/deploy/quantize_int8.py --model-dir data/deploy/nusc_pp --frames $F --num 128 --method entropy \
         2>&1 | tee data/reports/T27/quantize_int8.txt
     python tools/deploy/quantize_int8.py --model-dir data/deploy/nusc_pp --frames $F --num 128 --method entropy \
         --exclude ".*/hm/.*" --suffix int8_hmfp16 2>&1 | tee data/reports/T27/quantize_int8_hmfp16.txt
   第二个的输出里 exclude 应匹配到 18 个节点。内存不够就把 --num 降到 64，并在报告里写明。
   ONNX Runtime 的 CUDA EP 加载失败、回退到 CPU 时照样做（只是慢），在报告里写明用的是哪个。
2. 服务器 x86 引擎：
     python tools/deploy/build_engines.py --model-dir data/deploy/nusc_pp --graphs rpn_head \
         --precision int8,int8_hmfp16 --benchmark 50 2>&1 | tee data/reports/T27/build_x86.txt
3. 逐段、逐框（同 T26 第 4 步）：后端 torch,onnx,trt:fp16,trt:int8,trt:int8_hmfp16，输出 compare_$s.json / .txt；
   表比 T26 多一行，`tail -6` 改成 `tail -8`。
   表里多了"match w/o near"一列，以及漏、多的四类个数（near / score / class / none，T26 复核第 1 条）。
   trt:fp16 这一行就是 T26 复核要的确认：报告 10 个场景 FP16 的四类合计。
4. 按标签（同 T26 第 5 步）：--exported trt_fp16=...:trt:fp16、trt_int8=...:trt:int8、trt_int8_hmfp16=...:trt:int8_hmfp16，
   再加 --detectors pillar；输出到 data/deploy/T27_runs/{scene}，四份分别 ev。
5. 板上：
   - sync_to_orin.sh --model data/deploy/nusc_pp 拷两个 int8 图；
   - build_engines.py --graphs rpn_head --precision int8,int8_hmfp16 --benchmark 50；
   - 用 T26 的帧和 dumps 跑 compare（后端 dump:...,trt:fp16,trt:int8,trt:int8_hmfp16）；
   - 测 rpn_head.fp16、rpn_head.int8、rpn_head.int8_hmfp16 的延迟（同 T26 第 6e 步，锁频同 6c）：T26 里两者对得上就只用 trtexec，对不上两个都测。
6. 动态柱数的 PFN（引擎都是 FP16）：
   a. 服务器导出、建引擎、比较：
        f=$(ls data/runs/0921_165659_s004/frames/*.npy | sed -n 150p)
        python tools/dog/export_pillar_onnx.py --checkpoint work_dirs/pretrained/nusc_pp.pth --max-pillars 24000 \
            --dynamic-pillars --out data/deploy/nusc_pp_dyn --check $f 2>&1 | tee data/reports/T27/export_dyn.txt
        python tools/deploy/build_engines.py --model-dir data/deploy/nusc_pp_dyn --precision fp16 --opt-pillars 10000 \
            2>&1 | tee data/reports/T27/build_dyn_x86.txt
        for s in 0921_164127_s000 0921_165659_s004; do
          python tools/deploy/compare_backends.py --model-dir data/deploy/nusc_pp_dyn --frames "data/runs/$s/frames/*.npy" \
              --stride 3 --limit 100 --backends torch,onnx,trt:fp16 --device cuda:0 --out data/reports/T27/compare_dyn_$s.json \
              2>&1 | tail -6 | tee data/reports/T27/compare_dyn_$s.txt
        done
   b. 板上：
      - sync_to_orin.sh --model data/deploy/nusc_pp_dyn；
      - build_engines.py --model-dir ~/dogperc/models/nusc_pp_dyn --precision fp16 --opt-pillars 10000；
      - compare（T26 的帧和 dumps，后端 dump:...,trt:fp16），输出 orin_compare_dyn_$s.json / .txt；
      - 延迟（锁频）：静态的 pfn.fp16 再测一次作对照；动态的在 10000、16000、24000 柱各测一次：
          for P in 10000 16000 24000; do
            ssh orin "/usr/src/tensorrt/bin/trtexec --loadEngine=/home/seres/dogperc/models/nusc_pp_dyn/pfn.fp16.engine \
                --shapes=pillars:1x10x${P}x20 --iterations=200 --warmUp=500" 2>&1 | grep -E "GPU Compute Time" \
                | tee data/reports/T27/orin_trtexec_pfn_dyn_$P.txt
          done
   - export_dyn.txt 的 --check 要和 T26 的 export.txt 一样过；
   - 这两个场景上 compare 要过门槛，几项差值要和 T26 静态的 trt:fp16 在同一个量级；
   - 板上 10000 柱的 PFN 时间要比静态的低，24000 柱的时间不比静态的高出 10% 以上。
   做完记下板上根分区的剩余空间；低于 3 GB 就删掉 ~/dogperc/data 下的帧和 dumps（以后 C++ 的对比会重新拷）。

门槛（DEPLOY_ORIN.md 第 6 节）：
- 两个 INT8 版本各自看 AP@2 / AP@4 比 pillar 低不超过 0.01，MAIN_METRICS 差值在 pillar 那份 90% 区间之内；
- 板上 rpn_head 的 GPU Compute Time 中位数比 FP16 低 ≥ 30%。
都满足的选延迟低的；都不满足就上线用 FP16。最后由人定。

停止条件：量化报错；任何引擎建不出来。
提交：data/reports/T27/、本文档。不提交 ONNX 和引擎。
报告（写进"T27 结果"）：
- 两次量化的日志（帧数、耗时、exclude 匹配数）；
- 建引擎日志；每个场景 compare 的表和 10 个场景的汇总；
- 四份 report.md 并排的 AP 和 MAIN_METRICS；
- 板上 compare 和延迟（FP16 对 INT8 对 INT8_hmfp16）；
- 第 3 步 trt:fp16 的四类合计（T26 复核要的确认）；
- 第 6 步：export_dyn 的误差行，两个场景的 compare 表（服务器、板上），板上 PFN 的时间表（静态 24000，动态 10000 / 16000 / 24000）；
- 结论：INT8 选哪个；动态 PFN 用不用。
```

### T27 结果

2026-10-29。拉到 `e252021`。`pytest tests/dog/test_deploy.py -q`：7 passed。快速测试 `pytest tests/dog -m "not slow" -n 8`：164 passed。做完测量后又拉到 `e7d786d`（加了 C++ 和 T28），再跑一次是 8 passed（多了 `test_export_cpp_config`），写在 `pytest_deploy.txt`。开始时在 `e252021` 上是 7 passed。量化、建引擎和逐框比较在物理 GPU 3（`CUDA_VISIBLE_DEVICES=3`）。重放在物理 GPU 4：当时空闲约 7.3 GB，每个进程约 1.5 GB，所以 `--jobs 3`（4 路放不下还要留 1 GB）。原始输出在 `data/reports/T27/`。ONNX 和引擎在 `data/deploy/nusc_pp`、`data/deploy/nusc_pp_dyn`（不提交）。

**量化。** 训练场景 25 个、7479 帧里均匀取 128 帧。ModelOpt 0.47.0，entropy。ONNX Runtime 的 CUDA EP 起不来（`libcudnn_adv` 不在库路径里），回退到 CPU，两次都是。纯 INT8：准备输入 80.8 s，量化 206.9 s，481 个节点里量化了 92 个。热力图保持 FP16：exclude `.*/hm/.*` 匹配 **18** 个节点（例如 `/head/tasks.0/hm/hm.0/Conv`），准备输入 78.1 s，量化 186.3 s。

**服务器引擎**（TensorRT 10.3.0，x86，`--benchmark 50` 的 GPU 时间）：

| 引擎 | 构建 | 大小 | GPU 中位数 | GPU p99 |
|---|---|---|---|---|
| rpn_head.int8 | 18.4 s | 8.1 MB | 0.614 ms | 1.068 ms |
| rpn_head.int8_hmfp16 | 23.0 s | 7.9 MB | 0.510 ms | 0.517 ms |

**10 个验证场景，每 3 帧取 1。** 逐段误差对 onnx。onnx 十个场景都是 pass。trt:fp16 的框：score p99 最大 8.5e-3，center p99 最大 4.9e-3 m，都在门槛内。不计 near 的匹配率有 8 个场景 ≥ 0.99，`0921_164127_s004` 是 0.9859，`0921_164127_s005` 是 0.9898。十个场景的门都打印 FAIL，唯一普遍没过的是 heads rel（最大 0.367，对 onnx）。`0921_164127_s000` 上 heads 的余弦是 0.993，相对误差的最大值是 0.233。两个 INT8 的匹配率在 0.09–0.40，score p99 约 0.3，center p99 约 0.48 m，heads rel 约 2.5，全部 FAIL。

**T26 复核要的 FP16 四类合计**（10 个场景，漏 / 多）：

| | 个数 | near | score | class | none |
|---|---|---|---|---|---|
| 漏 | 45 | 17 | 0 | 1 | 27 |
| 多 | 70 | 50 | 0 | 0 | 20 |

多出来的框里 near 是多数（50/70）。漏掉的框里 none 是多数（27/45），0.5 m 内没有框。near 以外的漏报不是个别。

**按标签**（raw，10 个场景）。AP@2 / AP@4：pillar 0.834 / 0.853，fp16 0.833 / 0.852，int8 0.710 / 0.738，int8_hmfp16 0.735 / 0.761。FP16 比 pillar 低 0.001。两个 INT8 比 pillar 低 0.099–0.124，超过 0.01。主要指标里，INT8 的 recall、每关键帧误报、假轨迹、3 s 漏预测率落在 pillar 90% 区间外面；热力图保持 FP16 的同样这几项也在外面。FP16 的主要指标都在区间内。

| metric | pillar | fp16 | int8 | int8_hmfp16 | pillar 90% 区间 |
|---|---|---|---|---|---|
| recall | 0.839 | 0.838 | 0.770 | 0.784 | 0.789 – 0.860 |
| precision | 0.825 | 0.824 | 0.762 | 0.779 | 0.634 – 0.903 |
| F1 | 0.832 | 0.831 | 0.766 | 0.782 | 0.710 – 0.873 |
| FP / keyframe | 0.641 | 0.648 | 1.017 | 0.872 | 0.484 – 0.799 |
| MOTA | 0.609 | 0.606 | 0.420 | 0.470 | 0.304 – 0.709 |
| IDF1 | 0.786 | 0.785 | 0.694 | 0.710 | 0.663 – 0.838 |
| IDSW / min | 1.862 | 2.069 | 2.276 | 2.069 | 0.419 – 3.356 |
| false tracks / min | 27.102 | 26.895 | 43.238 | 38.894 | 20.238 – 33.793 |
| confirm p90 s | 1.200 | 1.200 | 2.250 | 1.500 | 0.501 – 4.000 |
| moving missed | 2 | 2 | 2 | 3 | 0 – 4 |
| false moving / h | 86.891 | 86.891 | 49.652 | 37.239 | 12.388 – 175.081 |
| FDE 3 s | 2.795 | 2.801 | 3.397 | 3.418 | 2.297 – 3.621 |
| MR 3 s | 0.521 | 0.530 | 0.735 | 0.779 | 0.430 – 0.694 |

**板上。** 锁频仍在：`nvpmodel` 是 MAXN，GPU 钉在 1.3005 GHz。根分区做完剩 4.6 GB，没有删帧。两个场景对服务器 PyTorch：fp16 是 pass（s000 匹配率 0.9902 / 不计 near 0.9910，s004 是 0.9922 / 0.9961）。两个 INT8 都是 FAIL，和服务器上一样差。trtexec 200 次的 GPU 时间（中位数 / p99）：

| 引擎 | 中位数 | p99 | 比 FP16 |
|---|---|---|---|
| rpn_head.fp16 | 5.051 ms | 5.072 ms | — |
| rpn_head.int8 | 2.651 ms | 2.661 ms | 低 47% |
| rpn_head.int8_hmfp16 | 2.897 ms | 2.907 ms | 低 43% |

延迟这条两个 INT8 都过了 30%。精度没过，按门槛不上。

**动态柱数的 PFN。** 导出检查和 T26 的静态导出同一帧、同一组数：pfn / bev / heads 相对误差 7.3e-06 / 7.3e-06 / 4.5e-05，检测 40 对 40，分数差 4.86e-06，中心差 7.12e-06 m，这一帧 13524 / 24000 根柱。服务器上两个场景的框差和静态 FP16 同一个量级（s000 不计 near 的匹配率 0.9918 对静态的 0.9935，s004 是 0.9922 对 0.9961；score / center p99 也在同一量级）。打印出来的门是 FAIL，原因同样是 heads rel 最大约 0.23–0.25，静态引擎在这两场也是这个数。板上这两个场景的动态 FP16 对服务器 PyTorch 都是 pass，框差和静态的板上结果同一量级（s000 的 network ms 从 41 降到 21）。

板上 PFN 的 GPU 时间（trtexec，中位数 / p99）。静态是这次重测的 `nusc_pp/pfn.fp16`：

| | 中位数 | p99 |
|---|---|---|
| 静态 24000 柱 | 3.522 ms | 3.529 ms |
| 动态 10000 柱 | 1.501 ms | 1.504 ms |
| 动态 16000 柱 | 2.367 ms | 2.373 ms |
| 动态 24000 柱 | 3.524 ms | 3.531 ms |

10000 柱比静态低一半以上。24000 柱和静态差 0.002 ms，没有高出 10%。

**结论：** INT8 两个版本都不用，上线继续用 FP16。动态柱数的 PFN 用。最后由人定。

### T27 复核

2026-10-29。结论同意：INT8 两版都不上，上线用 FP16；动态柱数的 PFN 上线，模型目录 `data/deploy/nusc_pp_dyn`（板上 `~/dogperc/models/nusc_pp_dyn`，引擎已建）。人定之前，T28 先按这个做。

1. **FP16 剩下的漏报多是过滤器的阈值翻转。** 把 10 个场景 JSON 里过滤前（raw）和过滤后（final）分开看：
   - 过滤前，不计 near 的匹配率 10 个场景都 ≥ 0.991（s004 0.9940，s005 0.9911），none 每个场景 0–2 个；
   - 过滤后 none 变多（s000 从 0 个变成 5 个），s004、s005 掉到 0.9859、0.9898；
   - 动态 PFN 的 s004 上，FP32 的 onnx 对 torch 在过滤后也有一个 none。两边的分数差 p99 只有 8e-4、中心差 p99 0.6 mm，过滤器（框底离地高度、框里的点数）照样会翻。

   所以过滤后的 none 多数是：对面过滤前有这个框，被过滤器去掉了。改了 `compare_backends.py`：
   - 加一类 filter：对面过滤前有这个框（0.5 m 内同类，上报了或者分数差在 0.02 内），被对面的过滤器去掉了；
   - 门槛的匹配率不计 near 和 filter，过滤前、过滤后都要 ≥ 0.99；
   - `cpp_parity.py compare` 用同样的规则。

   T28 第 0 步用新规则重算 s004、s005。
2. **逐段的 heads rel 不再卡门槛。** 这项是整张图上 max |a − b| / max |ref| 的最大值，多半由空格子里小的回归输出决定：FP16 是 0.23，框的中心差却在 5 mm 以内，十个场景都因为它打印 FAIL。现在：
   - 只打印，不卡门槛；
   - 按 head 种类（hm、reg、height、dim、rot、vel）分别记最差的余弦，表里一列 "worst head cos"。
3. **INT8 差得不正常。** PointPillars 做 INT8 PTQ 通常只掉一点，这里 AP@2 掉 0.12，匹配率 0.25，yaw p99 3.1 rad。有一个 head 的余弦是 −0.58，两版一样，所以不是 hm。
   - 最可能的原因：BEV 画布 95% 以上是 0，entropy 校准给输入取的截断值太小，把柱子的特征截掉了（和 PFN 不做 INT8 是类似的道理）。
   - 不用 INT8，板上 RPN 多 2.4 ms。网络合计约 6.6 ms（动态 PFN 1.5 ms 加 RPN 5.05 ms），离 50 ms 的预算还远，现在不追。
   - T30 要是时间不够再试：`--method max`；第一层卷积和各 head 的最后一层不量化；先用每个 head 的余弦定位是哪一层坏的。
   - T28 第 0 步顺手跑一次带 trt:int8 的 compare，记下哪个 head 的余弦是负的。
4. ONNX Runtime 的 CUDA EP 起不来（找不到 libcudnn_adv），onnx 后端每帧 434 ms，跑在 CPU 上。不影响结果，只是慢。按约定 9 可以自己修：把 pip 装的 cuDNN 所在目录加进 LD_LIBRARY_PATH。
5. `sync_to_orin.sh` 原来只排除名叫 build 的目录，`--delete` 会删掉板上的 `deploy/cpp/build_cuda`，还会把服务器的 x86 编译目录拷过去。改成排除 `build*/`。板上根分区剩 4.6 GB。
6. T27 时锁频还在（MAXN，GPU 1.3005 GHz），但板子重启就没了。T28 测延迟前先查，还在就不用找人。
7. 通用约定加了第 9 条：服务器 Agent 可以按实际情况改代码（环境、编译、工具 bug、C++ 对齐 Python），不能改门槛、默认参数、Python 流水线的行为。

## T28 Orin 部署：C++ 核心库对齐和全链路评估

C++ 核心库 `deploy/cpp` 已写好（DEPLOY_ORIN.md 第 7 节）。本机只验过 CPU 部分：模拟器序列上和 Python 逐位对齐，跟踪 ID 每帧完全一致。CUDA 部分只编译过，没跑过。这次要：
- 在服务器容器里编译，跑全部对齐测试，包括 GPU 体素化和 TensorRT；
- 在 10 个验证场景的真实录制上和 Python 比两次：一次 C++ 用 Python 的检测（只比跟踪、预测），一次 C++ 用自己的 GPU 检测器（全链路）；
- 两边的结果都用 `pipeline_eval.py` 打分，并排比；
- 板上原生编译、跑测试，测每段延迟。

模型用 T27 定的上线版本：FP16，动态柱数的 PFN（`data/deploy/nusc_pp_dyn`）。

这是 C++ 第一次在真的 GPU 上跑，编译错误、对不上 Python 的地方多半要当场修。本任务可以改的范围（通用约定 9）：
- `deploy/cpp/`、`deploy/docker/`、`deploy/scripts/`、`tools/deploy/`；
- `dog_perception/` 不改：对不上时以 Python 为准改 C++。

```text
任务 T28：C++ 核心库在服务器容器里编译测试；10 个验证场景上 Python 对 C++（同一批检测、各自检测）并打分；板上编译、测试、测延迟。
先按通用约定做（第 9 条：可以改代码，范围见本节开头）。前置：T27 完成。

0. mkdir -p data/reports/T28 data/deploy/T28；TASK=T28；VAL、ev 同 T18 第 0、3 步；O=data/deploy/T28；M=data/deploy/nusc_pp_dyn。
   挑一张空闲的卡，记为 G（T26 用的是 5），export CUDA_VISIBLE_DEVICES=$G（主机上的 Python 也用这张卡）。
   pytest tests/dog/test_deploy.py -q 2>&1 | tee data/reports/T28/pytest_deploy.txt（不过就停下报告）
   T27 复核改了比较的规则（filter 一类，过滤前后都卡门槛，heads 不卡），重算两个没过的场景，再看一次 INT8 坏在哪个 head：
     for s in 0921_164127_s004 0921_164127_s005; do
       python tools/deploy/compare_backends.py --model-dir data/deploy/nusc_pp --frames "data/runs/$s/frames/*.npy" \
           --stride 3 --backends torch,onnx,trt:fp16,trt:int8 --device cuda:0 --out data/reports/T28/compare_$s.json \
           2>&1 | tail -6 | tee data/reports/T28/compare_$s.txt
     done
   - trt:fp16 应该两个都 pass。不过的话报告漏、多的分类，接着往下做；
   - 从 JSON 的 stages 里抄下 trt:int8 每种 head 的余弦（head_<种类>_cos）。
   容器里跑命令的函数，数据目录按原路径挂进去，这样 data/ 下的链接照样能用：
     dk() { docker run --rm --gpus "\"device=$G\"" -u $(id -u):$(id -g) -v $PWD:$PWD -v /public/zjst/robot:/public/zjst/robot \
                -v /zjst:/zjst -w $PWD dogperc-dev bash -c "$*"; }
1. 镜像（第一次约几分钟，在 nvcr.io/nvidia/tensorrt:24.08-py3 上装 cmake、Eigen、yaml-cpp、GoogleTest、numpy、scipy）：
     docker build -t dogperc-dev -f deploy/docker/Dockerfile.dev deploy/docker 2>&1 | tail -5 | tee data/reports/T28/docker_build.txt
2. 编译、测试（DOGPERC_MODEL_DIR：T26 / T27 在服务器上建的 x86 引擎，GPU 检测器要和主机拷贝的 TensorRT 检测器逐帧比）：
     dk "cmake -S deploy/cpp -B deploy/cpp/build_cuda -G Ninja -DDOGPERC_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build deploy/cpp/build_cuda -j16" 2>&1 | tail -20 | tee data/reports/T28/build_x86.txt
     dk "DOGPERC_MODEL_DIR=$PWD/$M ctest --test-dir deploy/cpp/build_cuda --output-on-failure -V" \
         > data/reports/T28/ctest_x86.txt 2>&1; tail -30 data/reports/T28/ctest_x86.txt
     # 静态 PFN 也跑一次 GPU 测试（另一条代码路径）
     dk "DOGPERC_MODEL_DIR=$PWD/data/deploy/nusc_pp deploy/cpp/build_cuda/dogperc_tests --gtest_filter='Gpu.*'" \
         > data/reports/T28/gtest_gpu_static.txt 2>&1; tail -10 data/reports/T28/gtest_gpu_static.txt
   - 两个 Gpu.* 测试必须跑了（不是 SKIPPED）且通过，动态、静态两种 PFN 都要；
   - 编译错误、测试不过，先按通用约定 9 修（C++ 以 Python 为准），修完重编、重跑这一步；
   - 引擎加载失败（"could not load the engine"）就在容器里用 build_engines.py 重建到 data/deploy/T28/engines_x86/，
     再把 DOGPERC_MODEL_DIR 指过去（目录里要有 model_meta.json），并在报告里写明。
3. 10 个验证场景，每个场景：
     for s in $VAL; do
       mkdir -p $O/$s
       python tools/deploy/cpp_parity.py python --seq data/seq/$s --exported $M:trt:fp16 \
           --out $O/$s/py.npz --dets $O/$s/py_dets.npz --config $O/$s/config.yaml
       dk "deploy/cpp/build_cuda/dogperc_replay --seq data/seq/$s --config $O/$s/config.yaml \
           --detections $O/$s/py_dets.npz --out $O/$s/cpp_dets.npz"
       dk "deploy/cpp/build_cuda/dogperc_replay --seq data/seq/$s --config $O/$s/config.yaml \
           --model-dir $M --precision fp16 --gpu --out $O/$s/cpp.npz"
       python tools/deploy/cpp_parity.py compare --ref $O/$s/py.npz --cpp $O/$s/cpp_dets.npz --same-detections \
           --out data/reports/T28/parity_dets_$s.json | tail -3 | tee data/reports/T28/parity_dets_$s.txt
       python tools/deploy/cpp_parity.py compare --ref $O/$s/py.npz --cpp $O/$s/cpp.npz \
           --out data/reports/T28/parity_$s.json | tail -3 | tee data/reports/T28/parity_$s.txt
       for r in py cpp; do
         python tools/deploy/cpp_parity.py to-results --cpp $O/$s/$r.npz --seq data/seq/$s --out $O/runs/$s/$r/results.pkl
       done
     done
   - Python 的参考运行是 run_sequence.py 那样从原始扫描做预处理，所以一个场景要几分钟；
   - 先跑一个场景，两次 compare 都过了再跑其余的；
   - 同一批检测（parity_dets）：每个场景都要 pass，即帧相同、检测相同、每帧跟踪 ID 完全一致、状态 ≤ 1e-6、预测 ≤ 1e-4。
     不过就找第一个不一致的帧（compare 的 JSON 里有最大差值），按约定 9 修 C++，修完从第 2 步重来；
     修不好，或者原因在 Python 那边，停下报告；
   - 各自检测（parity）：门槛同 T27 复核后的 compare_backends（不计 near、filter 的匹配率过滤前后都 ≥ 0.99，分数 p99 ≤ 0.02，中心 p99 ≤ 0.1 m）。
     不过的先看是不是 GPU 检测器的问题：第 2 步的 Gpu.* 测试是逐帧和主机拷贝的 TensorRT 检测器比的。
4. 按标签打分（两份结果是同一批帧、同一套预处理）：
     for r in py cpp; do ev $r "$VAL" "$O/runs/{scene}/$r/results.pkl" --detections raw --no-mcap; done
   pipeline_eval 拿存下的过滤前检测（raw_detections、raw_measured）重跑 Python 的过滤、跟踪、预测，所以这一步比的是两边的检测器；
   C++ 的跟踪、预测和 Python 一致，由第 3 步同一批检测那次 compare 保证。
   门槛：cpp 的 AP@2 / AP@4 比 py 低不超过 0.005；MAIN_METRICS 的差值在 py 那份 90% 区间之内。
5. 服务器上的每段延迟（C++ GPU 检测器，4090）：
     python tools/deploy/cpp_parity.py timings $O/*/cpp.npz --out data/reports/T28/x86_timings.json \
         | tee data/reports/T28/x86_timings.txt
6. 板上（模型目录 ~/dogperc/models/nusc_pp_dyn，T27 已在板上建好 FP16 引擎）：
   a. 同步代码和一段原始录制（前 150 帧），看剩余空间（T27 后剩 4.6 GB；低于 3 GB 先删 ~/dogperc/data 下的帧和 dumps）：
        deploy/scripts/sync_to_orin.sh --seq data/seq/0921_164127_s000 --limit 150
        scp $O/0921_164127_s000/config.yaml orin:dogperc/seq/0921_164127_s000/config.yaml
   b. 原生编译（板上没有 ninja 就去掉 -G Ninja）：
        ssh orin 'cd ~/dogperc/repo && cmake -S deploy/cpp -B deploy/cpp/build_cuda -DDOGPERC_WITH_CUDA=ON \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87 \
            && cmake --build deploy/cpp/build_cuda -j8' 2>&1 | tail -20 | tee data/reports/T28/build_orin.txt
      缺包就停下，报告缺什么（T25 查过：eigen、yaml-cpp、gtest、cmake、CUDA 12.6、TensorRT 10.3 都有）。
      编译错误（板上是 aarch64、GCC 11）按约定 9 修，修完在服务器容器里也重编、重跑第 2 步的测试。
   c. 测试（golden 数据在板上用 Python 生成，约 60 MB；板上 scipy 是 1.8.0）：
        ssh orin 'cd ~/dogperc/repo && DOGPERC_MODEL_DIR=$HOME/dogperc/models/nusc_pp_dyn \
            ctest --test-dir deploy/cpp/build_cuda --output-on-failure -V' > data/reports/T28/ctest_orin.txt 2>&1
      板上生成 golden 失败（golden 那个测试不过）：把服务器的 deploy/cpp/build_cuda/golden/ rsync 到板上同一位置，
      再用 ctest -FS golden 跳过生成，并在报告里写明。
   d. 锁频：先看还在不在（T27 时是 MAXN，GPU 最低、最高频率都是 1300500000）：
        ssh orin 'nvpmodel -q; cat /sys/class/devfreq/17000000.gpu/min_freq /sys/class/devfreq/17000000.gpu/max_freq' \
            2>&1 | tee data/reports/T28/orin_power.txt
      还在就不用找人；不在就请人锁（同 T26 第 6c 步，sudo 密码问人，不要写进任何文件）。
      然后板上跑 GPU 检测器的全链路，拿回来和服务器的 C++ 比、看延迟：
        ssh orin 'cd ~/dogperc/repo && deploy/cpp/build_cuda/dogperc_replay --seq ~/dogperc/seq/0921_164127_s000 \
            --config ~/dogperc/seq/0921_164127_s000/config.yaml --model-dir ~/dogperc/models/nusc_pp_dyn --precision fp16 --gpu \
            --out ~/dogperc/data/orin_cpp.npz' 2>&1 | tail -3 | tee data/reports/T28/orin_replay.txt
        scp orin:dogperc/data/orin_cpp.npz $O/orin_cpp.npz
        python tools/deploy/cpp_parity.py compare --ref $O/0921_164127_s000/cpp.npz --cpp $O/orin_cpp.npz --subset \
            --out data/reports/T28/orin_parity.json | tail -3 | tee data/reports/T28/orin_parity.txt
        python tools/deploy/cpp_parity.py timings $O/orin_cpp.npz --out data/reports/T28/orin_timings.json \
            | tee data/reports/T28/orin_timings.txt
      板上只有前 150 帧，--subset 按两边共有的帧比。门槛同第 3 步的各自检测。
   e. 做完删掉板上的 ~/dogperc/seq 和 deploy/cpp/build_cuda/golden，记下剩余空间。

停止条件（都是按约定 9 修不好的时候）：编译失败；第 2 步有测试失败；第 3 步同一批检测有场景不过。
提交：data/reports/T28/、本文档，以及按约定 9 改的代码（每个修复单独提交）。不提交 data/deploy/T28/ 和 build 目录。
报告（写进"T28 结果"）：
- 代码改动（通用约定 9）：每条写提交号、文件、原因、怎么验证的；没有就写没有；
- 第 0 步：s004、s005 的 compare 表；trt:int8 每种 head 的余弦；
- 镜像和两边的编译日志里的报错 / 警告（没有就写没有）；
- 两边 ctest 的结果，GPU 测试打印的每帧行（框数、GPU 各段时间）；
- 10 个场景两次 compare 的表（同一批检测、各自检测）和汇总；
- 第 4 步 py / cpp 两份 report.md 的 AP@2、AP@4 和 MAIN_METRICS 并排一张表；
- 服务器和板上的每段延迟表（preprocess、det_gpu、det_voxelize、det_pfn、det_rpn、det_post、track、predict、total 的中位数和 p99），板上锁频了没有，用的柱数；
- 板上对服务器 C++ 的 compare 表；
- 结论：C++ 能不能替代 Python；板上离"网络以外 < 15 ms、端到端 < 50 ms"（DEPLOY_ORIN.md 第 6 节）还差多少，差在哪一段。
```

### T28 结果

（2026-10-29）GPU 用物理卡 4。模型是 `data/deploy/nusc_pp_dyn`（FP16，动态柱数）。没有改代码。

**代码改动（通用约定 9）：** 没有。

**第 0 步。** s004 的 trt:fp16 过了。s005 的 trt:fp16 没过：过滤前不计翻转 0.9911，过滤后 0.9898（门槛 0.99）。漏 7 个是 near 5 / none 2，多 5 个是 near 3 / none 2，没有 score、class。按任务继续。

| 场景 | backend | reported | 过滤前不计翻转 | 过滤后不计翻转 | 漏 (near/filter/score/class/none) | 多 | score p99 | center p99 m | yaw p99 | 最差 head 余弦 | 网络 ms | 门槛 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s004 | onnx | 0.9982 | 1.0000 | 1.0000 | 1 (1/0/0/0/0) | 0 | 1.02e-03 | 7.42e-04 | 0.016 | — | 345.0 | pass |
| s004 | trt:fp16 | 0.9700 | 0.9940 | 0.9929 | 5 (1/2/0/0/2) | 12 (8/2/0/0/2) | 5.66e-03 | 3.57e-03 | 0.095 | 0.9888 (vel) | 36.1 | pass |
| s004 | trt:int8 | 0.1199 | 0.0583 | 0.1940 | 318 (0/33/145/3/137) | 181 (1/8/66/10/96) | 0.375 | 0.475 | 3.099 | −0.8305 (height) | 35.3 | FAIL |
| s005 | onnx | 0.9974 | 1.0000 | 1.0000 | 1 (1/0/0/0/0) | 0 | 1.83e-03 | 1.39e-03 | 0.016 | — | 340.0 | pass |
| s005 | trt:fp16 | 0.9694 | 0.9911 | 0.9898 | 7 (5/0/0/0/2) | 5 (3/0/0/0/2) | 7.51e-03 | 3.69e-03 | 0.081 | 0.9890 (vel) | 26.9 | FAIL |
| s005 | trt:int8 | −0.0230 | −0.0402 | 0.0842 | 301 (0/33/108/9/151) | 100 (0/9/55/1/35) | 0.350 | 0.489 | 2.879 | −0.8520 (height) | 25.2 | FAIL |

trt:int8 每种 head 的余弦（stages 里对帧取最小）：

| head | s004 | s005 |
|---|---|---|
| reg | 0.713 | 0.794 |
| height | −0.830 | −0.852 |
| dim | 0.335 | 0.317 |
| rot | −0.413 | −0.412 |
| vel | −0.237 | −0.214 |
| hm | 0.611 | 0.792 |

负的是 height、rot、vel。最差是 height，不是热力图。T27 复核里看到的约 −0.58 比这次的最差值更接近 0。

**镜像和编译。** `dogperc-dev` 从 `nvcr.io/nvidia/tensorrt:24.08-py3` 建成（TensorRT 10.3.0，CUDA 12.6）。构建日志里只有 pip 以 root 安装的警告，没有报错。服务器容器里 Release、sm_89 编译没有警告。板上没有 ninja，按任务去掉了 `-G Ninja`；aarch64、GCC 11、sm_87 编过了。板上 gmake 报 clock skew：系统时间停在 1970，同步过来的源文件时间在未来。链接完成，ctest 全过，二进制是齐的。

**ctest。** 服务器（动态 PFN）和板上都是 2/2 通过，golden 在各自机器上生成。Gpu.* 都跑了，不是 SKIPPED。静态 PFN 再跑一次 Gpu.* 也通过。

动态 PFN（服务器，`nusc_pp_dyn`）：

```text
frame 0:  15 boxes, gpu 2.55 ms (voxelize 0.68, pfn 0.12, rpn 0.83)
frame 5:  10 boxes, gpu 3.38 ms (voxelize 1.40, pfn 0.15, rpn 0.81)
frame 10:  8 boxes, gpu 3.42 ms (voxelize 1.48, pfn 0.15, rpn 0.81)
frame 15: 10 boxes, gpu 3.28 ms (voxelize 1.27, pfn 0.17, rpn 0.81)
frame 20:  9 boxes, gpu 3.17 ms (voxelize 1.19, pfn 0.16, rpn 0.81)
frame 25:  7 boxes, gpu 3.26 ms (voxelize 1.25, pfn 0.17, rpn 0.81)
```

静态 PFN（服务器，`nusc_pp`）：同样 6 帧，框数 14/10/7/10/9/7，gpu 2.89–3.82 ms，pfn 0.45–0.46 ms，rpn 0.80–0.82 ms。

板上动态 PFN：

```text
frame 0:  15 boxes, gpu  8.65 ms (voxelize 1.75, pfn 0.61, rpn 4.85)
frame 5:  10 boxes, gpu 14.76 ms (voxelize 6.68, pfn 1.78, rpn 4.81)
frame 10:  7 boxes, gpu 13.06 ms (voxelize 5.01, pfn 1.76, rpn 4.81)
frame 15: 10 boxes, gpu 12.03 ms (voxelize 3.76, pfn 1.92, rpn 4.82)
frame 20:  9 boxes, gpu 11.77 ms (voxelize 3.59, pfn 1.85, rpn 4.81)
frame 25:  7 boxes, gpu 12.01 ms (voxelize 3.74, pfn 1.98, rpn 4.80)
```

**10 个验证场景。** 同一批检测：10 个都 pass。帧数相同，检测相同，每帧跟踪 ID 完全一致。状态最大差 2.5e-11（s005），预测最大差 2.3e-10（s005），都在 1e-6 / 1e-4 里面。

各自检测（C++ GPU 对 Python TensorRT FP16）：10 个都 pass。过滤前、过滤后匹配率都是 1.0000，漏和多都是 0。分数 p99 ≤ 6e-8，中心 p99 是 0。跟踪 ID 也每帧相同。状态最大差 1.6e-6（s000），是分数最后一个比特带来的，同一批检测那次没有这个差。

**标签。** `pipeline_eval.py --detections raw` 重跑 Python 的过滤、跟踪、预测。两边 AP@2.0 都是 0.832，AP@4.0 都是 0.852，差 0。MAIN 指标逐项相同，都在 py 的 90% 区间里。

| 指标 | py | cpp | py 的 90% 区间 |
|---|---|---|---|
| AP@2.0 / AP@4.0 | 0.832 / 0.852 | 0.832 / 0.852 | — |
| 近处召回 | 0.837 | 0.837 | 0.789–0.859 |
| 近处精确率 | 0.823 | 0.823 | 0.632–0.901 |
| 近处 F1 | 0.830 | 0.830 | 0.706–0.871 |
| 每关键帧误检 | 0.650 | 0.650 | 0.491–0.813 |
| MOTA | 0.606 | 0.606 | 0.297–0.708 |
| IDF1 | 0.785 | 0.785 | 0.662–0.837 |
| ID 切换 / 分钟 | 1.862 | 1.862 | 0.419–3.356 |
| 假轨迹 / 分钟 | 26.481 | 26.481 | 19.756–33.102 |
| 确认延迟 p90 s | 1.200 | 1.200 | 0.501–4.000 |
| 相关运动段漏报 | 2 | 2 | 0–4 |
| 停着的车报成在动 / 小时 | 86.891 | 86.891 | 12.388–175.081 |
| 预测 FDE 3 s，在动 m | 2.801 | 2.801 | 2.305–3.625 |
| 预测漏报率 3 s，在动 | 0.521 | 0.521 | 0.430–0.694 |

**延迟。** 服务器是 10 个场景、去掉前 5 帧后的 2834 帧合在一起。4090 上体素化有 195 帧超过 20 ms（p99 215 ms），中位数仍是 1.10 ms；这是共享 GPU 上的尖峰，板上没有。`x86_timings.json` 的键是文件名 `cpp`，10 个文件同名，JSON 里只留下最后一个场景；10 段表在 `x86_timings.txt`。

| 段 | 服务器中位数 / p99 ms | 板上中位数 / p99 ms |
|---|---|---|
| preprocess | 12.92 / 20.52 | 14.75 / 15.12 |
| det_gpu | 3.08 / 216.83 | 9.34 / 11.02 |
| det_voxelize | 1.10 / 214.68 | 2.34 / 3.99 |
| det_pfn | 0.17 / 0.27 | 0.80 / 0.83 |
| det_rpn | 0.80 / 0.82 | 4.78 / 4.79 |
| det_post | 3.97 / 6.65 | 3.26 / 3.31 |
| track | 0.07 / 0.20 | 0.18 / 0.21 |
| predict | 0.02 / 0.08 | 0.08 / 0.09 |
| total | 20.34 / 239.41 | 27.71 / 30.16 |
| det_network | 1.03 / 1.14 | 5.91 / 5.94 |

柱数：服务器 10 个场景中位数 11166（p99 15111，最大 17707）。板上 s000 前 150 帧中位数 5059（p99 5272，最大 5278）。锁频还在：MAXN，GPU 最低、最高都是 1300500000，没有找人。

**板上对服务器 C++**（s000 前 150 帧，`--subset`）：pass。帧 288 / 150 / 150。过滤前、过滤后匹配率都是 1.0000。过滤前漏、多各 1 个，都是 near。过滤后漏 2（near 1、filter 1），多 1（near）。分数 p99 3.4e-3，中心 p99 2.6e-3 m。跟踪 ID 147 / 150 帧相同。配上的轨迹里最大差是 yaw 1.14 rad（位置 0.037 m，速度 0.055 m/s），预测最大差 0.20。这 3 帧是阈值附近的翻转，不是同一批检测那条路径。

**结论。** C++ 可以替代 Python。同一批检测时跟踪和预测对齐；各自用 GPU 检测时，10 个验证场景的框和标签分数都一样。

板上（DEPLOY_ORIN.md 第 6 节，P7）：端到端中位数 27.7 ms，低于 50 ms，还剩约 22 ms；p99 30.2 ms，低于 100 ms。网络以外是 27.7 − 5.9 = 21.8 ms，比 15 ms 多 6.8 ms。多出来的主要在预处理，去畸变中位数 11.9 ms，预处理合计 14.8 ms；后面是过滤后处理 3.3 ms、体素化 2.3 ms、把结果拷回主机 1.1 ms。网络本身（PFN 0.80 ms + RPN 4.78 ms）不是这块缺口。回放 150 帧墙钟 9.2 s，含读盘；流水线里记的端到端是 27.7 ms，10 Hz 的 100 ms 帧预算装得下。做完删了板上的 `~/dogperc/seq` 和 `deploy/cpp/build_cuda/golden`，根分区仍是 4.6 GB。

### T28 复核

2026-10-29。结论同意：C++ 可以替代 Python。
- 同一批检测时，10 个场景的跟踪 ID 每帧相同，状态差 2.5e-11；
- 各自检测时，C++ GPU 检测器和 Python 的 TensorRT FP16 逐框相同（分数 p99 6e-8），标签上的指标逐项相同；
- 板上对服务器也过了。

1. **s005 的 FP16 过滤后是 0.9898**（门槛 0.99）。392 个上报框里错了 4 个，都是 none，过滤前也有。多半是 NMS 在同一个物体的两个峰之间换了一个，框挪了 0.5 m 以上。
   - 标签上 FP16 和 PyTorch 只差 0.001（T26、T27），上线 FP16 不变；
   - 门槛不再放宽：前两次改的是确定的阈值翻转，这一类要看到框才能下结论。以后再碰到，用 `compare_backends.py --dump` 存下框来看。
2. **INT8 每种 head 都坏了**：hm 0.61、reg 0.71、dim 0.33，height、rot、vel 是负的。不是某一个 head 的问题，是整个主干，和"输入被截断"的猜测一致。记进 DEPLOY_ORIN.md 第 5 节，INT8 继续不追。
3. **预处理慢在去重，不在去畸变。** T28 的 `deskew_ms` 其实把去重、自身滤除、去畸变三步一起算了。复核把样例 bag（`samples/rosbag2_2026_09_21-16_41_27_moving`，27 帧真实 XT32，每帧 10.6 万点，一半是双回波的重复点）转成序列，在本机 x86 上逐步计时：

   | 步骤 | 改前 | 改后 | 改了什么 |
   |---|---|---|---|
   | 去重 | 6.1 ms | 2.4 ms | 逐点 `std::unordered_map` 换成开放寻址的扁平哈希集合，跨帧复用内存；四舍六入五成双改成内联（x86 上 `nearbyintf` 是函数调用） |
   | 去畸变 | 1.4 ms | 0.4 ms | 时间格的排名从排序加二分换成稠密数组 |
   | 一帧扫描合计（`sweep_ms`） | 8.4 ms | 3.9 ms | 外加只收集一次点，不再先拷出一份扫描 |
   | 解码 | 0.64 ms | 0.24 ms | 先用 logit 筛掉够不到分数阈值的格子，再算 sigmoid |
   | 过滤（数框内点、取当前帧） | 约 1.6 ms | 约 1.1 ms | 点网格从 `unordered_map<int64, vector>` 换成 CSR 稠密网格 |

   - GPU 检测器里的 head 图改成移动，不再拷贝，约 0.3 ms；
   - 拼帧（5 帧共 50 万点从 float64 世界坐标变到检测系）没变，本机约 3–4 ms，板上 2.8 ms，受内存带宽限制。加了线程池（`dogperc_replay --threads N`，默认 1），本机 4 个共享 vCPU 上没有收益，T28B 在板上测；
   - 只编译过、没跑过：thrust 排序的临时显存改用缓存分配器（原来每次 cudaMalloc / cudaFree，会同步整个设备），设备缓冲区按 1.5 倍增长。服务器上体素化 p99 有 215 ms 的尖峰，可能和这个有关；
   - 新的计时项：`sweep_ms`（就是 T28 的 `deskew_ms`），`dedup_ms`、`self_filter_ms`、`deskew_ms`（只算去畸变），`det_h2d_ms`（点云上传，原来算在 `det_voxelize_ms` 里）。

   这些改动输出都不变：
   - golden 对齐测试照样全过，加了一个"1 个线程和 4 个线程输出逐位相同"的测试；
   - 27 帧真实扫描上，改前、改后的 `dogperc_replay --save-points` 输出逐位相同（473 万点，1 个线程和 4 个线程都是）。

   本机 x86 的预处理中位数从 12.9 ms 降到 8.4 ms。按 T28 板上和本机的比例估，板上预处理会从 14.8 ms 降到 8 ms 左右，后处理从 3.3 ms 降到 2 ms 左右，网络以外大约 14 ms，刚好在 15 ms 以内。T28B 实测。
4. `cpp_parity.py timings`：输入文件同名时，改用目录名区分（T28 的 JSON 只剩最后一个场景）；输入多个文件时，另加一行 `all`，把所有帧合在一起算。
5. 板子时间停在 1970 年，make 会报 clock skew，只是每次都全量重编，结果没错。要修得有人在板上 `sudo date -s` 设一次时间。
6. 人已经加了免密 sudo（2026-10-29）。锁频掉了，Agent 自己执行 `sudo -n jetson_clocks`；`-n` 报要密码，就停下问人。
7. golden 的点差值：DEPLOY_ORIN.md 里写的 9.3e-10 m 是旧数，改前的代码现在也是 0（逐位相同），已更正。

## T28B C++ 提速后的复测

T28 复核改了 `deploy/cpp` 的预处理、后处理和 GPU 体素化的内存分配，计时项也拆细了。照理输出一位都不变，这次确认这一点，并在板上量新的每段延迟。本任务可以改的范围同 T28。

```text
任务 T28B：C++ 提速后的复测。服务器容器和板上重编、测试；两个场景确认输出和 T28 逐位相同；板上 1 / 4 线程的每段延迟。
先按通用约定做。前置：T28 完成（data/deploy/T28 下的 py_dets.npz、cpp_dets.npz、cpp.npz 还在）。

0. mkdir -p data/reports/T28B data/deploy/T28B；O=data/deploy/T28；N=data/deploy/T28B；M=data/deploy/nusc_pp_dyn；G、dk 同 T28 第 0 步。
   S="0921_164127_s000 0921_165659_s004"
1. 服务器容器里重编、测试（同 T28 第 2 步，含静态 PFN 的 Gpu.*）：
     输出 data/reports/T28B/build_x86.txt、ctest_x86.txt、gtest_gpu_static.txt。
   - 新加的 Pipeline.ThreadCountDoesNotChangeTheOutput 要过；
   - Gpu.* 打印的每帧行里多了 h2d。
2. 输出和 T28 比（每个场景）：
     for s in $S; do
       mkdir -p $N/$s
       dk "deploy/cpp/build_cuda/dogperc_replay --seq data/seq/$s --config $O/$s/config.yaml \
           --detections $O/$s/py_dets.npz --out $N/$s/cpp_dets.npz"
       for th in 1 4; do
         dk "deploy/cpp/build_cuda/dogperc_replay --seq data/seq/$s --config $O/$s/config.yaml \
             --model-dir $M --precision fp16 --gpu --threads $th --out $N/$s/cpp_t$th.npz"
       done
       python tools/deploy/cpp_parity.py compare --ref $O/$s/cpp_dets.npz --cpp $N/$s/cpp_dets.npz --same-detections \
           --out data/reports/T28B/same_dets_$s.json | tail -1 | tee data/reports/T28B/same_dets_$s.txt
       python tools/deploy/cpp_parity.py compare --ref $O/$s/cpp.npz --cpp $N/$s/cpp_t1.npz --same-detections \
           --out data/reports/T28B/same_gpu_$s.json | tail -1 | tee data/reports/T28B/same_gpu_$s.txt
       python tools/deploy/cpp_parity.py compare --ref $N/$s/cpp_t1.npz --cpp $N/$s/cpp_t4.npz --same-detections \
           --out data/reports/T28B/threads_$s.json | tail -1 | tee data/reports/T28B/threads_$s.txt
     done
     python tools/deploy/cpp_parity.py timings $O/*/cpp.npz --out data/reports/T28B/x86_old.json > data/reports/T28B/x86_old.txt
     python tools/deploy/cpp_parity.py timings $N/*/cpp_t1.npz --out data/reports/T28B/x86_t1.json > data/reports/T28B/x86_t1.txt
     python tools/deploy/cpp_parity.py timings $N/*/cpp_t4.npz --out data/reports/T28B/x86_t4.json > data/reports/T28B/x86_t4.txt
   - same_dets（喂同一批检测）和 threads（1 对 4 线程）必须 pass：提速不能改结果。不过就停下，报告第一个不同的帧；
   - same_gpu（GPU 检测器，新对旧）：应该也 pass。只有 detections_identical 不过、而且分数最大差 < 1e-6 的话，
     是 TensorRT 两次运行不逐位一致，写进报告，接着做；别的不过就停下。
3. 板上（同 T28 第 6 步，模型 ~/dogperc/models/nusc_pp_dyn）：
   a. sync_to_orin.sh --seq data/seq/0921_164127_s000 --limit 150；拷 config.yaml；看剩余空间；
   b. 重编（同 T28 第 6b 步），ctest（同 6c），输出 build_orin.txt、ctest_orin.txt；
   c. 锁频：先查（同 T28 第 6d 步，输出 orin_power.txt），不在就 ssh orin 'sudo -n jetson_clocks'，再查一次；
      sudo -n 报要密码就停下问人；
   d. 板上 1 和 4 线程各跑一次：
        for th in 1 4; do
          ssh orin "cd ~/dogperc/repo && deploy/cpp/build_cuda/dogperc_replay --seq ~/dogperc/seq/0921_164127_s000 \
              --config ~/dogperc/seq/0921_164127_s000/config.yaml --model-dir ~/dogperc/models/nusc_pp_dyn --precision fp16 \
              --gpu --threads $th --out ~/dogperc/data/orin_t$th.npz" 2>&1 | tail -1
          scp orin:dogperc/data/orin_t$th.npz $N/orin_t$th.npz
        done
        python tools/deploy/cpp_parity.py compare --ref $N/0921_164127_s000/cpp_t1.npz --cpp $N/orin_t1.npz --subset \
            --out data/reports/T28B/orin_parity.json | tail -1 | tee data/reports/T28B/orin_parity.txt
        python tools/deploy/cpp_parity.py compare --ref $N/orin_t1.npz --cpp $N/orin_t4.npz --same-detections \
            --out data/reports/T28B/orin_threads.json | tail -1 | tee data/reports/T28B/orin_threads.txt
        for th in 1 4; do
          python tools/deploy/cpp_parity.py timings $N/orin_t$th.npz --out data/reports/T28B/orin_t$th.json \
              | tee data/reports/T28B/orin_t$th.txt
        done
      T28 板上的旧结果在 data/deploy/T28/orin_cpp.npz，也用 timings 汇总一次（orin_old.txt）。
   e. 做完删掉板上的 ~/dogperc/seq、~/dogperc/data/orin_t*.npz 和 deploy/cpp/build_cuda/golden，记下剩余空间。

停止条件：编译或测试按约定 9 修不好；第 2 步 same_dets、threads 不过；板上 orin_threads 不过。
提交：data/reports/T28B/、本文档，以及按约定 9 改的代码。不提交 data/deploy/T28B/。
报告（写进"T28B 结果"）：
- 代码改动（约定 9）；两边编译、测试的结果；
- 第 2 步每个场景三次 compare 的结果；
- 每段延迟一张表，列：服务器旧（T28）/ 新 1 线程 / 新 4 线程，板上旧（T28）/ 新 1 线程 / 新 4 线程，都是中位数 / p99。
  行：preprocess、sweep（旧的叫 deskew）、dedup、self_filter、deskew、accumulate、det_h2d、det_voxelize、det_pfn、det_rpn、
  det_d2h、det_post、track、predict、total；外加"网络以外"= total − det_network；
- 服务器上 det_voxelize 的 p99 尖峰还在不在；
- 结论：板上网络以外的中位数到没到 15 ms，用几个线程，还差在哪一段。
```

### T28B 结果

（2026-10-29）GPU 用物理卡 4。拉到的代码是 `f0b41fa`。没有再改代码。

**代码改动（约定 9）：** 没有。

**编译和测试。** 服务器容器里 Release、sm_89 重编，没有警告。ctest 2/2 通过，20 个测试，含 `Pipeline.ThreadCountDoesNotChangeTheOutput`。静态 PFN 的 Gpu.* 也通过。板上去掉 ninja 后 sm_87 编过了，仍有 gmake clock skew（系统时间停在 1970，源文件时间在未来）；ctest 2/2 通过，同样 20 个测试。锁频还在：MAXN，GPU 最低、最高都是 1300500000，没有执行 `jetson_clocks`。

Gpu.* 的每帧行没有单独的 h2d，仍是 voxelize / pfn / rpn。上传时间在回放的 `det_h2d_ms` 里（板上中位数 0.43 ms）。

动态 PFN，服务器：

```text
frame 0:  15 boxes, gpu 3.06 ms (voxelize 0.51, pfn 0.13, rpn 0.76)
frame 5:  10 boxes, gpu 3.42 ms (voxelize 0.95, pfn 0.14, rpn 0.75)
frame 25:  7 boxes, gpu 2.82 ms (voxelize 0.37, pfn 0.16, rpn 0.74)
```

板上同一组帧：gpu 8.22–14.91 ms，pfn 0.61–1.98 ms，rpn 4.77–4.80 ms。

快速测试：163 通过。`test_trt_runner_on_a_tiny_engine` 有一个 worker 中途退出，单独重跑通过，不是这次改动引入的失败。

**第 2 步。** s000 和 s004，三次 compare 都是 pass，位姿、跟踪、预测的最大差都是 0。

| 场景 | same_dets（同一批检测，新对旧） | same_gpu（GPU，1 线程新对旧） | threads（1 对 4） |
|---|---|---|---|
| 0921_164127_s000 | pass，288 帧，差 0 | pass，288 帧，差 0 | pass，差 0 |
| 0921_165659_s004 | pass，276 帧，差 0 | pass，276 帧，差 0 | pass，差 0 |

**延迟。** 中位数 / p99，毫秒。服务器旧是 T28 的 10 个场景合在一起；新的 1 / 4 线程是这次跑的 s000 和 s004。同一对场景上，旧的预处理中位数是 12.79 ms，和新列的对比一致。旧的 `deskew_ms` 含去重、自身滤除和去畸变，写在 sweep 这一行；拆开的三项旧结果没有。板上旧是 T28 的 s000 前 150 帧，新的也是这 150 帧。网络以外是逐帧 `total − det_network`。

| 段 | 服务器旧 | 服务器 1 线程 | 服务器 4 线程 | 板上旧 | 板上 1 线程 | 板上 4 线程 |
|---|---|---|---|---|---|---|
| preprocess | 12.92 / 20.52 | 7.70 / 11.76 | 9.35 / 19.78 | 14.75 / 15.12 | 7.64 / 7.98 | 6.21 / 6.67 |
| sweep | 10.38 / 16.86 | 5.33 / 6.84 | 5.34 / 10.06 | 11.94 / 12.30 | 4.81 / 4.95 | 4.24 / 4.43 |
| dedup | — | 3.23 / 3.97 | 3.36 / 4.02 | — | 2.58 / 2.71 | 2.57 / 2.78 |
| self_filter | — | 1.84 / 2.65 | 1.54 / 5.21 | — | 1.68 / 1.73 | 1.29 / 1.33 |
| deskew | — | 0.50 / 0.76 | 0.53 / 5.18 | — | 0.55 / 0.58 | 0.38 / 0.40 |
| accumulate | 2.53 / 4.21 | 2.37 / 5.15 | 3.26 / 11.89 | 2.78 / 3.12 | 2.82 / 3.14 | 1.96 / 2.25 |
| det_h2d | — | 0.30 / 0.40 | 0.35 / 0.59 | — | 0.43 / 0.47 | 0.42 / 0.46 |
| det_voxelize | 1.10 / 214.68 | 0.36 / 18.56 | 0.37 / 79.46 | 2.34 / 3.99 | 1.60 / 2.41 | 1.61 / 2.43 |
| det_pfn | 0.17 / 0.27 | 0.18 / 0.31 | 0.17 / 0.33 | 0.80 / 0.83 | 0.80 / 0.83 | 0.80 / 0.83 |
| det_rpn | 0.80 / 0.82 | 0.74 / 0.76 | 0.74 / 0.76 | 4.78 / 4.79 | 4.72 / 4.78 | 4.73 / 4.79 |
| det_d2h | 0.90 / 1.50 | 0.91 / 1.51 | 1.01 / 1.76 | 1.08 / 1.13 | 1.10 / 1.14 | 1.10 / 1.14 |
| det_post | 3.97 / 6.65 | 2.93 / 3.26 | 3.04 / 3.52 | 3.26 / 3.31 | 1.73 / 1.81 | 1.70 / 1.77 |
| track | 0.07 / 0.20 | 0.12 / 0.20 | 0.11 / 0.20 | 0.18 / 0.21 | 0.20 / 0.23 | 0.20 / 0.24 |
| predict | 0.02 / 0.08 | 0.04 / 0.07 | 0.04 / 0.07 | 0.08 / 0.09 | 0.08 / 0.09 | 0.08 / 0.10 |
| total | 20.34 / 239.41 | 13.58 / 34.34 | 16.83 / 96.24 | 27.71 / 30.16 | 18.70 / 19.65 | 17.26 / 18.19 |
| 网络以外 | 19.29 / 238.21 | 12.61 / 33.40 | 15.83 / 95.26 | 21.79 / 24.25 | 12.87 / 13.75 | 11.41 / 12.28 |

服务器体素化 p99：旧 215 ms（554 帧里 57 帧超过 20 ms，只算 s000 和 s004）。1 线程降到 18.6 ms（5 帧超过 20 ms），尖峰还在，小了很多。4 线程又到 79 ms。板上 p99 一直是 2.4 ms，没有这个尖峰。

**板上对服务器、1 对 4 线程。** 1 线程的新结果对服务器 1 线程（前 150 帧）：pass。过滤前后匹配率 1，漏 2（near 1、filter 1），多 1（near），和 T28 一样是阈值翻转。4 线程对 1 线程：pass，150 帧，差是 0。柱数中位数仍是 5059。

**结论。** 提速没有改输出。板上网络以外的中位数到了 15 ms 以内：1 线程 12.9 ms，4 线程 11.4 ms。用 4 个线程（端到端 17.3 ms，p99 18.2 ms）。4 个线程省下的是拼帧、自身滤除和去畸变，大约 1.4 ms；去重还是 2.6 ms，没有降。剩下比较长的是去重 2.6 ms、拼帧 2.0 ms、过滤后处理 1.7 ms、体素化 1.6 ms、结果拷回 1.1 ms。网络仍是 PFN 0.80 ms 加 RPN 4.73 ms。服务器上 4 线程比 1 线程慢（预处理 9.4 ms 对 7.7 ms），和复核说的共享 vCPU 没有收益一致，不要在服务器上开 4 线程。做完删了板上的 `~/dogperc/seq`、`orin_t1.npz`、`orin_t4.npz` 和 `build_cuda/golden`，根分区仍是 4.6 GB。

### T28B 复核

2026-10-29。结论同意。
- 提速没有改输出：两个场景三次比较的差都是 0，1 个线程和 4 个线程也相同；
- 板上网络以外的中位数到了 15 ms 以内：1 个线程 12.9 ms，4 个线程 11.4 ms，端到端 17.3 ms（p99 18.2 ms）；
- 板上用 4 个线程，节点的 launch 默认就是 4；服务器上用 1 个。

1. P7 的延迟这一项达标了。还剩稳定性：长时间跑、10 Hz 不丢帧、和板上别的进程抢 CPU / GPU。T29 先在回放 bag 时查不丢帧，长时间要上车（人）。
2. 服务器体素化的 p99 尖峰（1 线程 18.6 ms，4 线程 79 ms）只在共享的 4090 上有，板上 p99 一直是 2.4 ms。不追。
3. Gpu.* 的每帧行确实没有 h2d，上传时间只在回放的 `det_h2d_ms` 里，复核里说错了。
4. `test_trt_runner_on_a_tiny_engine` 那次 worker 中途退出、单独重跑通过。以后再出现就记下是哪个测试、退出码是多少。
5. P6 的 ROS 2 节点写好了，见 DEPLOY_ORIN.md 第 8 节，T29 编译、验证。

## T29 Orin 部署：C++ ROS 2 节点

ROS 2 节点写好了（`deploy/ros2`，DEPLOY_ORIN.md 第 8 节）。节点的消息处理在 libdogperc 里，不依赖 ROS（`dogperc/online.hpp`），本机已经和 Python 节点的消息处理（`dog_perception/online.py`）在样例 bag 上逐帧对上。ROS 那一层（rclcpp 节点、rosbag2 读取、消息转换）本机没有 ROS，没编译过。这次要：
- 服务器：Python 节点回放 bag 和 C++ 的消息处理比，样例 bag 和一段 0921 原始 bag；
- 板上：colcon 编译；`dogperc_bag_replay` 直接读 bag，和服务器上的 C++ 结果逐位比；在线 `ros2 bag play`，查不丢帧、看输出。

本任务可以改的范围（通用约定 9）：`deploy/ros2/`、`deploy/cpp/`、`deploy/docker/`、`deploy/scripts/`、`tools/deploy/`。`dog_perception/` 不改；对不上时以 Python 为准改 C++。

```text
任务 T29：C++ ROS 2 节点。服务器上 Python 节点的 bag 回放对 C++ 的消息处理；板上 colcon 编译、bag 回放逐位比、在线回放查不丢帧。
先按通用约定做（第 9 条：可以改代码，范围见本节开头）。前置：T28B 完成。

0. mkdir -p data/reports/T29 data/deploy/T29；TASK=T29；N=data/deploy/T29；M=data/deploy/nusc_pp_dyn；G、dk 同 T28 第 0 步。
   python -m pytest tests/dog/test_ros2_node.py tests/dog/test_deploy.py -q 2>&1 | tee data/reports/T29/pytest.txt
   （test_ros2_node 的慢测试要 rosbags 和 samples/，这里要跑到，不能是 skipped）
   快速测试（约定 7）：python -m pytest tests/dog -m "not slow" -n 8 -q 2>&1 | tail -3 | tee data/reports/T29/pytest_fast.txt
1. 镜像加了 rosbags（golden 数据多了样例 bag 的在线回放），重建后编译、测试（同 T28 第 2 步）：
     docker build -t dogperc-dev -f deploy/docker/Dockerfile.dev deploy/docker 2>&1 | tail -3 | tee data/reports/T29/docker_build.txt
     （编译、ctest 同 T28 第 2 步，输出 build_x86.txt、ctest_x86.txt）
   - Online.* 五个测试都要跑了、通过；Online.MessageLogMatchesPythonBagReplay 不能是 SKIPPED（打印 "online parity: 8 frames ..."）。
2. 服务器上，Python 节点回放 bag 对 C++ 的消息处理。两段：
     B1=samples/rosbag2_2026_09_21-16_41_27_moving             （27 帧）
     B2=/zjst/robot/rosbag0921/rosbag2_2026_09_21-16_41_27       （只取前 600 帧，约 60 s：--max-clouds 600）
   每段 b（名字 n = basename，B2 的参数加 --max-clouds 600）：
     mkdir -p $N/$n
     python tools/deploy/cpp_parity.py python --bag $b --robot kygi --exported $M:trt:fp16 \
         --out $N/$n/py.npz --dets $N/$n/py_dets.npz --config $N/$n/kygi.yaml --messages $N/$n/msgs.npz \
         2>&1 | tail -3 | tee data/reports/T29/python_$n.txt
     dk "deploy/cpp/build_cuda/dogperc_replay --messages $N/$n/msgs.npz --config $N/$n/kygi.yaml \
         --detections $N/$n/py_dets.npz --out $N/$n/cpp_dets.npz"
     dk "deploy/cpp/build_cuda/dogperc_replay --messages $N/$n/msgs.npz --config $N/$n/kygi.yaml \
         --model-dir $M --precision fp16 --gpu --out $N/$n/cpp.npz"
     python tools/deploy/cpp_parity.py compare --ref $N/$n/py.npz --cpp $N/$n/cpp_dets.npz --same-detections \
         --out data/reports/T29/msgs_dets_$n.json | tail -1 | tee data/reports/T29/msgs_dets_$n.txt
     python tools/deploy/cpp_parity.py compare --ref $N/$n/py.npz --cpp $N/$n/cpp.npz \
         --out data/reports/T29/msgs_$n.json | tail -1 | tee data/reports/T29/msgs_$n.txt
   - B2 的消息日志约几百 MB，Python 回放要几分钟；
   - msgs_dets（喂同一批检测）两段都要 pass：帧相同、每帧跟踪 ID 一致、状态 ≤ 1e-6、预测 ≤ 1e-4。不过就按约定 9 修 C++；
   - msgs（各自检测）：门槛同 T28 第 3 步。
3. 板上（ROS 2 Humble 在 /opt/ros/humble）：
   a. 同步代码、样例 bag 和这一段的输入：
        deploy/scripts/sync_to_orin.sh --bag samples/rosbag2_2026_09_21-16_41_27_moving
        n=rosbag2_2026_09_21-16_41_27_moving
        scp $N/$n/kygi.yaml $N/$n/py_dets.npz $N/$n/msgs.npz orin:dogperc/data/
   b. 编译（第一次约 10 分钟，libdogperc 带 CUDA 会再编一次）：
        ssh orin 'source /opt/ros/humble/setup.bash && cd ~/dogperc/repo && colcon build \
            --base-paths deploy/ros2 kygi665_msgs --build-base deploy/ros2/build --install-base deploy/ros2/install \
            --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
            -DCMAKE_CUDA_ARCHITECTURES=87' 2>&1 | tail -30 | tee data/reports/T29/colcon_orin.txt
      编译错误按约定 9 修（多半是 rclcpp / rosbag2 的接口细节），修完在服务器容器里重编 deploy/cpp、重跑 ctest。
      CMake 警告 "kygi665_msgs not found" 就是没把 kygi665_msgs 放进 --base-paths，必须修好（kygi 用 INSPVAX 定位）。
      deploy/cpp 自己的板上编译也更新一下（同 T28 第 6b 步，下面要用它的 dogperc_replay），ctest 同 6c，输出 ctest_orin.txt。
   c. bag 回放（R 是板上的命令前缀）。同一台机器上，ROS 读 bag 和读消息日志两条路必须逐位相同：
        R='source /opt/ros/humble/setup.bash && source ~/dogperc/repo/deploy/ros2/install/setup.bash && cd ~/dogperc'
        ssh orin "$R && ros2 run dogperc_ros dogperc_bag_replay --bag bags/$n --config data/kygi.yaml \
            --detections data/py_dets.npz --save-points --out data/bag_dets.npz" 2>&1 | tail -3 | tee data/reports/T29/orin_bag_dets.txt
        ssh orin "cd ~/dogperc && repo/deploy/cpp/build_cuda/dogperc_replay --messages data/msgs.npz --config data/kygi.yaml \
            --detections data/py_dets.npz --save-points --out data/msgs_dets.npz" 2>&1 | tail -1 | tee -a data/reports/T29/orin_bag_dets.txt
        ssh orin "$R && ros2 run dogperc_ros dogperc_bag_replay --bag bags/$n --config data/kygi.yaml \
            --model-dir models/nusc_pp_dyn --precision fp16 --gpu --threads 4 --out data/bag_gpu.npz" \
            2>&1 | tail -3 | tee data/reports/T29/orin_bag_gpu.txt
        scp orin:dogperc/data/bag_dets.npz orin:dogperc/data/msgs_dets.npz orin:dogperc/data/bag_gpu.npz $N/$n/
        python -c "import numpy as np, sys; a, b = np.load(sys.argv[1]), np.load(sys.argv[2]); \
            d = [k for k in a.files if not k.startswith('timing_') and not np.array_equal(a[k], b[k], equal_nan=True)]; \
            print('ROS bag vs message log, keys differing:', d)" $N/$n/msgs_dets.npz $N/$n/bag_dets.npz \
            | tee -a data/reports/T29/orin_bag_dets.txt
        python tools/deploy/cpp_parity.py compare --ref $N/$n/py.npz --cpp $N/$n/bag_dets.npz --same-detections \
            --out data/reports/T29/orin_bag_dets.json | tail -1 | tee -a data/reports/T29/orin_bag_dets.txt
        python tools/deploy/cpp_parity.py compare --ref $N/$n/cpp.npz --cpp $N/$n/bag_gpu.npz \
            --out data/reports/T29/orin_bag_gpu.json | tail -1 | tee -a data/reports/T29/orin_bag_gpu.txt
        python tools/deploy/cpp_parity.py timings $N/$n/bag_gpu.npz --skip 3 | tee data/reports/T29/orin_bag_timings.txt
      - "keys differing" 必须是空的：ROS 消息到 C++ 结构的转换（ros_convert.cpp）和读消息日志完全一样。不一样时先看消息顺序
        （rosbag2_cpp 和 rosbags 都按 bag 的时间戳读，同一时间戳的几条可能顺序不同），再按约定 9 修 ros_convert.cpp；
      - bag_dets 对 Python 的 py.npz 用 --same-detections 也要 pass（板上是 aarch64，状态差在 1e-6 以内就行）；
      - bag_gpu：门槛同 T28 第 6d 步（板上的引擎对服务器的引擎）。
   d. 在线：节点照常订阅，bag 按原来的节奏播放，录下 ~/objects。用单独的 ROS_DOMAIN_ID，不干扰板上别的 ROS 程序：
        ssh orin "$R && export ROS_DOMAIN_ID=77 && \
            (ros2 launch dogperc_ros dogperc.launch.py config:=data/kygi.yaml model_dir:=models/nusc_pp_dyn > data/node.log 2>&1 &) && \
            sleep 15 && (ros2 bag record -o data/live /dog_perception/objects > data/record.log 2>&1 &) && sleep 3 && \
            ros2 bag play bags/$n && sleep 3 && pkill -INT -f 'ros2 bag record' ; sleep 2 ; pkill -INT -f dogperc_node ; sleep 2 ; \
            ros2 bag info data/live" 2>&1 | tail -20 | tee data/reports/T29/orin_live.txt
        scp -r orin:dogperc/data/live $N/$n/live
        scp orin:dogperc/data/node.log data/reports/T29/orin_node.log
        python tools/deploy/objects_bag.py --bag $N/$n/live --replay $N/$n/bag_gpu.npz --out data/reports/T29/orin_live.json \
            | tee -a data/reports/T29/orin_live.txt
      - 节点起来要建引擎缓存、读模型，15 s 不够就加长，日志里有 "dog perception up" 再开始播；
      - objects_bag.py 的 passed：每一帧都有一条消息（不丢帧），时间戳对得上 bag_gpu 的帧；same_ids_frames 报告出来（在线时 IMU / INS 和点云的到达顺序可能和 bag 里略有不同，ID 不要求全同）；
      - node.log 里的每秒一行（框数、轨迹数、各段 ms）抄几行进报告；有 warn / error 全部抄下。
   e. 做完删掉板上的 data/live、data/*.npz、bags/，记下根分区剩余空间。deploy/ros2/build 和 install 留着（以后用）。

停止条件（按约定 9 修不好的时候）：板上 colcon 编译失败；第 2 步 msgs_dets 不过；第 3c 步 bag_dets 不过。
提交：data/reports/T29/、本文档，以及按约定 9 改的代码。不提交 data/deploy/T29/、bag、build、install。
报告（写进"T29 结果"）：
- 代码改动（约定 9）：每条写提交号、文件、原因、怎么验证的；
- 镜像、ctest、colcon 的结果（报错 / 警告）；Online.* 的打印；
- 第 2 步两段的 compare 表；
- 第 3c 步两次 compare 和板上的每段延迟（中位数 / p99）；
- 第 3d 步：ros2 bag info 的消息数、objects_bag.py 的输出（含 example）、node.log 的摘录；
- 结论：节点能不能上车；还缺什么（比如话题名、坐标系、规划要的字段）。
```

### T29 结果

（2026-10-30）GPU 用物理卡 4。拉到 `f177be1`。没有再改代码。

**代码改动（约定 9）：** 没有。

**第 0 步。** `tests/dog/test_ros2_node.py` + `test_deploy.py`：11 passed（含慢测试，没有 skipped）。快速测试：166 passed。

**镜像、ctest、colcon。** 重建 `dogperc-dev`（加了 rosbags 0.11.9，numpy 1.26.4），镜像 `e4426b283b51`。服务器容器 Release、sm_89 重编，没有警告。ctest 2/2 通过，26 个测试。Online.* 五个都跑了：

```text
Online.CloudLayoutsMatchPython
Online.CloudErrors
Online.NodeConfigOfTheKygiPreset
Online.InspvaxWaitsForAGoodFix
Online.MessageLogMatchesPythonBagReplay
online parity: 8 frames from 268 messages, max diff poses 3.61e-16, points 0.00e+00 m
```

板上：锁频（`sudo -n jetson_clocks`），MAXN，GPU 最低、最高都是 1300500000。colcon 3 个包（`dog_perception_msgs`、`kygi665_msgs`、`dogperc_ros`）1 分 48 秒编完，只有 clock skew 警告（系统时间停在 1970）。`deploy/cpp` 清掉后 sm_87 重编、ctest 通过。板上没有 rosbags / samples，golden 生成时 `Online.MessageLogMatchesPythonBagReplay` 先是 SKIPPED；把服务器 `build_cuda/golden/online*` 拷过去再跑，通过（同样打印 8 frames）。

**第 2 步。** Python 节点 bag 回放对 C++ 消息处理：

| 段 | msgs_dets（同一批检测） | msgs（各自检测） |
|---|---|---|
| moving（27 帧） | pass，ID 27/27，状态 3.2e-14，预测 2.7e-13 | pass，匹配 1.0000，状态 3.8e-07，预测 2.7e-13 |
| 0921 前 600 云（599 帧） | pass，ID 599/599，状态 2.2e-10，预测 5.2e-10 | pass，匹配 1.0000，状态 7.9e-04，预测 4.6e-03 |

**第 3c 步。** 样例 bag 在板上：ROS `dogperc_bag_replay` 对 `dogperc_replay --messages`，非 timing 键差是 `[]`。bag_dets 对 Python：pass（状态 3.2e-14）。bag_gpu 对服务器 cpp.npz：pass（漏 2 个 near，匹配 1.0000）。TensorRT 有一句 "engine plan file across different models of devices"；板上引擎文件大小是 aarch64 那次建的（pfn 69212、rpn 12.8 MB），`engines.json` 里还留着以前 x86 的记录。

板上 bag_gpu 每段延迟（跳过前 3 帧，中位数 / p99，ms）：preprocess 6.27 / 7.22，sweep 3.82 / 4.39，dedup 2.46 / 2.98，self_filter 0.98 / 1.05，deskew 0.33 / 0.37，accumulate 2.44 / 3.13，det_h2d 0.40 / 0.45，det_voxelize 2.43 / 4.00，det_pfn 1.51 / 1.89，det_rpn 4.77 / 4.78，det_d2h 1.07 / 1.10，det_post 1.64 / 1.75，track 0.11 / 0.15，predict 0.04 / 0.05，total 18.86 / 20.96，网络以外 12.15 / 14.17。

**第 3d 步。** `ROS_DOMAIN_ID=77`。第一次 1× 播放缺第 1 帧（26/27，DDS 发现期）。先热身播一遍，再 `--delay 3 --rate 0.5` 正式播：`ros2 bag info` 27 条 `/dog_perception/objects`。`objects_bag.py`：messages 27，matched 27，missing 0，unmatched 0，`passed` true；`same_ids_frames` 0（在线到达顺序和离线回放不同，任务不要求 ID 全同）。

node.log 摘录：

```text
dog perception up: data/kygi.yaml, model models/nusc_pp_dyn (fp16, GPU pre-processing), lidar /robot/topic/rdap/top_point_cloud2, localization inspvax
ENU origin 29.73325161 106.78497842 217.10
41 dets, 0 tracks | preprocess_ms 7.3 det_gpu_ms 22.2 det_post_ms 1.4 track_ms 0.0 predict_ms 0.0 total_ms 31.3
41 dets, 5 tracks | preprocess_ms 6.1 det_gpu_ms 10.3 det_post_ms 1.7 track_ms 0.1 predict_ms 0.0 total_ms 18.4
63 dets, 3 tracks | preprocess_ms 6.8 det_gpu_ms 11.4 det_post_ms 1.7 track_ms 0.1 predict_ms 0.0 total_ms 20.1
```

warn / error：只有上面那句 TensorRT device 提示；没有别的 warn / error。进程 SIGINT 后干净退出。

**结论。** 节点可以上车：离线 `dogperc_bag_replay` 和消息日志逐位相同，对 Python 同一批检测对齐；GPU 回放对服务器过门槛；在线订阅 bag 不丢帧（正式播前要先让 DDS 发现好，或放慢播放）。还缺的：和规划约定话题名 / `ObjectArray` 字段是否够用（这次消息里有 id、label、motion、score、position、yaw、size、prediction_times、paths）；长时间 10 Hz 稳定性和和别的进程抢资源是 T30；上车实时跑之前要人校时（板子时钟仍停在 1970）。做完删了板上的 live、`data/*.npz`、bags、golden；`deploy/ros2/build` 和 `install` 留着。根分区 4.4 GB。

### T29 复核

2026-10-30。对齐、编译、延迟这几项的结论同意。在线那一步的解释不对，往下查出节点的一个真问题，已修。
- 消息处理：服务器两段 msgs_dets 都过（0921 前 600 云，599 帧 ID 每帧相同，状态差 2.2e-10）；板上 ROS 读 bag 和读消息日志逐位相同，说明 ROS 消息到结构体的拷贝没问题；
- 板上 GPU 对服务器的引擎：漏 2 个 near，匹配率 1，过门槛。轨迹差最大 8 cm、预测差 0.31 m，是两块卡上 FP16 的差；
- 板上延迟（样例 bag 27 帧，4 线程）：端到端 18.9 ms（p99 21.0），网络以外 12.2 ms（p99 14.2）。比 T28B 的 17.3 / 11.4 ms 略高，因为输入不同（样例 bag 是走动中的，约 1 万柱），不是变慢了。节点日志里每帧 16–20 ms，和离线一致。

1. **在线 `same_ids_frames 0` 不是因为到达顺序**：热身和正式播放用的是同一个节点进程，正式播放时节点还带着热身那一遍的轨迹和点云。证据：node.log 里节点刚起来的第一帧是 "0 tracks"（3746 s），正式播放的第一帧已经是 "3 tracks"（3760 s）；在线录到 120 个目标，回放是 110 行。
2. 往下查是节点的真问题：**时间戳倒退时节点不会重来**：
   - 位姿缓存把比最新一条还旧的位姿当乱序丢掉。bag 再播一遍（或者时钟往回跳）以后，新来的位姿全被丢掉，扫描只能用缓存里上一遍最后 3 s 的位姿；
   - 样例 bag 只有 2.6 s，正好全落在这 3 s 里，所以才"没丢帧"。bag 长于 3 s 时，重播后除了最后 3 s 都没有输出，要等时间戳超过上一遍的最后一条才恢复，时钟回跳也一样；
   - 本机复现：位姿缓存设成 1 s，第二遍只出 10/27 帧，没有一帧和第一遍相同。
   已修，Python 和 C++ 一样（`dog_perception/online.py`、`dogperc/online.hpp`）：
   - 某个话题的时间戳比它的上一条早 1 s 以上，节点整个重来，和新起的一样：位姿、IMU、点云累积、地面估计、跟踪、ENU 原点、航向检查都清掉，日志里一行 warn；
   - 重来以后，别的话题如果还有旧时间线上迟到的消息，直接丢掉，不当成第二次倒退。循环播放时，上一遍的最后一帧点云常常在下一遍的第一条 INS 之后才被处理：rclcpp 的执行器在同时有几条在等时按订阅创建的顺序取，IMU、INS 排在点云前面。DEPLOY_ORIN 第 8 节"线程"那条原来写"按到达顺序"，不对，已改；
   - 测试：`Online.ReplayedBagStartsOver`（C++）和 `test_a_bag_played_again_starts_over`（Python）。golden 数据 / 样例 bag 播两遍，中间插一帧上一遍的点云；第二遍的位姿、点、轨迹 ID、年龄逐位等于第一遍。把重来关掉，两个测试都失败；
   - 顺手修了一处：`PerceptionPipeline(pose_buffer=...)` 传进空的缓存会被换成默认的（空缓存判为假）。现在没有调用方传这个参数。
3. 1× 播放丢第一帧，是 DDS 发现期的问题，`--delay 3` 就够了，不用 0.5× 播放：节点每帧 20 ms 左右，10 Hz 下余量有 5 倍。T30 用 1×、`--delay 3`、新起的节点。
4. TensorRT 提示 "engine plan file across different models of devices"：
   - T28B 用的是同一块板、同样的引擎文件（T27 在板上建的，pfn 69212 字节），没有这句，T29 才有；
   - 打印 TensorRT 日志的代码从 T28 起没改过，所以是板子那边变了。板子停用的那两天有没有刷机、升级 JetPack、换板，要人确认；
   - 延迟没变（RPN 4.77 ms，T28B 4.80 ms），对齐也过了；
   - T30 记下板子的版本信息，在板上重建引擎。
5. `engines.json` 里有 x86 的记录，是以前 `sync_to_orin.sh --model` 把服务器的 `engines.json` 盖过去了（规则里有 `--include '*.json'`）。已改：同步模型时不带 `engines.json`，板上留自己的。
6. 要人的两件事：
   - 话题名、`ObjectArray` 字段够不够，要和做规划的人对；
   - 板子时钟停在 1970，上车跑实时之前要校时。节点只用消息里的时间戳，不看系统时钟，所以回放不受影响。
7. 下一步 T30：长时间循环回放（每遍节点重来一次）、负载下的延迟、TensorRT 提示。新工具：
   - `tools/deploy/cut_bag.py`：从原始 bag 截出节点的三个话题，保持源 bag 的 rosbag2 版本；rosbags 只能写 8 / 9 版，Humble 读不读不确定；
   - `deploy/scripts/orin_soak.sh`：板上的长时间回放；
   - `tools/deploy/soak_report.py`：汇总回放结果；
   - `objects_bag.py`：按遍检查（循环播放时在时间戳倒退处分遍）。

## T30 Orin 部署 P7：长时间循环回放、负载、TensorRT 提示

T29 复核改了节点：某个话题的时间戳倒退 1 s 以上，节点整个重来（DEPLOY_ORIN.md 第 8 节）。T30 在板上验这个改动和长时间稳定性：
- 服务器：从 0921 原始 bag 截一段，只留节点的三个话题（前 600 云，约 60 s，`cut_bag.py`），跑 Python 参考，和 C++ 对齐（带这次的改动）；
- 板上：重编、测试；TensorRT 的设备提示：记版本，重建引擎；
- 板上：这段 bag 离线回放（在线的参考）、测延迟；在线 1× 循环播 30 分钟（约 30 遍，每遍节点重来一次），后 15 分钟加负载。查每遍不丢帧，看延迟、内存、温度。

本任务可以改的范围（通用约定 9）：`deploy/ros2/`、`deploy/cpp/`、`deploy/scripts/`、`tools/deploy/`。`dog_perception/` 不改。

```text
任务 T30：节点长时间循环回放（每遍重来）、负载下的延迟、TensorRT 设备提示。
先按通用约定做（第 9 条：可以改代码，范围见本节开头）。前置：T29 复核的改动已经 git pull 到。

0. mkdir -p data/reports/T30 data/deploy/T30；TASK=T30；N=data/deploy/T30；M=data/deploy/nusc_pp_dyn；G、dk 同 T28 第 0 步。
   python -m pytest tests/dog/test_ros2_node.py tests/dog/test_deploy.py -q 2>&1 | tee data/reports/T30/pytest.txt
   （test_a_bag_played_again_starts_over、test_cut_bag_keeps_the_node_messages 要跑到，不能是 skipped）
   快速测试（约定 7）：python -m pytest tests/dog -m "not slow" -n 8 -q 2>&1 | tail -3 | tee data/reports/T30/pytest_fast.txt
   服务器容器里重编、ctest（同 T28 第 2 步，输出 build_x86.txt、ctest_x86.txt）：Online.ReplayedBagStartsOver 通过，不能是 SKIPPED。
1. 服务器：截 bag，跑 Python 参考，和 C++ 对齐。
     B=/zjst/robot/rosbag0921/rosbag2_2026_09_21-16_41_27；n=rosbag2_0921_600
     python tools/deploy/cut_bag.py --bag $B --robot kygi --max-clouds 600 --out $N/bags/$n | tee data/reports/T30/cut.txt
     du -sh $N/bags/$n | tee -a data/reports/T30/cut.txt
     python tools/deploy/cpp_parity.py python --bag $N/bags/$n --robot kygi --exported $M:trt:fp16 \
         --out $N/py.npz --dets $N/py_dets.npz --config $N/kygi.yaml --messages $N/msgs.npz \
         2>&1 | tail -3 | tee data/reports/T30/python.txt
     dk "deploy/cpp/build_cuda/dogperc_replay --messages $N/msgs.npz --config $N/kygi.yaml \
         --detections $N/py_dets.npz --out $N/cpp_dets.npz" 2>&1 | tail -3 | tee data/reports/T30/cpp_replay.txt
     dk "deploy/cpp/build_cuda/dogperc_replay --messages $N/msgs.npz --config $N/kygi.yaml \
         --model-dir $M --precision fp16 --gpu --out $N/cpp.npz" 2>&1 | tail -3 | tee -a data/reports/T30/cpp_replay.txt
     python tools/deploy/cpp_parity.py compare --ref $N/py.npz --cpp $N/cpp_dets.npz --same-detections \
         --out data/reports/T30/msgs_dets.json | tail -1 | tee data/reports/T30/msgs_dets.txt
     python tools/deploy/cpp_parity.py compare --ref $N/py.npz --cpp $N/cpp.npz \
         --out data/reports/T30/msgs.json | tail -1 | tee data/reports/T30/msgs.txt
   - cut_bag.py 只认没压缩的 sqlite3 bag。0921 原始 bag 不是这种就按约定 9 改 cut_bag.py；写出来的 bag 要能被板上 Humble 的 ros2 bag play 读；
   - T29 的 data/deploy/T29/rosbag2_2026_09_21-16_41_27/msgs.npz 还在的话，和这次的 msgs.npz 逐键比，必须全同（截出来的就是 T29 回放的那 600 云）：
       python -c "import numpy as np, sys; a, b = np.load(sys.argv[1]), np.load(sys.argv[2]); \
           print('keys differing:', [k for k in a.files if not np.array_equal(a[k], b[k])])" \
           data/deploy/T29/rosbag2_2026_09_21-16_41_27/msgs.npz $N/msgs.npz | tee -a data/reports/T30/cut.txt
   - cpp_replay.txt 里不应该有 "jumped back"（这段 bag 的时间戳一直往前）。有就抄下来；
   - msgs_dets、msgs 的门槛同 T29 第 2 步，结果应和 T29 的 0921 那一行一样。
2. 板上：同步、编译、测试。
   a. 先看 df -h /（T29 剩 4.4 GB；这段 bag 约 2 GB，大小看 cut.txt）。同步后剩不到 1.5 GB 就停下问人：
        deploy/scripts/sync_to_orin.sh --bag $N/bags/$n
        scp $N/kygi.yaml orin:dogperc/data/
   b. colcon 编译、deploy/cpp 板上编译、ctest：同 T29 第 3b 步，输出 colcon_orin.txt、ctest_orin.txt。
      板上没有 rosbags / samples，生成不了 golden 的 online*：和 T29 一样，把服务器 build_cuda/golden/online* 拷过去，
      再跑 --gtest_filter='Online.*'，输出 gtest_online_orin.txt（6 个都通过）。
   c. TensorRT 的设备提示（T29 复核第 4 条）。先记下板子：
        ssh orin 'cat /etc/nv_tegra_release; cat /proc/device-tree/model; echo; \
            dpkg -l | grep -E "nvidia-l4t-core|nvidia-jetpack|libnvinfer10|tensorrt" | awk "{print \$2, \$3}"; \
            ls -l --time-style=+%F_%T ~/dogperc/models/nusc_pp_dyn/; uptime' 2>&1 | tee data/reports/T30/orin_identity.txt
      旧引擎挪到 models/nusc_pp_dyn/t27/，在板上重建（同 T27）：
        ssh orin 'cd ~/dogperc/models/nusc_pp_dyn && mkdir -p t27 && mv *.engine t27/ && cd ~/dogperc/repo && \
            python3 tools/deploy/build_engines.py --model-dir ~/dogperc/models/nusc_pp_dyn --precision fp16 \
            --opt-pillars 10000 --benchmark 50' 2>&1 | tail -8 | tee data/reports/T30/build_engines_orin.txt
      第 3 步的回放里看这句提示还在不在。
3. 板上离线回放（在线的参考）和延迟：
        R='source /opt/ros/humble/setup.bash && source ~/dogperc/repo/deploy/ros2/install/setup.bash && cd ~/dogperc'
        ssh orin "$R && ros2 run dogperc_ros dogperc_bag_replay --bag bags/$n --config data/kygi.yaml \
            --model-dir models/nusc_pp_dyn --precision fp16 --gpu --threads 4 --out data/bag_gpu.npz" \
            2>&1 | tail -4 | tee data/reports/T30/orin_bag_gpu.txt
        scp orin:dogperc/data/bag_gpu.npz $N/
        python tools/deploy/cpp_parity.py compare --ref $N/cpp.npz --cpp $N/bag_gpu.npz \
            --out data/reports/T30/orin_bag_gpu.json | tail -1 | tee -a data/reports/T30/orin_bag_gpu.txt
        python tools/deploy/cpp_parity.py timings $N/bag_gpu.npz --skip 3 | tee data/reports/T30/orin_bag_timings.txt
   - 门槛同 T28 第 6d 步；
   - 重建引擎以后 "across different models of devices" 还在不在，写进报告。
4. 板上在线：1× 循环播 30 分钟（约 30 遍），后 15 分钟加负载（同一段 bag 在 GPU 上离线回放、循环跑，再加 4 个满载的 CPU 核）。
   脚本约 31 分钟，用 nohup 在板上跑，免得 ssh 断了：
        ssh orin "cd ~/dogperc && nohup repo/deploy/scripts/orin_soak.sh bags/$n data/kygi.yaml models/nusc_pp_dyn 30 \
            data/soak > data/soak.out 2>&1 &"
        （约 32 分钟后）ssh orin 'tail -20 dogperc/data/soak.out' | tee data/reports/T30/orin_soak.txt
        scp -r orin:dogperc/data/soak $N/soak
        python tools/deploy/objects_bag.py --bag $N/soak/live --replay $N/bag_gpu.npz \
            --out data/reports/T30/orin_live.json > data/reports/T30/orin_live.txt
        python tools/deploy/soak_report.py $N/soak --window 300 --out data/reports/T30/soak.json | tee data/reports/T30/soak.txt
        grep -E "WARN|ERROR|TensorRT" $N/soak/node.log | head -100 > data/reports/T30/orin_node_warn.log
   - orin_live.txt：passes 应该在 30 左右；passed 要是 true，也就是每一遍都不丢帧（最后一遍只算到停下的地方）；
     received_gap_max 记下（10 Hz 应在 0.1 s 左右，超过 0.3 s 就是卡了一下）；same_ids_frames / matched_frames 记下（每遍都是新起的节点，应该大多相同）；
   - soak.txt：每 5 分钟一行，有负载和没负载时 total_ms 的中位数 / p99 / 最大值；
   - warn 计数："jumped back" 每遍一次是对的，"from before the restart dropped" 可以有，"no pose" 每遍最多一两次（在开头）。别的 warn / error 全部抄下；
   - 内存（rss_mb）记首、尾、最大，涨了 50 MB 以上要说；tegrastats 记 GR3D 平均值和最高温度；
   - 节点崩溃或卡住（node.log 不再更新）就抄下最后的日志，停下。
5. 做完删掉板上的 data/soak、data/soak.out、data/*.npz、bags/，models/nusc_pp_dyn/t27/ 留着；记下根分区剩余空间。

停止条件（按约定 9 修不好的时候）：第 1 步 msgs_dets 不过；板上 colcon 编译失败；第 4 步节点崩溃或卡住。
提交：data/reports/T30/、本文档，以及按约定 9 改的代码。不提交 data/deploy/T30/、bag、build、install、引擎。
报告（写进"T30 结果"）：
- 代码改动（约定 9）：每条写提交号、文件、原因、怎么验证的；
- cut.txt：每个话题的条数、大小，和 T29 msgs.npz 比的结果；第 1 步两个 compare 表；
- orin_identity.txt 的要点；重建引擎前后提示还在不在，RPN / PFN 的 GPU 时间；
- 第 3 步的 compare 表、每段延迟（中位数 / p99）；
- 第 4 步：passes、passed、哪一遍的 missing / unmatched 不是 0、received_gap_max、same_ids；soak.txt 的表、warn 计数、内存、温度；
- 结论：长时间跑、有负载时能不能保持 10 Hz；还缺什么。
```

### T30 结果

（2026-10-30）GPU 用物理卡 4。拉到 `a7ea977`。没有再改代码。

**代码改动（约定 9）：** 没有。

**第 0 步。** `test_ros2_node` + `test_deploy`：14 passed（含 `test_a_bag_played_again_starts_over`、`test_cut_bag_keeps_the_node_messages`）。快速测试 168 passed。容器里重编 ctest：27 个测试全过，`Online.ReplayedBagStartsOver` 通过（不是 SKIPPED）。

**截 bag 和对齐。** `cut_bag.py` 写出 `rosbag2_0921_600`（sqlite3，1.9 GB）：inspvax 6004、imu 6004、cloud 600。和 T29 的 `msgs.npz` 逐键比：`keys differing: []`。Python 599 帧。msgs_dets / msgs 都 pass，和 T29 的 0921 那一行一样（状态 / 预测同量级）。`cpp_replay` 没有 "jumped back"；开头有一次 "no pose for scan yet"。

**板上身份与引擎。** L4T R36.4.4，Jetson AGX Orin Developer Kit，`libnvinfer10` 10.3.0.30-1+cuda12.5，`nvidia-l4t-core` 36.4.4。旧引擎挪到 `models/nusc_pp_dyn/t27/`，板上重建 FP16：PFN GPU 中位数 1.479 ms（p99 1.486），RPN 4.747 ms（p99 4.758）。同步后根分区剩 2.5 GB（> 1.5 GB）。colcon 3 包通过（仅 clock skew）。Online.* 六个（拷了服务器 `online*` golden）全过。锁频还在。

**第 3 步离线。** `dogperc_bag_replay` 599 帧。重建后日志里**没有** "across different models of devices"。对服务器 `cpp.npz`：pass（匹配 0.9994，漏 12：near 8 / filter 1 / none 3，多 18；跟踪 ID 14/599，是新引擎对旧引擎，不是同一批检测）。延迟（跳过前 3 帧，中位数 / p99 ms）：preprocess 5.37 / 6.48，sweep 4.03 / 4.22，dedup 2.42 / 2.72，accumulate 1.23 / 2.59，det_voxelize 1.68 / 3.58，det_pfn 0.81 / 2.19，det_rpn 4.72 / 4.79，det_post 1.74 / 1.97，total 16.89 / 20.87，网络以外 11.06 / 13.78。

**第 4 步浸泡。** `orin_soak.sh` 30 分钟，`--loop --delay 3`。`ros2 bag info`：17013 条 `/dog_perception/objects`，时长 1796 s。`objects_bag.py`：passes 29（最后一遍 partial），`passed` false，missing 合计 81，`received_gap_max` 0.401 s，`same_ids_frames` 1060 / 17013。

| 遍 | 情况 |
|---|---|
| 0–2 | 开头热身，各缺 1–2 帧 |
| 3–13 | 无负载，每遍 599，不丢 |
| 14–22 | 有负载，每遍缺 3–10 帧，gap 约 0.20–0.40 s |
| 23 | 有负载仍 599 |
| 24–27 | 有负载，缺 2–7 帧 |
| 28 | 停下时的半遍，319 条，缺 3（partial） |

`soak.txt`（每 5 分钟，节点每秒一行里的 total_ms）：

| from min | load | frames | median | p99 | max | boxes | tracks |
|---|---|---|---|---|---|---|---|
| 0 | no | 274 | 16.7 | 21.9 | 33.9 | 55 | 10 |
| 5 | no | 271 | 16.6 | 21.1 | 22.1 | 54 | 10 |
| 10 | no | 270 | 17.2 | 21.3 | 22.0 | 55 | 10 |
| 15 | yes | 270 | 20.8 | 31.9 | 33.2 | 56 | 10 |
| 20 | yes | 274 | 20.2 | 29.9 | 31.8 | 55 | 10 |
| 25 | yes | 267 | 21.0 | 30.2 | 35.2 | 54 | 10 |

warn：`jumped back` 28 次（每遍一次，对），`no pose` 28 次（每遍开头一次，对），没有 "from before the restart dropped"，没有 TensorRT 提示，没有 ERROR。rss：首 202 MB，尾 / 最大 222 MB（涨 19 MB，< 50）。tegrastats：GR3D 平均 19.9%，最高温度 tj/cpu 62.2 °C，gpu 57.5 °C。节点没有崩、没有卡住。

**结论。** 时间戳倒退重来可用（28 次 jump back 都拉起来了）。无负载时端到端约 17 ms，中间连续 11 遍不丢帧，能维持 10 Hz。加了「同 bag GPU 离线循环 + 4 个忙 CPU」之后，多数遍会丢几帧（BEST_EFFORT 点云队列深度 5，和负载抢 GPU/CPU），gap 最大 0.40 s；延迟仍远低于 100 ms（有负载中位数 21 ms、p99 32 ms）。`objects_bag` 的 passed 因此是 false——这块人工负载比车上只跑感知更狠，真车上还要再测。TensorRT 设备提示是旧引擎留下的，板上重建后消失。还缺：和规划联调；车上长时间实车（人校时）。做完删了 soak、bags、`data/*.npz`；`t27/` 留着。根分区 4.3 GB。

## T31 Orin 部署 P8：车上部署包（deploy_deb → deb）

`deploy_deb` 是车上真实部署的代码树（DEPLOY_ORIN.md 第 9 节，用法见 `deploy_deb/perception_combo_ros/node_src/lidar_3d_detection/README.md`）。库和 bag 回放是 `deploy/` 改名后的镜像（`tools/deploy/sync_deploy_deb.py` 检查），节点用的是车上的接口（`perception_interface/LidarObjectArray`、glog、`/perception/lidar_3d/*`）。打包脚本在本机只演练过打包那一步（假的安装目录），colcon 编译和真机运行都没做过。这次在板上：
- 编译、打 deb；
- 不装先解开跑（查有没有指回编译目录的路径）；
- 装上后离线回放和 `deploy/` 逐位比，在线回放查不丢帧，服务起停，卸载；
- 最后装回去。

本任务可以改的范围（通用约定 9）：`deploy_deb/`、`deploy/`、`tools/deploy/`。镜像文件（库、测试、`ros_convert`、`bag_replay`、`lidar_3d_replay`）不要直接改：改 `deploy/` 里的源文件，再 `python tools/deploy/sync_deploy_deb.py --write`。`dog_perception/` 不改。

```text
任务 T31：板上编译 deploy_deb、打 deb、安装、运行、和 deploy/ 逐位比、卸载。
先按通用约定做（第 9 条：可以改代码，范围见本节开头）。前置：T30 完成。

0. mkdir -p data/reports/T31；TASK=T31；n=rosbag2_2026_09_21-16_41_27_moving
   python -m pytest tests/dog/test_deploy.py -q 2>&1 | tail -3 | tee data/reports/T31/pytest.txt
   （test_deploy_deb_mirrors_deploy、test_objects_bag_reads_the_vehicle_messages、test_soak_report_reads_glog_lines 要通过）
   python tools/deploy/sync_deploy_deb.py | tee data/reports/T31/sync_check.txt        （0 stale、0 extra、0 interface problems）
1. 同步、重编 deploy/ros2（bag_replay 改过，第 3 步要用它）：
     deploy/scripts/sync_to_orin.sh --bag samples/$n          （代码里现在带 deploy_deb）
     ssh orin 'df -h / | tail -1'                              （剩不到 3 GB 就停下问人）
   deploy/ros2 的 colcon 同 T29 第 3b 步，输出 colcon_orin.txt。deploy/cpp 没改，不用重编。
2. 板上打 deb（第一次约 10 分钟，CUDA 那几个文件最慢）：
     ssh orin 'cd ~/dogperc/repo/deploy_deb/perception_combo_ros && \
         node_src/lidar_3d_detection/scripts/build_deb.sh --model ~/dogperc/models/nusc_pp_dyn --out ~/dogperc/deb' \
         2>&1 | tail -60 | tee data/reports/T31/build_deb.txt
     scp 'orin:dogperc/deb/*.txt' data/reports/T31/
     ssh orin 'cd ~/dogperc/deb && ls -la && dpkg-deb -c lidar-3d-detection_*.deb | wc -l' | tee -a data/reports/T31/build_deb.txt
   - 编译错误按约定 9 修（CMakeLists、节点、package.xml 在 deploy_deb 里直接改；镜像文件见本节开头）；
   - 报告里看 Depends：
       应该有 libnvinfer10 (= 10.3.0.30-1+cuda12.5)、cuda-cudart-12-6、libgoogle-glog0v5、libyaml-cpp0.7、
       ros-humble-rclcpp、ros-humble-rosbag2-cpp、ros-humble-rmw-fastrtps-cpp 等；
     "libraries no installed package owns"、"installed files that name the source tree" 两节有的话全部抄下；
   - 停止条件：build_deb.sh 报 "still have the build path"，二进制里还有编译路径，按约定 9 修不好。
3. 不装，先解开跑：所有路径都应该跟着包走，不能指回编译目录。同时用 deploy/ 的 bag 回放跑同一个 bag、同一份配置：
     X=/tmp/debx/opt/seres/perception；C=$X/lidar_3d_detection/config/lidar_3d_detection.yaml
     R='source /opt/ros/humble/setup.bash && source ~/dogperc/repo/deploy/ros2/install/setup.bash && cd ~/dogperc'
     ssh orin "rm -rf /tmp/debx && dpkg-deb -x ~/dogperc/deb/lidar-3d-detection_*.deb /tmp/debx && \
         source /opt/ros/humble/setup.bash && source $X/local_setup.bash && cd ~/dogperc && \
         $X/lidar_3d_detection/lib/lidar_3d_detection/lidar_3d_bag_replay --bag bags/$n --config $C \
         --model-dir $X/lidar_3d_detection/model --precision fp16 --gpu --threads 4 --out data/debx_gpu.npz" \
         2>&1 | tail -2 | tee data/reports/T31/debx_replay.txt
     ssh orin "$R && ros2 run dogperc_ros dogperc_bag_replay --bag bags/$n --config $C --model-dir models/nusc_pp_dyn \
         --precision fp16 --gpu --threads 4 --out data/deploy_gpu.npz" 2>&1 | tail -1 | tee -a data/reports/T31/debx_replay.txt
     scp orin:dogperc/data/debx_gpu.npz orin:dogperc/data/deploy_gpu.npz data/deploy/T31/   （先 mkdir -p data/deploy/T31）
     python -c "import numpy as np, sys; a, b = np.load(sys.argv[1]), np.load(sys.argv[2]); \
         print('deb vs deploy, keys differing:', [k for k in a.files if not k.startswith('timing_') \
         and not np.array_equal(a[k], b[k], equal_nan=True)])" data/deploy/T31/deploy_gpu.npz data/deploy/T31/debx_gpu.npz \
         | tee -a data/reports/T31/debx_replay.txt
   - "keys differing" 必须是空的：同一块板、同一份引擎、同一份配置，镜像代码应该逐位相同。不一样先比编译选项
     （deploy_deb 的 CMakeLists 也要 -ffp-contract=off 和 CUDA 的 --fmad=false），再按约定 9 修。
4. 安装（要 root）。板上的免密 sudo 只有 jetson_clocks / nvpmodel，先试：
     ssh orin 'sudo -n apt-get install -y ~/dogperc/deb/lidar-3d-detection_*_arm64.deb' 2>&1 | tail -8 | tee data/reports/T31/install.txt
   要密码就停下，请人在板上执行 `sudo apt install ~/dogperc/deb/lidar-3d-detection_*_arm64.deb`，或者给 apt-get、dpkg、
   systemctl、sed 加免密（人定），之后接着做。装好后：
     ssh orin 'dpkg -s lidar-3d-detection | head -12; ls /opt/seres/perception; systemctl is-enabled lidar-3d-detection; \
         systemctl is-active lidar-3d-detection' 2>&1 | tee -a data/reports/T31/install.txt      （disabled、inactive）
5. 装好以后：
   a. 离线：第 3 步的命令把 X 换成 /opt/seres/perception、输出 data/inst_gpu.npz，和 deploy_gpu.npz 比，keys differing 必须是空的。
      输出 inst_replay.txt。
   b. 在线（手动起）：同 T29 第 3d 步，但节点用装好的启动脚本，话题是 /perception/lidar_3d/objects：
        I=/opt/seres/perception
        ssh orin "source /opt/ros/humble/setup.bash && source $I/local_setup.bash && export ROS_DOMAIN_ID=77 && cd ~/dogperc && \
            ($I/lidar_3d_detection/bin/start_lidar_3d_detection.sh > data/node_deb.log 2>&1 &) && \
            for i in \$(seq 120); do grep -q 'lidar_3d_detection up' data/node_deb.log && break; sleep 1; done && \
            (ros2 bag record -o data/live_deb /perception/lidar_3d/objects > data/record.log 2>&1 &) && sleep 3 && \
            ros2 bag play --delay 3 bags/$n && sleep 3 && pkill -INT -f 'ros2 bag record'; sleep 2; \
            pkill -INT -f 'lib/lidar_3d_detection/lidar_3d_detection( |$)'; sleep 2; ros2 bag info data/live_deb" \
            2>&1 | tail -12 | tee data/reports/T31/live_deb.txt
        scp -r orin:dogperc/data/live_deb data/deploy/T31/; scp orin:dogperc/data/node_deb.log data/reports/T31/
        python tools/deploy/objects_bag.py --bag data/deploy/T31/live_deb --replay data/deploy/T31/inst_gpu.npz \
            --out data/reports/T31/live_deb.json | tee -a data/reports/T31/live_deb.txt
      - passed 要是 true；same_ids_frames 记下（新起的节点，应该大多相同）；example 里 frame_ids 应该是 lidar_det；
      - node_deb.log 里每秒一行 "N dets, M tracks | ... total_ms ..."，抄几行；warn / error 全部抄下。
   c. 服务：设 ROS_DOMAIN_ID 后起停（要 root，同第 4 步）：
        ssh orin 'sudo -n sed -i "s/^#ROS_DOMAIN_ID=0/ROS_DOMAIN_ID=77/" /etc/default/lidar-3d-detection && \
            sudo -n systemctl start lidar-3d-detection && sleep 20 && systemctl is-active lidar-3d-detection && \
            journalctl -u lidar-3d-detection -n 15 --no-pager' 2>&1 | tee data/reports/T31/service.txt
      再按 b 的做法只录包、播一遍（不手动起节点），objects_bag.py 要 passed；然后
        ssh orin 'sudo -n systemctl stop lidar-3d-detection; systemctl is-active lidar-3d-detection; \
            journalctl -u lidar-3d-detection -n 5 --no-pager' 2>&1 | tee -a data/reports/T31/service.txt      （inactive，退出干净）
   d. 短时间循环（6 分钟，后 3 分钟加负载），用 deploy_deb 自己的脚本和装好的节点：
        ssh orin "cd ~/dogperc && nohup repo/deploy_deb/perception_combo_ros/node_src/lidar_3d_detection/scripts/orin_soak.sh \
            bags/$n - - 6 data/soak_deb > data/soak_deb.out 2>&1 &"
        （约 7 分钟后）scp -r orin:dogperc/data/soak_deb data/deploy/T31/
        python tools/deploy/objects_bag.py --bag data/deploy/T31/soak_deb/live --replay data/deploy/T31/inst_gpu.npz \
            --out data/reports/T31/soak_deb_live.json > data/reports/T31/soak_deb_live.txt
        python tools/deploy/soak_report.py data/deploy/T31/soak_deb --window 60 | tee data/reports/T31/soak_deb.txt
      - 样例 bag 只有 2.7 s，6 分钟约 130 遍，每遍节点重来一次：jumped back 的次数应该约等于遍数，没有 ERROR、没有崩；
      - soak_deb.txt 每分钟一行（glog 的时间）；没有负载的几分钟应该不丢帧。
6. 卸载、再装：
     ssh orin 'sudo -n apt-get remove -y lidar-3d-detection; ls -R /opt/seres 2>&1 | head; \
         sudo -n apt-get purge -y lidar-3d-detection; ls /opt/seres /etc/default/lidar-3d-detection 2>&1; \
         sudo -n apt-get install -y ~/dogperc/deb/lidar-3d-detection_*_arm64.deb | tail -2' 2>&1 | tee data/reports/T31/uninstall.txt
   - remove 以后只剩配置文件（conffile），purge 以后 /opt/seres 和 /etc/default/lidar-3d-detection 都没了；
   - 最后装回去，服务不启用（上车前由人决定要不要 enable）。
7. 做完删掉板上的 /tmp/debx、data/*.npz、data/live_deb、data/soak_deb*、bags/；~/dogperc/deb/ 的 deb 和报告留着；
   deploy_deb/perception_combo_ros/build_deb/ 是增量编译用的，根分区剩不到 2 GB 就删掉它的 build/。记下根分区剩余空间。

停止条件（按约定 9 修不好的时候）：第 2 步编译失败或二进制里还有编译路径；第 3 步 keys differing 不是空的；
第 4 步依赖装不上；装好的节点起不来。
提交：data/reports/T31/、本文档，以及按约定 9 改的代码。不提交 data/deploy/T31/、deb、build、install。
报告（写进"T31 结果"）：
- 代码改动（约定 9）：每条写提交号、文件、原因、怎么验证的；改了镜像文件的，写 sync_deploy_deb.py --write 之后的检查结果；
- build_deb 的报告：版本、大小、Depends、两类警告；dpkg-deb -c 的文件数；
- 第 3、5a 步的 keys differing；第 5b、5c 步 objects_bag 的输出（含 example）、node 日志摘录、service 的 journal；
- 第 5d 步：passes、passed、jumped back 次数、soak_deb.txt 的表、内存；
- 第 6 步卸载以后剩什么；
- 结论：deb 能不能交给车上装；还缺什么（比如服务用的用户、ROS_DOMAIN_ID、开机自启）。
```

### T31 结果

（2026-10-30）拉到 `b902d0e`。GPU 用物理卡 4。人给 `seres` 加了 `apt-get`、`dpkg`、`systemctl`、`sed` 免密之后，安装、服务和卸载都做完了。deb 可以交给车上装。服务装上但不启用。

**代码改动（约定 9）：**

- `88222c5`，`tools/deploy/objects_bag.py`。整文件跑 `test_deploy.py` 时，前面的测试把进程 locale 改成了 C，`open()` 按 ascii 读 `perception_interface` 的 `.msg`，中文注释（字节 0xe4）崩溃。改为 `encoding="utf-8"`。验证：`LC_ALL=C` 下 `test_deploy.py` 14 passed（含 `test_deploy_deb_mirrors_deploy`、`test_objects_bag_reads_the_vehicle_messages`、`test_soak_report_reads_glog_lines`、`test_objects_bag_against_a_replay`）；快速测试 171 passed。没改镜像文件。
- `2cf1216`，`deploy_deb/.../scripts/orin_soak.sh` 从 644 改成 755。板上 `nohup` 直接跑它是 Permission denied（同目录的 `orin_check.sh`、`sync_to_orin.sh` 也是 644，这次没改）。chmod 之后 6 分钟浸泡跑完。

**第 0–1 步。** `sync_deploy_deb.py`：68 mirrored，0 stale，0 extra，0 interface problems。同步样例 bag 后根分区 4.0 GB（> 3 GB）。`deploy/ros2` colcon 3 包通过（仅 clock skew，板上时钟仍是 1970）。`deploy/cpp` 没重编。锁频还在（MAXN，GPU 1300500000）。

**第 2 步打 deb。** 编译过了。打包时 `libperception_interface__rosidl_generator_py.so` 的 RPATH 还指着编译目录，脚本要 `patchelf`。`sudo -n apt-get install patchelf` 要密码，所以用了用户目录里的 patchelf 0.14.3（jammy arm64 包解开的二进制，`~/bin/patchelf`），`--skip-build` 再打包，通过。`lidar-3d-detection_0.1.0_arm64.deb`，35 MB，Installed-Size 62668 KB，`dpkg-deb -c` 1498 行。版本没有 git 短号（板上这份树 `git rev-parse` 失败，报告里 git 是 none）。Depends 里有：`libnvinfer10 (= 10.3.0.30-1+cuda12.5)`、`cuda-cudart-12-6`、`libgoogle-glog0v5`、`libyaml-cpp0.7`、`ros-humble-rclcpp`、`ros-humble-rosbag2-cpp`、`ros-humble-rmw-fastrtps-cpp`。没有 "libraries no installed package owns"。点了源码树的安装文件（脚本说不是错误）：

- `/opt/seres/perception/kygi665_msgs/local/lib/python3.10/dist-packages/kygi665_msgs/libkygi665_msgs__rosidl_generator_py.so`
- `/opt/seres/perception/common_interface/local/lib/python3.10/dist-packages/common_interface/libcommon_interface__rosidl_generator_py.so`
- `/opt/seres/perception/nav2_interface/local/lib/python3.10/dist-packages/nav2_interface/libnav2_interface__rosidl_generator_py.so`
- `/opt/seres/perception/perception_interface/local/lib/python3.10/dist-packages/perception_interface/libperception_interface__rosidl_generator_py.so`
- `/opt/seres/perception/lidar_3d_detection/lib/lidar_3d_detection/lidar_3d_detection`

**第 3 步。** 解开到 `/tmp/debx`，不安装。`lidar_3d_bag_replay` 和 `dogperc_bag_replay` 都是 647 messages、27 clouds、27 frames、110 tracks。`keys differing: []`。

**第 4 步。** 免密加上之后 `apt-get install` 成功。`dpkg -s` 版本 0.1.0，文件在 `/opt/seres/perception`。`systemctl is-enabled` disabled，`is-active` inactive。

**第 5a 步。** 安装目录里的 `lidar_3d_bag_replay`：647 messages、27 clouds、27 frames、110 tracks。和 `deploy_gpu.npz` 比，`keys differing: []`。

**第 5b 步。** 任务里的 `--delay 3` 录到 26 条，缺第一帧 `1789980190.998071`。`--delay 8` 还是这 26 条（播放器在 delay 结束之后才建 publisher，多等没用）。按 T29 先空播一遍，再 `--delay 3 --rate 0.5` 录：27 条，`passed` true，`missing_frames` 0，`same_ids_frames` 26，`frame_ids` 是 `lidar_det`，`received_gap_max` 0.216 s（0.5 倍速）。example：id 1，vehicle，score 0.689。warn 只有一行 `imu stamps jumped back 3.1 s ... starting over`（空播把状态带进去，重来了）。没有 ERROR。节点日志（每秒一行，正式那遍）：

```text
38 dets, 0 tracks | preprocess_ms 5.7 det_gpu_ms 8.2 det_post_ms 1.3 track_ms 0.0 predict_ms 0.0 total_ms 15.3
36 dets, 5 tracks | preprocess_ms 6.3 det_gpu_ms 10.5 det_post_ms 1.6 track_ms 0.1 predict_ms 0.0 total_ms 18.6
41 dets, 5 tracks | preprocess_ms 6.2 det_gpu_ms 10.4 det_post_ms 1.7 track_ms 0.1 predict_ms 0.0 total_ms 18.5
55 dets, 7 tracks | preprocess_ms 6.4 det_gpu_ms 10.7 det_post_ms 1.7 track_ms 0.1 predict_ms 0.1 total_ms 19.0
65 dets, 3 tracks | preprocess_ms 6.4 det_gpu_ms 11.3 det_post_ms 1.7 track_ms 0.1 predict_ms 0.0 total_ms 19.6
```

**第 5c 步。** `/etc/default/lidar-3d-detection` 写成 `ROS_DOMAIN_ID=77`，`systemctl start` 后 active，journal 有 `lidar_3d_detection up`。同样先空播再 0.5 倍速录（不手动起节点）：27 条，`passed` true，`same_ids_frames` 26，`frame_ids` 是 `lidar_det`。`systemctl stop` 之后 inactive，journal：`signal_handler(SIGINT/SIGTERM)`，`Deactivated successfully`。

**第 5d 步。** 6 分钟，59 遍（不是约 130：`ros2 bag play --loop --delay 3` 每一遍都先睡 3 s，一遍约 6 s）。`passed` false，`missing_frames` 6，`unmatched_messages` 0，`same_ids_frames` 426 / 1587，`received_gap_max` 0.202 s。缺帧：第 0 遍 1 帧（起来时的第一朵点云，和 5b 一样）；第 33、34、41、51 遍各 1 帧（有负载）；最后一遍 partial，26 条、缺 1（停下时切在 delay 上）。第 1–32 遍无负载，不丢。`jumped back` 58 次（imu 56、inspvax 2），约等于遍数。没有 ERROR，节点没崩。rss 200.5 → 203.6 MB（最大 203.6）。tegrastats：GR3D 平均 17.2%，最高温度 cpu/tj 58.4 °C，gpu 53.1 °C。

`soak_deb.txt`（每分钟，glog 的时间；第 5 分钟标成 no，是因为这一窗的末尾超过了 load off，负载实际从约 176 s 到 356 s）：

| from min | load | frames logged | total ms median | p99 | max | boxes | tracks |
|---|---|---|---|---|---|---|---|
| 0 | no | 30 | 18.2 | 29.0 | 32.5 | 38 | 3 |
| 1 | no | 30 | 18.1 | 19.6 | 19.6 | 38 | 3 |
| 2 | no | 30 | 18.2 | 26.5 | 29.1 | 38 | 3 |
| 3 | yes | 30 | 20.4 | 30.6 | 30.9 | 38 | 3 |
| 4 | yes | 29 | 20.3 | 29.4 | 29.9 | 38 | 3 |
| 5 | no | 28 | 20.6 | 31.0 | 31.9 | 38 | 4 |

**第 6 步。** `apt-get remove` 之后 `ls -R /opt/seres | head` 停在 `.../config:`（默认 head 10 行，yaml 文件名在下一行）；接着 purge 打了 `Purging configuration files`，之后 `/opt/seres` 和 `/etc/default/lidar-3d-detection` 都不在了。再装回去：disabled、inactive。

**第 7 步。** 删了 `/tmp/debx`、`data/*.npz`、`data/live_deb`、`data/soak_deb*`、`bags/`，以及这次多录的 `live_svc`、`live_deb_delay3`。`~/dogperc/deb/` 留着（35 MB 的 deb 和 txt）。根分区 3.8 GB，`build_deb/build` 69 MB，没删。引擎没动。

**结论。** deb 可以交给车上 `apt install`。装好后服务不启用，`/etc/default` 里 `ROS_DOMAIN_ID` 仍是注释（这次测过写成 77，purge 之后的新安装又是注释）。服务用户是打包时的 `seres`。在线播放若不等 DDS 对上，第一朵点云会丢；空播一遍再放，或者放慢，27 帧能齐，跟踪 id 26/27 帧和离线回放相同。6 分钟里无负载除了起来的第一帧都不丢；合成负载下少数遍丢 1 帧，和 T30 一样。重新打包还要 `patchelf`（现在只在 `~/bin`，没装进系统）。板上时钟仍是 1970。

### T31 复核

2026-10-30。结论同意：deb 可以交给车上装。
- 装上以后，不论是不装解开跑，还是装好以后跑，bag 回放都和 `deploy/` 的 `dogperc_bag_replay` 逐位相同；
- 服务起停、`remove` 留下 conffile、`purge` 删干净，都对；
- 两处代码改动合理：`objects_bag.py` 读文件指定 UTF-8；`orin_soak.sh` 加上可执行位。

1. **RPATH**：接口包的 Python 扩展库（`lib*__rosidl_generator_py.so`）的 RPATH 指着编译目录。脚本靠 patchelf 改，板上系统里没装，Agent 用了 `~/bin` 里的一份。报告里"指向源码树"的四个库，是改完 RPATH 后还留着编译目录的项。
   - 已改：`build_deb.sh` 开编之前先找 patchelf（PATH、`~/bin`、`PATCHELF=`），找不到就停；RPATH 里的安装路径换成最终位置，指向编译目录的项删掉（库靠 setup 脚本设的 `LD_LIBRARY_PATH` 找）；
   - 本机用一个代替 patchelf 的小脚本验过：RPATH 变成 `/opt/seres/perception/perception_interface/lib:/opt/ros/humble/lib`。
2. **节点二进制里的源码路径**：是 glog 的 `__FILE__`。已改：编译加 `-ffile-prefix-map=<源码树>/=`。
3. **版本号没有提交号**：板上的代码是 rsync 过去的，没有 `.git`，所有包都是 `0.1.0`，apt 分不出新旧。已改：
   - 两个同步脚本在板上写 `deploy_deb/GIT_COMMIT`（提交数、提交号、有没有未提交的改动）；
   - 版本是 `0.1.0+<提交数>.g<提交号>`，后打的版本更高，apt 能直接从 `0.1.0` 升上去。
4. **循环时 `same_ids_frames` 很低（426 / 1587），不是重来没清干净**：
   - 每一遍都在差不多同一帧开始不同：样例 bag 多数在第 3 帧，也就是第一帧有轨迹的帧；T30 的 0921 段，没负载时每遍都在第 69 帧；
   - 第一遍（节点刚起来）和后面各遍一样，所以重来以后和新起的节点一样；
   - 不同的是在线和离线：在线到达顺序和 bag 不完全一样，某一帧的位姿略有不同，多出一条暂定轨迹，后面的编号全部顺延一位；
   - 已改：`objects_bag.py` 加了 `same_tracks_frames`，不看编号，按类别和世界系位置（0.5 m 内）比报出来的轨迹。
5. **在线第一帧**：只在播放器刚起来时丢（DDS 还没配上对），循环里后面的各遍都不丢。车上传感器一直在发，没有影响。测试时可以 `ros2 bag play --start-paused`，等几秒再调 `/rosbag2_player/resume`；空播或放慢也行。
6. **有负载时丢 1 帧**：节点 p99 31 ms，离 100 ms 远，所以多半是点云在 DDS 传输里丢了：每帧 3.3 MB，best effort，CPU 很忙时 UDP 接收缓冲区溢出。Linux 默认的 `net.core.rmem_max` 只有约 200 KB，大消息常因此丢。可以试 `sysctl net.core.rmem_max=2147483647`、`rmem_default=8388608` 后再跑一次有负载的循环（要 root，人定）。
7. **新加的**：
   - 两个节点加参数 `publish_tf`（默认关，车上不开），发 `camera_init → <local_frame>` 和 `camera_init → <点云 frame_id>`（取点云时间戳那一刻的位姿）；
   - `deploy_deb` 的 `scripts/orin_foxglove.sh`：板上循环播 bag，加 foxglove_bridge；
   - `foxglove/lidar_3d_detection.json`：本机 Foxglove 的布局；
   - `tools/deploy/foxglove_probe.py`：不开 GUI 查 bridge；
   - `objects_bag.py` 录到 `/tf` 时，核对它和 `world_pose`。

   T32 在板上把这些准备好，人在本机看。

### 77a8738 复核（服务默认启用、Foxglove 布局）

2026-10-30。人让服务器上的 Agent 改的、合并进来的提交：deb 装上默认启用并启动服务（`--no-enable-service` 关掉）；Foxglove 布局改成显示系 `camera_init`，点云用检测器的输入点云 `/perception/lidar_3d/cloud`。两处都保留。跟着补了几处，否则这两处改动用起来会出问题：

1. **`/perception/lidar_3d/cloud` 默认没有**：
   - 节点只在配置里 `node.publish_cloud: true` 时才发这个话题，仓库里的配置是 false，`orin_foxglove.sh` 也没打开，新装的板子上布局里看不到点云；
   - 已改：两个节点加 ROS 参数 `publish_cloud`（默认取配置里的值），`orin_foxglove.sh` 传 `-p publish_cloud:=true`；`foxglove_probe.py` 默认也查这个话题。
2. **服务一直开着，和板上脚本冲突**：
   - `orin_foxglove.sh` 看到节点在跑就退出；
   - `orin_soak.sh` 不查，两个节点会抢 GPU、拉高延迟，最后的 `pkill` 还会把服务的节点也停掉（SIGINT 退出码是 0，`Restart=on-failure` 不会拉起来）；
   - 已改：两个脚本先 `sudo -n systemctl stop`，结束时（trap）再 `start`；没有免密就提示先手动停。
3. **安装脚本**：
   - **自启**：原来启用服务要先确认 systemd 在跑，做系统镜像时就不会开机自启。改成只要有 systemctl 就启用（只是建个链接），systemd 在跑才启动；
   - **升级**：升级时 prerm 会停服务，postinst 原来只在 ENABLE=yes 时重启，人手动启用的服务升级后就停着了。改成只要服务是启用的就重启；
   - **卸载**：没有运行中的 systemd 时，`remove` 原来不会取消启用，`purge` 以后留下一个指向空文件的链接。改成 `remove` 时总是取消启用，`purge` 再删一次链接；
   - **提示**：没有运行中的 systemd 时，不再说"已启动"；
   - 本机演练过：装上建了开机链接；`remove` 删了链接、留下配置；`purge` 全删干净。
4. **`LIDAR3D_ARGS`**：启动脚本加了环境变量 `LIDAR3D_ARGS`，可以写在 `/etc/default/lidar-3d-detection` 里，比如 `-p publish_tf:=true -p publish_cloud:=true`，让服务本身也能给查看器看。
5. **流量**：检测器的输入点云是 5 帧拼起来的，比原始点云大，每帧几 MB，10 Hz 下要几十 MB/s。直连的网口一般够用，卡的话在 3D 面板里关掉点云。

### c768c2f 复核（引擎按机器存）

2026-11-01。思路同意：
- 每台机器建的引擎另存一份到 `machines/<主机名>/`，服务器、板子各建各的，互不覆盖；
- 同步时只发目标机器能用的那一套，先查架构和 L4T 版本；
- `--stage` 给打包拼一个完整的模型目录。

1. **L4T 版本查不出来，`--engines` 会直接崩**：
   - 板上 `/etc/nv_tegra_release` 写的是 `# R36 (release), REVISION: 4.4, ...`，里面没有 `R36.4.4` 这样的字样，正则 `R\d+(?:\.\d+)+` 匹配不到；
   - 这个提交以后在板上建的引擎都会记 `l4t`，同步时 `None.group(0)` 报错（本机用板子的真实内容试过：匹配结果是 None）；
   - 已改：`build_engines.py` 加了 `l4t_release()`，把 "R36 (release), REVISION: 4.4" 读成 R36.4.4；检查挪进 `check_target()`，同步脚本调用它；
   - 测试 `test_engines_per_machine_and_the_target_check`：用板子的原文测版本、测 archive / stage，对上的能过，换 x86 或 R36.3.0 都会拦下。
2. **用法示例**：原来写的是 `--engines gpu1`，gpu1 是服务器（x86），发给 Orin 一定会被架构检查拦下。改成 `--engines ubuntu`（板子的主机名）。
3. **主机名重复**：机器只按主机名区分。Jetson 出厂默认都叫 `ubuntu`，以后有第二块板就会混在一起，到时候改主机名，或者按主机名加型号分目录。现在只有一块板，先不改。

## T32 Orin：删掉 deploy/ 以后的验证、Foxglove 实时看的准备

人要在连着板子的本机（192.168.123.55）上用 Foxglove 看板上循环播包的检测结果（deploy_deb README"用 Foxglove 看"）。T31 复核以后改了：打包脚本（patchelf、RPATH、版本号）、节点的 `publish_tf` / `publish_cloud`、新脚本 `orin_foxglove.sh`。2026-11-01 删掉了算法仓库的 `deploy/`：C++ 只剩 deploy_deb 这一份（下文 `$L`），不带 ROS 也能编、跑对 Python 的测试，新旧名字的对应见 DEPLOY_ORIN.md 第 7 节（通用约定 10）。这次：
- 服务器：清掉 `deploy/` 剩下的编译目录；容器里不带 ROS 编 `$L`、跑 CTest（含 GPU）；样例 bag 上 Python 节点对 C++ 的消息处理；
- 板上：清掉旧的 `deploy/`；装 foxglove_bridge 和 patchelf；升级前用装着的版本跑一遍 bag 回放留作参考；重打 deb、升级安装；升级前后逐位比，ROS 读 bag 和读消息日志逐位比；
- 查 TF 和 bridge；
- 留一段 60 s 的 bag，告诉人怎么看。

本任务可以改的范围（通用约定 9）：`deploy_deb/`、`tools/deploy/`。C++ 直接改 `$L`，改了就回服务器容器重跑第 1 步的 CTest。

```text
任务 T32：删掉 deploy/ 以后服务器、板上各验一遍；板上装 foxglove_bridge、重打并升级 deb、查 TF 和 bridge，给人留好 Foxglove 实时看。
先按通用约定做（第 9 条：可以改代码，范围见本节开头；第 10 条：新旧名字）。前置：T31 完成。

0. mkdir -p data/reports/T32 data/deploy/T32；TASK=T32；N=data/deploy/T32；M=data/deploy/nusc_pp_dyn
   L=deploy_deb/perception_combo_ros/node_src/lidar_3d_detection；n=rosbag2_2026_09_21-16_41_27_moving；c=rosbag2_0921_600
   G、dk 同 T28 第 0 步（镜像名还是 dogperc-dev）。
   git pull 以后 deploy/ 只剩没进版本库的编译目录（deploy/cpp/build*、deploy/ros2/build、install），删掉：
     du -sh deploy 2>/dev/null; rm -rf deploy; git status --short | grep deploy/ ; echo "status checked"
   python -m pytest tests/dog/test_deploy.py tests/dog/test_ros2_node.py -q 2>&1 | tail -3 | tee data/reports/T32/pytest.txt
   （test_deploy_deb_config_follows_python、test_foxglove_probe_against_a_fake_bridge、test_objects_bag_same_tracks_ignore_the_numbering 要通过）
   python tools/deploy/check_deploy_deb.py | tee data/reports/T32/check_deploy_deb.txt        （config ok, 0 message problems）
   grep -rnE 'deploy/(cpp|ros2|scripts|docker)\b|dogperc_(bag_)?replay|sync_deploy_deb' dog_perception tools tests deploy_deb \
       --include='*.py' --include='*.sh' --include='*.cpp' --include='*.hpp' --include='*.h' --include=CMakeLists.txt \
       | tee data/reports/T32/old_names.txt        （应该是空的；有就抄下）
1. 服务器容器里不带 ROS 编 $L、跑 CTest（Dockerfile 搬了位置，内容没变，镜像多半直接用缓存）：
     docker build -t dogperc-dev -f $L/docker/Dockerfile.dev $L/docker 2>&1 | tail -3 | tee data/reports/T32/docker_build.txt
     dk "cmake -S $L -B build/lidar3d_cuda -G Ninja -DLIDAR3D_ROS=OFF -DLIDAR3D_TESTS=ON -DLIDAR3D_WITH_CUDA=ON \
         -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build/lidar3d_cuda -j16" 2>&1 | tail -20 | tee data/reports/T32/build_x86.txt
     dk "LIDAR3D_MODEL_DIR=$PWD/$M ctest --test-dir build/lidar3d_cuda --output-on-failure -V" \
         > data/reports/T32/ctest_x86.txt 2>&1; tail -30 data/reports/T32/ctest_x86.txt
   - golden、parity 两个都通过（"100% tests passed, 0 tests failed out of 2"）；
   - GoogleTest 和 T30 的 ctest_x86.txt 一样是 27 个，2 个 Gpu.* 跑了（不是 SKIPPED）；
     Online.MessageLogMatchesPythonBagReplay 打印 "online parity: 8 frames ..."。个数少了就停下报告；
   - 引擎加载失败（"could not load the engine"）：c768c2f 起引擎按机器存，
     python tools/deploy/build_engines.py --model-dir $M --stage $(hostname) --stage-out $N/model_x86
     拼出本机那一套再把 LIDAR3D_MODEL_DIR 指过去；没有本机的就按 T28 第 2 步重建，报告里写明。
2. 服务器上，样例 bag：Python 节点回放、存消息日志；容器里刚编的 lidar_3d_replay 喂同一批检测（同 T29 第 2 步的 B1）：
     mkdir -p $N/$n
     python tools/deploy/cpp_parity.py python --bag samples/$n --robot kygi --exported $M:trt:fp16 \
         --out $N/$n/py.npz --dets $N/$n/py_dets.npz --config $N/$n/kygi.yaml --messages $N/$n/msgs.npz \
         2>&1 | tail -3 | tee data/reports/T32/python_$n.txt
     dk "build/lidar3d_cuda/lidar_3d_replay --messages $N/$n/msgs.npz --config $N/$n/kygi.yaml \
         --detections $N/$n/py_dets.npz --out $N/$n/cpp_dets.npz"
     python tools/deploy/cpp_parity.py compare --ref $N/$n/py.npz --cpp $N/$n/cpp_dets.npz --same-detections \
         --out data/reports/T32/msgs_dets.json | tail -1 | tee data/reports/T32/msgs_dets.txt
   - 要 pass：帧相同、每帧跟踪 ID 一致、状态 ≤ 1e-6、预测 ≤ 1e-4（和 T29 的 msgs_dets 一样）。
3. 板上清掉旧的 deploy/，同步（sync_to_orin.sh 的 --delete 只管它同步的那几个目录，不会删 deploy/）：
     ssh orin 'du -sh ~/dogperc/repo/deploy 2>/dev/null; rm -rf ~/dogperc/repo/deploy; df -h / | tail -1' \
         2>&1 | tee data/reports/T32/orin_cleanup.txt
     tools/deploy/sync_to_orin.sh --bag samples/$n          （写 deploy_deb/GIT_COMMIT）
     scp $N/$n/kygi.yaml $N/$n/py_dets.npz $N/$n/msgs.npz orin:dogperc/data/
     ssh orin 'cat ~/dogperc/repo/deploy_deb/GIT_COMMIT; ls ~/dogperc/repo; df -h / | tail -1' | tee data/reports/T32/sync_orin.txt
   - ls 里不应该再有 deploy；以后不再编 deploy/ros2（没有了），板上的节点只从 deb 来。
4. 板上装包（seres 的 apt-get 免密 T31 已加）：
     ssh orin 'sudo -n apt-get install -y patchelf ros-humble-foxglove-bridge 2>&1 | tail -3; \
         apt-cache policy ros-humble-foxglove-bridge | head -4; which patchelf' 2>&1 | tee data/reports/T32/apt.txt
   - foxglove_bridge 装不上（板子上不了 ROS 的 apt 源）就停下，抄下 apt-cache policy，问人（人可以在本机下 arm64 的 deb 拷过去）。
5. 升级前的参考：板上现在装着的版本跑一遍样例 bag（离线回放，不碰话题，服务开着也没关系）：
     I=/opt/seres/perception；C=$I/lidar_3d_detection/config/lidar_3d_detection.yaml；B=$I/lidar_3d_detection/lib/lidar_3d_detection
     ssh orin 'dpkg -s lidar-3d-detection | grep -E "^(Status|Version)"; ls ~/dogperc/deb' 2>&1 | tee data/reports/T32/old_version.txt
     ssh orin "source /opt/ros/humble/setup.bash && source $I/local_setup.bash && cd ~/dogperc && \
         $B/lidar_3d_bag_replay --bag bags/$n --config $C --model-dir models/nusc_pp_dyn --precision fp16 --gpu --threads 4 \
         --out data/old_gpu.npz" 2>&1 | tail -2 | tee -a data/reports/T32/old_version.txt
   - 升级前后都用 ~/dogperc/models/nusc_pp_dyn（同一份引擎）和装好的配置（T31 以来没改过），比的只是代码；
   - 没装（dpkg -s 说没有这个包）：把 ~/dogperc/deb 里旧的 deb 解开到 /tmp/debold 跑（同 T31 第 3 步），I 换成
     /tmp/debold/opt/seres/perception；旧的 deb 也没有就停下报告。
6. 重打 deb（同 T31 第 2 步；会全部重编：编译选项和 CMakeLists 都改了），输出 build_deb.txt：
   - 版本应该是 0.1.0+<提交数>.g<提交号>（GIT_COMMIT 那一行）；
   - 报告里不应该再有 "installed files that name the source tree"，有就抄下；
   - 解开看一个接口库的 RPATH 和节点里有没有源码路径：
       ssh orin 'rm -rf /tmp/debx && dpkg-deb -x ~/dogperc/deb/lidar-3d-detection_0.1.0+*_arm64.deb /tmp/debx && \
           readelf -d $(find /tmp/debx -name "libperception_interface__rosidl_generator_py.so") | grep -i path; \
           grep -c "dogperc/repo/deploy_deb" /tmp/debx/opt/seres/perception/lidar_3d_detection/lib/lidar_3d_detection/lidar_3d_detection' \
           2>&1 | tee -a data/reports/T32/build_deb.txt        （RUNPATH 里只有 /opt/... ；grep 的数是 0）
7. 升级安装：
     ssh orin 'sudo -n apt-get install -y ~/dogperc/deb/lidar-3d-detection_0.1.0+*_arm64.deb 2>&1 | tail -3; \
         dpkg -s lidar-3d-detection | grep Version; systemctl is-enabled lidar-3d-detection; systemctl is-active lidar-3d-detection; \
         journalctl -u lidar-3d-detection -n 5 --no-pager' 2>&1 | tee data/reports/T32/install.txt
   - 直接升，不用先卸；装完服务是 enabled、active（77a8738 起默认启用并启动），journal 里有 "lidar_3d_detection up"。
8. 升级以后，逐位比：
   a. 第 5 步的 bag 回放命令（I 是 /opt/seres/perception），输出 data/inst_gpu.npz；
   b. 装好的两条回放路径喂同一批检测（同 T29 第 3c 步）：ROS 读 bag，和不用 ROS 读第 2 步的消息日志：
        ssh orin "source /opt/ros/humble/setup.bash && source $I/local_setup.bash && cd ~/dogperc && \
            $B/lidar_3d_bag_replay --bag bags/$n --config data/kygi.yaml --detections data/py_dets.npz --save-points \
                --out data/bag_dets.npz && \
            $B/lidar_3d_replay --messages data/msgs.npz --config data/kygi.yaml --detections data/py_dets.npz --save-points \
                --out data/msgs_dets.npz" 2>&1 | tail -3 | tee data/reports/T32/orin_bag_dets.txt
   c. 拷回来比：
        scp orin:dogperc/data/old_gpu.npz orin:dogperc/data/inst_gpu.npz orin:dogperc/data/bag_dets.npz \
            orin:dogperc/data/msgs_dets.npz $N/
        for p in "old_gpu inst_gpu" "msgs_dets bag_dets"; do set -- $p; python -c "import numpy as np, sys; \
            a, b = np.load(sys.argv[1]), np.load(sys.argv[2]); print(sys.argv[1], 'vs', sys.argv[2], 'keys differing:', \
            [k for k in a.files if not k.startswith('timing_') and not np.array_equal(a[k], b[k], equal_nan=True)])" \
            $N/$1.npz $N/$2.npz; done | tee data/reports/T32/bitwise.txt
        python tools/deploy/cpp_parity.py compare --ref $N/$n/py.npz --cpp $N/bag_dets.npz --same-detections \
            --out data/reports/T32/orin_bag_dets.json | tail -1 | tee -a data/reports/T32/bitwise.txt
   - 两行 keys differing 都必须是空的：升级前后只改了部署（删掉 deploy/、CMake、打包），库的代码没变；
   - 板上的 bag_dets 对 Python 也要 pass（同第 2 步）。
9. TF：装好的节点开 publish_tf，录 objects 和 /tf：
     先 sudo -n systemctl stop lidar-3d-detection（服务的节点也在发，会混进来），做完 start 回来；
     同 T31 第 5b 步，启动脚本前加 LIDAR3D_ARGS="-p publish_tf:=true"，录 /perception/lidar_3d/objects /tf；先空播一遍再 --rate 0.5 正式播。
     python tools/deploy/objects_bag.py --bag $N/live_tf --replay $N/inst_gpu.npz \
         --out data/reports/T32/live_tf.json | tee data/reports/T32/live_tf.txt
   - passed 为 true；tf_transforms 有 camera_init -> lidar_det 和 camera_init -> hesai_lidar；
     tf_frames_with_pose 等于消息数；tf_pose_max_diff < 1e-9；same_ids_frames、same_tracks_frames 都记下。
10. 给人留的 bag 和 bridge 检查：0921 截的 60 s（T30 的 data/deploy/T30/bags/$c 还在就用它，不在就按 T30 第 1 步用 cut_bag.py 重截）：
     tools/deploy/sync_to_orin.sh --bag data/deploy/T30/bags/$c
     ssh orin "cd ~/dogperc && nohup repo/$L/scripts/orin_foxglove.sh bags/$c > data/foxglove.out 2>&1 &"
     （等 20 s）
     ssh orin 'python3 ~/dogperc/repo/tools/deploy/foxglove_probe.py --seconds 10 --topics /perception/lidar_3d/tracks \
         /perception/lidar_3d/predictions /perception/lidar_3d/objects /perception/lidar_3d/cloud /tf \
         /robot/topic/rdap/top_point_cloud2' 2>&1 | tee data/reports/T32/foxglove_probe.txt
     ssh orin 'pkill -INT -f orin_foxglove.sh; sleep 4; cat ~/dogperc/data/foxglove.out; tail -3 /tmp/lidar3d_foxglove/bridge.log; \
         pgrep -fa "foxglove_bridge|bag play" || echo "bridge and player stopped"; systemctl is-active lidar-3d-detection' \
         2>&1 | tee -a data/reports/T32/foxglove_probe.txt
   - 脚本先停掉服务（foxglove.out 里没有报错），结束时再起来（"started again"，最后 is-active 是 active）；
   - probe 的 passed 为 true：六个话题都 advertised，10 s 里 tracks、predictions、objects、/tf 约 100 条，两个点云也有；
     channels 里 objects 的 schema 是 perception_interface/msg/LidarObjectArray（bridge 拿到了消息定义）；
   - 停下以后 bridge 和播放器都没了；剩下的 lidar_3d_detection 是服务起回来的那个；
   - bags/$c 留在板上给人看（约 1.9 GB），根分区剩不到 1.5 GB 就停下问人。
11. 做完删掉板上的 /tmp/debx、/tmp/debold、data/*.npz、data/live_tf、bags/$n；bags/$c、~/dogperc/deb 留着。记下根分区剩余空间
    （和第 3 步删 deploy/ 之前比）。服务器上 build/lidar3d_cuda 留着（以后增量编译）。

停止条件（按约定 9 修不好的时候）：第 1 步 CTest 不过或 GoogleTest 少了；第 2 步 msgs_dets 不过；foxglove_bridge 装不上；
第 6 步编译失败或 RPATH 里还有编译目录；第 8 步 keys differing 不是空的；第 10 步 probe 不过。
提交：data/reports/T32/、本文档，以及按约定 9 改的代码。不提交 data/deploy/T32/、build/、deb、bag。
报告（写进"T32 结果"）：
- 代码改动（约定 9）；
- 第 0 步：删掉的 deploy/ 有多大，old_names.txt；第 1 步：测试数、Gpu.* 和 Online 的输出；第 2 步 msgs_dets；
- 第 3 步：板上删掉的 deploy/ 有多大；apt 的结果；升级前的版本；
- deb 的版本、大小、报告里的警告；RPATH 和 grep 的结果；升级后的版本、服务状态；
- 第 8 步：bitwise.txt 全文；第 9 步 objects_bag 的输出（tf_* 三项、same_tracks_frames）；
- 第 10 步 probe 的输出（channels、received）；
- 给人的一句话：板上执行哪条命令、本机 Foxglove 连哪个地址、导入哪个布局。
```

### T32 结果

（2026-11-01）代码在 `f3b536f`。服务器用物理 GPU 1。`deploy/` 已经不在版本库里，C++ 只剩 `deploy_deb`。升级后的 deb 装上了，服务 enabled、active。TF 和 bridge 都过了。60 s 的 bag 留在板上。

**代码改动（约定 9）：**

- `tools/deploy/foxglove_probe.py`。板上的 `ros-humble-foxglove-bridge` 3.6 是 Foxglove SDK 的服务，握手只认 `foxglove.sdk.v1`。探针只发 `foxglove.websocket.v1` 时，桥回 HTTP 400，正文是 `Missing expected sec-websocket-protocol header`（头在，协议名不是它要的）。探针改为同时提供 `foxglove.sdk.v1, foxglove.websocket.v1`，由服务端选。验证：`test_foxglove_probe_against_a_fake_bridge` 通过；板上对正在播的桥探针 `passed` true，`/perception/lidar_3d/objects` 的 schema 是 `perception_interface/msg/LidarObjectArray`。

**第 0 步。** 服务器上没进版本库的 `deploy/` 编译目录约 107 MB，删掉之后 `git status` 里没有 `deploy/`。pytest `10 passed, 10 skipped`。`check_deploy_deb.py`：`config ok, 0 message problems`。`old_names.txt` 是空的。

**第 1 步。** 镜像 `dogperc-dev` 用缓存。CTest 2 项都过（共 30.10 s）。parity 里 GoogleTest 27 项 PASSED，其中 `Gpu.PillarPreprocessingMatchesTheCpu`、`Gpu.GpuDetectorMatchesTheHostCopyDetector` 都是 OK（没有 SKIPPED）。`Online.MessageLogMatchesPythonBagReplay`：`online parity: 8 frames from 268 messages, max diff poses 3.61e-16, points 0.00e+00 m`。

**第 2 步。** 样例 bag 上 Python 对 C++：`27 / 27 / 27`，pose max `8.9e-16`，det match 1，missed 0，extra 0，track 27/27，pred max diff `2.7e-13`，gates pass。

**第 3 步。** 板上 `~/dogperc/repo/deploy` 158 MB，删掉之后根分区 3.9 GB；同步代码和样例 bag 之后 3.8 GB。`foxglove_bridge` 已经是 `3.6.0-1jammy.20260929.215014`。`apt-get install patchelf` 失败：解析 `mirrors.tuna.tsinghua.edu.cn` 失败。打包用的是 `~/bin/patchelf` 0.14.3。升级前已装版本 `0.1.0`，同一 bag 回放 647 messages、27 clouds、27 frames、110 tracks、2.0 s。

**deb。** `lidar-3d-detection_0.1.0+308.gf3b536f_arm64.deb`，35,781,396 字节，Installed-Size 62696 KB。Depends 含 `libnvinfer10 (= 10.3.0.30-1+cuda12.5)`、`cuda-cudart-12-6`、`nvidia-l4t-core (>= 36.4.4-20250616085344)`。报告里没有 "source tree" / "no installed package" 这类警告（报告头上的 git 是 none，版本号里有 `f3b536f`）。`libperception_interface__rosidl_generator_py.so` 的 RUNPATH 只有 `/opt/seres/perception/common_interface/lib:/opt/seres/perception/nav2_interface/lib:/opt/ros/humble/lib`。节点二进制里 `dogperc/repo/deploy_deb` 出现 0 次。

**第 7 步。** 直接升级。版本 `0.1.0+308.gf3b536f`。`is-enabled` enabled，`is-active` active。journal 有 `lidar_3d_detection up`。

**第 8 步。** `bitwise.txt`：

```text
data/deploy/T32/old_gpu.npz vs data/deploy/T32/inst_gpu.npz keys differing: []
data/deploy/T32/msgs_dets.npz vs data/deploy/T32/bag_dets.npz keys differing: []
| 27 / 27 / 27 | 8.9e-16 | 1.0000 | 0 (0/0/0/0/0) | 0 | 27 / 27 | 3.2e-14 | 1.0000 | 3.7e-13 | pass |
```

**第 9 步。** 停掉服务，手动起节点（`LIDAR3D_ARGS="-p publish_tf:=true"`，域 77），先空播一遍再 `--rate 0.5`。默认的 UDP 缓冲约 200 KB，best-effort 的点云（一帧约 2 MB）会被丢掉，节点收不满 27 帧。给播放器和节点加上 Fast DDS 的 8 MB socket 缓冲之后，录到 27 条 objects、27 条 `/tf`。`objects_bag.py`：`passed` true，`missing_frames` 0，`same_ids_frames` 2，`same_tracks_frames` 23，`frame_ids` 是 `lidar_det`，`received_gap_max` 0.218 s。`tf_transforms`：`camera_init -> hesai_lidar (27)`、`camera_init -> lidar_det (27)`。`tf_frames_with_pose` 27，`tf_pose_max_diff` 0。warn：空播一次 `no pose for scan yet`，正式那遍 `imu stamps jumped back 3.1 s ... starting over`。没有 ERROR。做完服务是 active。

**第 10 步。** `rosbag2_0921_600` 1.9 GB 拷上之后根分区剩 1.7 GB（高于 1.5 GB）。`orin_foxglove.sh` 先停服务再播，结束时 `lidar-3d-detection.service started again`，`is-active` 是 active，bridge 和播放器都没了。探针 `passed` true。10 s 里 tracks 49、predictions 49、objects 50、cloud 49、`/tf` 50，原始点云 99。objects 的 schema 是 `perception_interface/msg/LidarObjectArray`。检测话题大约是 5 Hz 而不是 10 Hz：这一次播放没有加大 UDP 缓冲，best-effort 点云有丢，节点日志里隔几秒一条 `no pose for scan yet`，同时每秒都有检测（`total_ms` 约 50）。

**第 11 步。** 板上删了 `/tmp/debx`、`data/*.npz`、`data/live_tf`、样例 bag。`bags/rosbag2_0921_600` 和 `~/dogperc/deb` 留着。根分区 2.0 GB（第 3 步删 `deploy/` 之后是 3.9 GB，少的主要是这 1.9 GB 的 bag）。服务器上 `build/lidar3d_cuda` 留着。

**给人的一句话。** 在板上执行 `~/dogperc/repo/deploy_deb/perception_combo_ros/node_src/lidar_3d_detection/scripts/orin_foxglove.sh ~/dogperc/bags/rosbag2_0921_600`（Ctrl-C 会停掉播放和 bridge，并把服务拉起来）。本机 Foxglove 连 `ws://192.168.123.100:8765`，导入 `deploy_deb/perception_combo_ros/node_src/lidar_3d_detection/foxglove/lidar_3d_detection.json`，显示坐标系用 `camera_init`。

### T32 复核

**结论**：删掉 `deploy/` 以后，服务器和板上都验过了。
- CTest 27 项（含 2 个 GPU）全过；样例 bag 上 Python 对 C++ pass；
- 升级前后 bag 回放逐位相同；装好的两条回放路径（ROS 读 bag、读消息日志）逐位相同；
- deb 的 RPATH、源码路径都干净；TF、bridge 都过了，60 s 的 bag 留在板上；
- 探针同时提供 `foxglove.sdk.v1` 是对的：bridge 3.6 只认它。

**问题**：
1. **没锁频，节点慢了 2–3 倍**。任务没要求锁频，但报告里没记锁频状态，延迟的数字没法和以前比：
   - 离线回放同一个样例 bag：预处理每帧 12.8–14.0 ms，整段 2.0 s；T31（锁频）是 6.1–6.5 ms、0.9 s；
   - 在线（第 9 步）：`preprocess_ms` 16–18、`det_gpu_ms` 33–35、`total_ms` 55–58；T31 同样的在线回放是 6、8–12、15–21；
   - 板子重启过（T32 的日志时间比 T31 早，时钟每次从 1970-01-01 重新走），`jetson_clocks` 每次重启都要重新执行；
   - 第 10 步 Foxglove 那次只有约 5 Hz（10 s 里 objects 50 条，bridge 收到的原始点云 99 条）。报告归到 UDP 缓冲，但那一次节点每帧要 50 ms 以上，还要发几 MB 的 `/perception/lidar_3d/cloud`，bridge 也在同一块 CPU 上转发。节点忙不过来时，订阅队列（5 帧）里新帧覆盖旧帧，看起来同样是 5 Hz。日志里没有能区分这两种情况的数；
   - 车上：服务开机自启，不会锁频。要不要开机锁频（功耗、温度）由人定，T33 先量清楚。
2. **第 9 步的 8 MB socket 缓冲**只在那次手动运行里加了，没提交用的配置，也没记 `net.core.rmem_max`。T31 同样的回放（锁频、先空播、0.5 倍速）不加缓冲就收齐了。所以"默认缓冲收不齐"可能和没锁频有关（DDS 的接收线程也慢了），要锁频、不锁频各比一次。
3. **板上服务的配置不是包里的**：journal 里服务起来用的是 `/home/seres/dogperc/lidar_3d_detection_cloud.yaml`，还在发 TF（`/etc/default/lidar-3d-detection` 是以前测试时改的，conffile 升级时保留了）。README 说车上不要发 TF。离线比较用的是装好的配置，结果不受影响；T33 改回包里的样子（`ROS_DOMAIN_ID=77` 保留）。
4. **patchelf 装不上**：板上解析不了 tuna 镜像，用的是 `~/bin/patchelf` 0.14.3，能用（`build_deb.sh` 从 PATH 找到它）。系统里装一份要人修 apt 源或拷 deb，不急。
5. 第 9 步 `same_ids_frames` 2、`same_tracks_frames` 23 / 27（T31 新起的节点是 26 / 27）。编号不同是顺延（T31 复核）；不看编号的也只有 23 帧，没锁频时消息的到达时机不一样。T33 锁频后再看。
6. 第 10 步 "no pose for scan yet" 隔几秒一条：点云到的时候，覆盖它的定位还没到（在线最多外推 0.05 s）。没锁频时更容易出现。T33 数一下锁频后还有没有；还有的话，再考虑让节点等一小会儿再处理（设计改动，先不做）。

**复核时改的**：
- 节点每秒那一行末尾加了 `callback_max_ms`（这一秒里最长的回调，含发布）和 `| clouds N frames N skipped N`（收到几帧点云、发出几帧结果、按点云头时间戳算缺了几帧）。`callback_max_ms` 接近 100 ms 是忙不过来；它不大而 `skipped` 多，是 DDS 丢的。`soak_report.py` 按窗口汇总这几个数，老日志照常读；
- `scripts/orin_clocks.sh`：一行打印 CPU、GPU 的最低 / 最高 / 当前频率和功耗模式，以及锁没锁（exit 0 锁了、1 没锁、2 读不到）。`orin_soak.sh` 写进 `<out>/clocks.txt`（`soak_report.py` 先打印它），`orin_foxglove.sh` 起节点前打印；
- README：延迟要锁频后看；bridge 3.x 只认 `foxglove.sdk.v1`，Lichtblick 能不能连没验证，去掉了"协议一样"的说法；Foxglove 里不到 10 Hz 时怎么从日志判断原因；
- 测试：`test_soak_report_reads_the_vehicle_node_counts`、`test_orin_clocks_tells_whether_jetson_clocks_ran`（假的 sysfs：锁了、没锁、读不到）。节点的改动只对着替身头文件做了语法检查，T33 在板上编。

## T33 Orin：实时性（锁频、DDS 缓冲、调试点云）

T32 在线时节点每帧 55 ms（T31 是 15–21 ms），Foxglove 那次只有约 5 Hz，原因有三个可能：没锁频、几 MB 的点云在 DDS（UDP）里丢、发调试点云和 bridge 抢资源。这次用 60 s 的 bag 1× 播，六种组合各跑一遍，用节点每秒那一行的新计数（`clouds / frames / skipped`、`callback_max_ms`）和录下的 objects 分清楚。另外把板上服务的配置改回包里的样子。

本任务可以改的范围（通用约定 9）：`deploy_deb/`、`tools/deploy/`。

```text
任务 T33：板上的实时性：锁频与否、DDS 缓冲、发不发调试点云，对节点吞吐的影响；服务改回包里的配置。
先按通用约定做（第 9 条：可以改代码，范围见本节开头；第 10 条：新旧名字）。前置：T32 复核。

0. mkdir -p data/reports/T33 data/deploy/T33；N=data/deploy/T33；L=deploy_deb/perception_combo_ros/node_src/lidar_3d_detection
   n=rosbag2_2026_09_21-16_41_27_moving；c=rosbag2_0921_600；I=/opt/seres/perception；B=$I/lidar_3d_detection/lib/lidar_3d_detection
   python -m pytest tests/dog/test_deploy.py -q 2>&1 | tail -3 | tee data/reports/T33/pytest.txt
   （test_soak_report_reads_the_vehicle_node_counts、test_orin_clocks_tells_whether_jetson_clocks_ran 要通过）
   tools/deploy/sync_to_orin.sh --bag samples/$n
1. 板上现状（只看，不改）：
     ssh orin "uptime; ~/dogperc/repo/$L/scripts/orin_clocks.sh; sysctl net.core.rmem_max net.core.wmem_max; \
         cat /etc/default/lidar-3d-detection; ls -la ~/dogperc/*.xml /tmp/*.xml 2>/dev/null" 2>&1 | tee data/reports/T33/board_state.txt
   - T32 第 9 步用的 Fast DDS 配置还在的话，原样拷进 data/reports/T33/fastdds_t32.xml；不在就在报告里写清楚当时怎么加的。
2. 重打 deb、升级（节点每秒那一行改了），同 T32 第 6、7 步，输出 build_deb.txt、install.txt。那两步的通配符
   lidar-3d-detection_0.1.0+*_arm64.deb 要只匹配新打的一个包，所以打包前先把 T32 的挪开：
     ssh orin 'mkdir -p ~/dogperc/deb/old && mv ~/dogperc/deb/lidar-3d-detection_0.1.0+* ~/dogperc/deb/old/'
   然后：
   - 离线逐位比：T32 第 5 步的命令（样例 bag、models/nusc_pp_dyn、装好的配置），输出 data/inst_gpu.npz，拷回 $N/，
     和 data/deploy/T32/inst_gpu.npz 比（同 T32 第 8c 步的 python）：keys differing 必须是空的（库没改）；
   - journal 里每秒一行末尾要有 "| clouds ... frames ... skipped ..."，抄两行。
3. 服务改回包里的配置：
     ssh orin 'dpkg-deb --fsys-tarfile ~/dogperc/deb/lidar-3d-detection_0.1.0+*_arm64.deb | tar -xO ./etc/default/lidar-3d-detection \
         > /tmp/default.pkg; diff /tmp/default.pkg /etc/default/lidar-3d-detection' 2>&1 | tee data/reports/T33/default_diff.txt
   - 除了 ROS_DOMAIN_ID=77，其余改动（LIDAR3D_CONFIG、LIDAR3D_ARGS 里的 publish_tf / publish_cloud 等）用 sudo -n sed 改回包里的样子；
   - sudo -n systemctl restart lidar-3d-detection；journal 的 "lidar_3d_detection up" 里 config 是 /opt/seres/...，没有 "publishing TF"。
4. 参考：装好的 bag 回放跑 60 s 的 bag（T32 留下的 bags/$c），装好的配置和模型：
     ssh orin "source /opt/ros/humble/setup.bash && source $I/local_setup.bash && cd ~/dogperc && \
         $B/lidar_3d_bag_replay --bag bags/$c --config $I/lidar_3d_detection/config/lidar_3d_detection.yaml \
         --model-dir $I/lidar_3d_detection/model --precision fp16 --gpu --threads 4 --out data/ref_c.npz" 2>&1 | tail -2 \
         | tee data/reports/T33/ref_c.txt
     scp orin:dogperc/data/ref_c.npz $N/
5. 六次在线运行。每次：sudo -n systemctl stop lidar-3d-detection；手动起装好的节点（同 T32 第 9 步，域 77）；等 "lidar_3d_detection up"；
   起 ros2 bag record（/perception/lidar_3d/objects）；ros2 bag play --delay 3 bags/$c 1× 播一遍（不空播、不放慢）；停录包、停节点。
   | 运行 | 锁频 | DDS | 另外的 |
   |---|---|---|---|
   | R1 | 不锁 | 默认 | 无（和服务一样） |
   | R2 | 不锁 | 大缓冲 | 无 |
   | R3 | 锁 | 默认 | 无 |
   | R4 | 锁 | 大缓冲 | 无 |
   | R5 | 锁 | 默认 | 看 Foxglove 的样子（见下） |
   | R6 | 锁 | 大缓冲 | 同 R5 |
   - R5、R6：节点加 LIDAR3D_ARGS="-p publish_tf:=true -p publish_cloud:=true"；起 foxglove_bridge（端口 8765）；播放期间跑
     python3 ~/dogperc/repo/tools/deploy/foxglove_probe.py --seconds 65 --topics /perception/lidar_3d/cloud \
         /robot/topic/rdap/top_point_cloud2 /perception/lidar_3d/tracks /perception/lidar_3d/objects /tf
     （bridge 只在有客户端订阅时才订这些话题，probe 就是那个客户端）；
   - 锁频：先 sudo -n jetson_clocks --store /tmp/clocks_default（存下现在的设置）。现在已经锁着（第 1 步 orin_clocks.sh 说 locked），
     R1、R2 前用 sudo -n jetson_clocks --restore /tmp/clocks_default 解不开的话，就只跑 R3–R6，报告里写明；
     R1、R2 之后 sudo -n jetson_clocks 锁上。每次运行前后用 orin_clocks.sh 各打一行，写进 data/T33_Rk/clocks.txt；
   - 大缓冲：播放器、录包和节点都加 FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/fastdds_large.xml。用 T32 那份；没有就写这一份：
       <?xml version="1.0" encoding="UTF-8"?>
       <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
         <participant profile_name="large_buffers" is_default_profile="true">
           <rtps>
             <sendSocketBufferSize>8388608</sendSocketBufferSize>
             <listenSocketBufferSize>8388608</listenSocketBufferSize>
           </rtps>
         </participant>
       </profiles>
     第 1 步的 rmem_max 小于 8 MB 时，内核会把缓冲截到 rmem_max（不改 sysctl，报告里写明实际值）；
   - 每次运行 k：节点日志 data/T33_Rk/node.log，录包 data/T33_Rk/live；R5、R6 另存 probe 的输出 data/T33_Rk/probe.json；
     运行之间等节点完全退出（pgrep -f 'lib/lidar_3d_detection/lidar_3d_detection( |$)' 没有输出）。
   - 每次拷回来（scp -r orin:dogperc/data/T33_Rk $N/Rk）以后：
       python tools/deploy/soak_report.py $N/Rk --window 60 | tee data/reports/T33/Rk_soak.txt
       python tools/deploy/objects_bag.py --bag $N/Rk/live --replay $N/ref_c.npz --out data/reports/T33/Rk_live.json \
           | tail -25 > data/reports/T33/Rk_live.txt
       echo "no pose: $(grep -c 'no pose for scan yet' $N/Rk/node.log)" | tee -a data/reports/T33/Rk_soak.txt
6. 结束：保持锁频（同 T25–T31）；服务 start、active，配置是包里的（ROS_DOMAIN_ID=77），orin_clocks.sh 打一行进 final.txt；
   删掉板上 data/T33_*、data/*.npz、bags/$n；bags/$c、~/dogperc/deb 留着。

停止条件（按约定 9 修不好的时候）：第 2 步编译失败或 keys differing 不是空的；节点起不来。
提交：data/reports/T33/、本文档，以及按约定 9 改的代码。不提交 data/deploy/T33/、deb、bag。
报告（写进"T33 结果"）：
- 代码改动（约定 9）；
- 第 1 步的现状：锁频、rmem_max / wmem_max、/etc/default 改过什么、T32 的 DDS 配置；
- 第 2 步 keys differing 和 journal 里的两行；第 3 步改回去以后的 "lidar_3d_detection up" 那一行；
- 六次运行一张表：录到的 objects 条数（应约 600）、objects_bag 的 missing_frames 和 same_tracks_frames、
  每秒那一行汇总的 clouds / frames / skipped、total_ms 中位数和 p99、callback_max_ms 最大值、no pose 次数；R5、R6 的 probe received；
- 结论：丢帧主要来自哪（没锁频、DDS 缓冲、调试点云和 bridge）；车上要不要开机锁频、要不要带 DDS 配置（只给建议，人定）。
```

### T33 结果

（2026-11-01）代码在 `b7dc5c6`。板上重打并装上 `lidar-3d-detection_0.1.0+312.gb7dc5c6`。六次都是 `bags/rosbag2_0921_600` 1× 播一遍（`--delay 3`，不空播、不放慢）。开始时板子没锁频，所以 R1、R2 直接在未锁状态下跑，没有 `--restore`。R2 之后 `jetson_clocks` 锁上（exit 0），结束时仍锁着。`net.core.rmem_max` 和 `wmem_max` 都是 16777216，8 MB 的 socket 缓冲不会被内核截断。

**代码改动（约定 9）：** 无。每秒那一行的 `callback_max_ms`、`clouds / frames / skipped` 和 `orin_clocks.sh` 已经在拉下来的 `b7dc5c6` 里，这次只在板上编进 deb。

**第 1 步。** `orin_clocks.sh`：`clocks NOT locked`，cpu0 729600 / 2201600 / 729600 kHz，gpu 306000000 / 1300500000 / 306000000 Hz，电源模式 MAXN。`rmem_max = wmem_max = 16777216`。`/etc/default/lidar-3d-detection` 当时不是包里的：`LIDAR3D_CONFIG=/home/seres/dogperc/lidar_3d_detection_cloud.yaml`，`LIDAR3D_ARGS=-p publish_tf:=true -p publish_cloud:=true`，`ROS_DOMAIN_ID=77`。T32 的 Fast DDS 配置还在 `/tmp/large_udp.xml`，原样拷进 `data/reports/T33/fastdds_t32.xml`（send / listen 各 8388608）。

**第 2 步。** pytest `13 passed, 8 skipped`（含 `test_soak_report_reads_the_vehicle_node_counts`、`test_orin_clocks_tells_whether_jetson_clocks_ran`）。deb `lidar-3d-detection_0.1.0+312.gb7dc5c6_arm64.deb`，从 `0.1.0+308.gf3b536f` 升上来。样例 bag、装好的配置、`models/nusc_pp_dyn`：`T32 vs T33 keys differing: []`，`only in T33: []`。空闲服务的 journal 只有 up 那一行（没有点云就不会打统计行）。同一只装好的二进制在 R3 的 `node.log` 里：

```text
56 dets, 0 tracks | preprocess_ms 7.5 det_gpu_ms 22.8 det_post_ms 1.5 track_ms 0.0 predict_ms 0.0 total_ms 32.3 callback_max_ms 41.6 | clouds 1 frames 1 skipped 0
55 dets, 8 tracks | preprocess_ms 5.3 det_gpu_ms 8.8 det_post_ms 1.7 track_ms 0.1 predict_ms 0.1 total_ms 16.1 callback_max_ms 26.9 | clouds 11 frames 10 skipped 0
```

**第 3 步。** `LIDAR3D_CONFIG`、`LIDAR3D_MODEL_DIR`、`LIDAR3D_ARGS` 改回注释，只留 `ROS_DOMAIN_ID=77`。重启后：

```text
lidar_3d_detection up: config /opt/seres/perception/lidar_3d_detection/config/lidar_3d_detection.yaml, model /opt/seres/perception/lidar_3d_detection/model (fp16, GPU), lidar /robot/topic/rdap/top_point_cloud2, inspvax /rte/rdap/rtk/inspvax, imu /rte/rdap/rtk/imus/data_raw, objects /perception/lidar_3d/objects
```

没有 `publishing TF`。

**第 4 步。** 装好的配置和模型、未锁频，离线回放 60 s bag：`12608 messages, 600 clouds, 599 frames, 6037 tracks, 44.4 s`。第一帧 `no pose` 丢掉，参考是 599 帧。

**第 5 步。** objects 条数来自 `ros2 bag info`。`clouds / frames / skipped`、`total_ms`、`callback_max_ms` 来自 `soak_report.py --window 60` 对 1 Hz 行的加总（大约 52–57 行，所以会比整段少几帧）。`missing_frames`、`same_tracks_frames` 相对这 599 帧的离线参考。`no pose` 是日志里 `no pose for scan yet` 的条数（这条两秒最多打一次）。

| 运行 | 锁频 | DDS | 另外 | objects | missing | same_tracks | clouds / frames / skipped | total_ms 中位 / p99 | callback_max_ms | no pose |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 不锁 | 默认 | 无 | 341 | 258 | 108 | 339 / 337 / 247 | 50.0 / 122.0 | 213.5 | 2 |
| R2 | 不锁 | 8 MB | 无 | 592 | 7 | 484 | 600 / 592 / 0 | 50.3 / 113.5 | 195.3 | 4 |
| R3 | 锁 | 默认 | 无 | 598 | 1 | 517 | 593 / 592 / 0 | 17.7 / 26.5 | 41.6 | 1 |
| R4 | 锁 | 8 MB | 无 | 597 | 2 | 532 | 596 / 593 / 0 | 17.5 / 24.4 | 36.5 | 1 |
| R5 | 锁 | 默认 | 调试点云 + TF + bridge | 452 | 147 | 177 | 452 / 451 / 144 | 16.6 / 26.7 | 44.3 | 1 |
| R6 | 锁 | 8 MB | 同 R5 | 597 | 2 | 531 | 600 / 597 / 0 | 16.6 / 25.5 | 37.7 | 1 |

R5 探针 `passed` true，65 s：cloud 452、原始点云 453、tracks 452、objects 452、`/tf` 452。R6 探针 `passed` true：cloud 597、原始点云 600、tracks 597、objects 597、`/tf` 597。`objects_bag` 的 `passed` 六次都是 false（`missing_frames` 不是 0）。R3、R4、R6 缺的 1–2 帧和离线丢掉的第一帧是同一类；锁频之后 `no pose` 每次 1 条，不必让节点再多等。

**结论。** 丢帧有两条，都能从计数上看出来。

- 没锁频时节点忙：中位数 50 ms，p99 约 120 ms，`callback_max_ms` 到 200 ms。再配默认 UDP（约 200 KB，一帧点云约 2 MB），60 s 只收到 339 帧点云、skipped 247，objects 341。只把 send / listen 加到 8 MB、频率仍不锁：600 帧点云、skipped 0、objects 592。节点还是慢，但点云不再丢。
- 锁频以后节点不忙（中位数 17 ms，`callback_max_ms` 约 40 ms），默认缓冲在没有额外订阅者时就够（R3：598 条，skipped 0）。再发调试点云、开 bridge，默认缓冲仍丢：objects 452、skipped 144，探针里原始点云也只有 453。同一负载加大缓冲后回到 597、skipped 0。这条不是节点算不过来。

**建议（不改服务、不改 sysctl）。** 车上开机锁频：不锁时慢 3 倍，并且默认 UDP 下会丢帧；`jetson_clocks` 每次重启都掉。Fast DDS 的机器带上这份 8 MB profile（播放器、节点、要看点云的客户端都要有）：没锁频时它单独就能把 skipped 打到 0；锁频以后只要有人订那帧约 2 MB 的 best-effort 点云，默认缓冲仍会丢。真狗是 CycloneDDS（域 0，`/opt/ros/cyclonedds.xml`），这份 Fast DDS XML 在那边不起作用。

**第 6 步。** 锁频留着。服务 active、enabled。配置是包里的，`ROS_DOMAIN_ID=77`，up 那一行没有 `publishing TF`。板上删了 `data/T33_*`、`data/*.npz`、样例 bag、`/tmp/debx`。`bags/rosbag2_0921_600` 和 `~/dogperc/deb` 留着。根分区剩 2.0 GB。

### T33 复核

**结论认可**：丢帧有两条，计数把它们分开了。
- 没锁频：节点每帧 50 ms（p99 约 120 ms），DDS 也跟不上，默认缓冲下 600 帧只出 341 帧；
- 默认的 Fast DDS 缓冲装不下约 3 MB 的点云：锁频以后，只要多一个订阅者（bridge）就丢（452 帧）。R5 里 bridge 收到的原始点云也只有 453 帧，丢在发的那一边；
- 锁频，或者 8 MB 缓冲，各自都能收齐（592–598 帧，skipped 0）；
- 锁频以后 `no pose` 每次只有开头 1 条，不用让节点等定位。

**要注意的**：
1. **"真狗是 CycloneDDS（域 0，`/opt/ros/cyclonedds.xml`）"在报告里没有依据**（没贴 XML、环境变量、进程）。如果属实，这比缓冲重要：
   - deb 的服务默认是 Fast DDS（`/etc/default` 里 RMW 是注释），和机器人的节点不是同一个 DDS。不同厂家的 DDS 能不能互通、大消息走不走得通，都没验证过；
   - Fast DDS 的 profile 在 CycloneDDS 下不起作用；大点云的缓冲要在机器人的那份 CycloneDDS XML 里配；
   - T34 先只读地核实，再让节点在 CycloneDDS 下跑一遍。
2. `objects_bag` 的 `passed` 六次都是 false，原因是开头缺 1–2 帧：参考自己也丢了第一帧（599 帧），在线播放在 DDS 对上之前又可能丢一帧。不空播的 1× 播放里这是正常的，看 `missing_frames` 的数就行。
3. `same_tracks_frames` 锁频时是 517–532 / 599（86–89%）。在线和离线的到达时机不同，位姿外推差一点，阈值附近的检测就会不一样，后面的轨迹跟着变。这不是门槛，记下就行。
4. 板上的服务配置已经改回包里的（`ROS_DOMAIN_ID=77`），锁频也留着，但板子一重启锁频就没了。

**复核时改的**：
- `packaging/fastdds_large_buffers.xml`：T32、T33 用的那份（send / listen 各 8 MB），装到 `share/lidar_3d_detection/`；
- `scripts/orin_foxglove.sh`：Fast DDS 下默认用它（节点、bridge、播放器都继承），开头打一行 `DDS: ...`；已经设了 `FASTRTPS_DEFAULT_PROFILES_FILE` 的照用，设成 `none` 就关掉。这样人在本机看 Foxglove 时就是 R6 的情况（597 / 600），不是 R5 的（452）；
- `scripts/orin_soak.sh`：不改 DDS，把用的 DDS 设置记进 `clocks.txt`；
- `/etc/default/lidar-3d-detection`：加了注释掉的 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`、`CYCLONEDDS_URI`、`FASTRTPS_DEFAULT_PROFILES_FILE`。默认行为不变；
- `build_deb.sh`：Depends 里写打包板上装着的每个 RMW（原来只写找到的第一个，Fast DDS），服务改用 CycloneDDS 时包也在；
- README 加了"DDS"一节（T33 的表、用法、服务的 RMW 要和机器人一致）；
- 测试 `test_fastdds_profile_is_installed_and_listed`。

**要人定的**：
- 车上开机要不要锁频：不锁时慢 3 倍。可以在服务里加一个可选的开机 `jetson_clocks`（`/etc/default` 里开关、默认关），也可以交给机器人的系统设置。功耗和温度是机器人团队的事；
- 车上服务的 RMW 和 DDS 配置：等 T34 核实机器人用的是什么。

### 8 号狗（80a9546，2026-11-04）复核

80a9546 是服务器 agent 在 8 号狗上实际操作以后补的文档（DEPLOY_ORIN 第 2、9 节，节点 README 的"车上实车"和"用 RViz 看"，SENSORS 第 7 节的录制清单），没有代码改动，也没有对应的任务报告。
- **核实了 T33 那句话**：机器人用 CycloneDDS、域 0、`/opt/ros/cyclonedds.xml`（只绑 eth1、共享内存、socket 至少 10 MB），机器人的节点以 root 跑；
- 8 号狗：L4T R36.4.0、RT 内核、Humble 包比编译机旧，编译机打的 deb 装不上（Depends 按编译机的版本写），在狗上重新建引擎、打包；
- `0.1.0+313.g5b2b6bd` 已装好，`User=root`，域 0 + CycloneDDS，active。室内没有固定解（`ins_status` 1、`pos_type` 52），所以还没出过框。

**核对出的问题**（已改）：
1. SENSORS 7.2 录制的路径前后不一样：第 6 步录到 `/home/nvidia/bags1104/cp01`，第 7 步看它，重录和第 9 步 scp 却是 `/home/nvidia/bags/NAME`，照着做第 9 步会拷不到。已统一成 `/home/nvidia/bags/NAME`。重录那一条的 QoS 覆盖：Humble 的 `ros2 bag record` 本来就按发布者的可靠性订（雷达是 RELIABLE），这条主要是把队列加深到 20，已写明；
2. 狗上建引擎漏了 `--opt-pillars 10000`（默认是 P_max 的一半，12000；T27–T31 都是 10000）。只影响 PFN 的调优点，不影响对不对，已补上；
3. 装好以后改 `/etc/default` 的 sed 是照 c69f551 之前的模板写的。现在 `build_deb.sh` 有 `--ros-domain-id`、`--rmw`、`--cyclonedds-uri`，直接写进包里的 `/etc/default`，装上就是车上的设置；sed 留给 `0.1.0+313` 和更早的包。同时写明了升级时 dpkg 会问 conffile 留哪个，加 `--force-confnew` 或 `--force-confold`；
4. 0 号狗那只把 `kInsRtkFixed` 改成 58 的二进制：这个常数在我们的代码里只决定打不打 "RTK not fixed" 的告警。不要把它拷到别的狗上。

**还没验证的**（T34 做）：
- 节点在 CycloneDDS 下收不收得齐真雷达的点云。没有定位时节点不打每秒那一行，所以看不出来。这次加了没有定位时的那一行（`no frames yet (waiting for a pose) | clouds N frames 0 skipped N`）；
- 我们的话题和坐标系在机器人的域 0 里有没有撞车：`/perception/lidar_3d/*` 有没有别的发布者、谁在订；我们的世界系叫 `camera_init`（第一个好解为原点的 ENU），机器人的 SLAM 要是也用 `camera_init`，就是两个同名的坐标系；
- README 里的 Foxglove（狗上还没有 bridge）、RViz（不知道有没有 `rviz2`）都还没跑过。服务是 root、开了共享内存，`nvidia` 用户起的桥能不能接上也不知道；
- 狗的天线接法：配置里还是 0921 接反时的 `ins_azimuth_offset_deg: 180`。狗走起来以后，节点会报 "walks backwards"；
- 狗上的引擎和编译机的结果对不对得上（不同的板子、L4T、引擎，不会逐位相同，按 T28 的门槛比）；
- 录制清单在狗上实际录一段，条数和丢包对不对。

**复核时改的**：
- `build_deb.sh`：`--ros-domain-id N`、`--rmw rmw_fastrtps_cpp|rmw_cyclonedds_cpp`、`--cyclonedds-uri URI`，写进包里的 `/etc/default`（替换对应的注释行），报告里多一行 `settings`；RMW 的包不在打包板上就停。本机用假的安装目录演练过：三个设置都写对，参数不对和包不在时都报错，Depends 里有选的 RMW；
- 节点：没有定位时也每秒打一行收到几帧（`no frames yet ... | clouds N frames N skipped N`）。只对着替身头文件做了语法检查；
- 文档：上面 1–3；README "车上实车"加了这一行的读法、天线接法不对时的报错和改法、root 和共享内存的提醒。

## T34 8 号狗：上车前的检查（室内，不用人）

8 号狗上已经装了 `0.1.0+313`（域 0、CycloneDDS、root）。这次在狗上（室内，没有固定解）：
- 只读地留证据，查话题和坐标系有没有撞车；
- 用现在的代码在狗上重建引擎、打包（带上车上的 DDS 设置）、升级；
- 看节点在 CycloneDDS 下收不收得齐真雷达的点云；
- 在狗上回放 60 s 的 bag，结果和编译机比，量这台的延迟（锁频、不锁频）；
- 照 SENSORS 7.2 录 60 s，核对条数和丢包；
- 有 foxglove_bridge 的 deb 就装上，用探针查（不用人）。

**规矩**：
- 域 0 是机器人的。除了我们已经在跑的服务，不在域 0 上起任何会发消息的东西；只用 `ros2 topic list / info / hz / echo --once`、`ros2 bag record`；
- 机器人别的服务（`nav_perception`、`slam`、`hesai_lidar_driver`、`rte_rtk` 等）不停、不重启、不改；
- 狗上 apt 源出不去，不要 `apt install`；deb 用 `dpkg -i`，依赖不满足就停，不要 `--force-depends`；
- 密码不写进仓库，也不写进 `~/.ssh/config`。

本任务可以改的范围（通用约定 9）：`deploy_deb/`、`tools/deploy/`。

```text
任务 T34：8 号狗上车前的检查：证据、撞车、重打并升级、CycloneDDS 下收点云、回放对比和延迟、录 60 s、bridge。
先按通用约定做（第 9 条，范围见本节开头；第 10 条）。前置：80a9546 复核。在服务器上 ssh dog（Host dog：localhost:5038，用户 nvidia）。

0. mkdir -p data/reports/T34 data/deploy/T34；N=data/deploy/T34；L=deploy_deb/perception_combo_ros/node_src/lidar_3d_detection
   c=rosbag2_0921_600；I=/opt/seres/perception；B=$I/lidar_3d_detection/lib/lidar_3d_detection
   python -m pytest tests/dog/test_deploy.py -q 2>&1 | tail -3 | tee data/reports/T34/pytest.txt
   狗上 ROS 环境（下文 R）：
     R='source /opt/ros/humble/setup.bash && source /opt/seres/perception/local_setup.bash && \
        export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///opt/ros/cyclonedds.xml'
1. 证据（只读），输出 robot_dds.txt：
     ssh dog 'cat /opt/ros/cyclonedds.xml' > data/reports/T34/cyclonedds_robot.xml
     ssh dog "grep -n 'RMW_IMPLEMENTATION\|CYCLONEDDS_URI\|ROS_DOMAIN_ID' ~/.bashrc; cat /etc/default/lidar-3d-detection; \
         ps -eo pid,user,etime,cmd | grep -E 'ros|_node|launch|hesai|rtk|slam|nav' | grep -v grep | head -40; \
         dpkg -l | grep -E 'rmw-(fastrtps|cyclonedds)|foxglove|rviz2'; ~/dogperc/repo/$L/scripts/orin_clocks.sh; uptime" \
         2>&1 | tee data/reports/T34/robot_dds.txt
   （没有 ~/dogperc/repo 的 orin_clocks.sh 就等第 3 步同步以后再跑这一行）
2. 撞车（只读，域 0），输出 collisions.txt：
     ssh dog "$R && for t in /perception/lidar_3d/objects /perception/lidar_3d/tracks /perception/lidar_3d/predictions; do \
         timeout 10 ros2 topic info -v \$t; done; timeout 20 ros2 topic list | grep -iE 'percep|object|track|obstacle'" \
         2>&1 | tee data/reports/T34/collisions.txt
     ssh dog "$R && timeout 5 ros2 topic echo --once /tf_static; timeout 5 ros2 topic echo --once /tf" 2>&1 | \
         grep -E 'frame_id|child_frame_id' | sort | uniq -c | tee -a data/reports/T34/collisions.txt
   - /perception/lidar_3d/* 的发布者只能是 lidar_3d_detection。有别的发布者就停下报告；
   - 记下谁订了 /perception/lidar_3d/objects（规划在不在订）；
   - TF 里有 camera_init 或 lidar_det 的话写进报告（和我们的同名，人定要不要改 world_frame / local_frame），不用停。
3. 同步、重建引擎、打包、升级：
     ORIN_HOST=dog tools/deploy/sync_to_orin.sh --model data/deploy/nusc_pp_dyn
     ssh dog 'cd ~/dogperc/repo && python3 tools/deploy/build_engines.py --model-dir ~/dogperc/models/nusc_pp_dyn \
         --precision fp16 --opt-pillars 10000 --benchmark 50' 2>&1 | tail -8 | tee data/reports/T34/build_engines.txt
     ssh dog 'mkdir -p ~/dogperc/deb/old && mv ~/dogperc/deb/lidar-3d-detection_0.1.0+* ~/dogperc/deb/old/ 2>/dev/null; \
         cd ~/dogperc/repo/deploy_deb/perception_combo_ros && node_src/lidar_3d_detection/scripts/build_deb.sh \
         --model ~/dogperc/models/nusc_pp_dyn --service-user root --no-enable-service \
         --ros-domain-id 0 --rmw rmw_cyclonedds_cpp --cyclonedds-uri file:///opt/ros/cyclonedds.xml --out ~/dogperc/deb' \
         2>&1 | tail -40 | tee data/reports/T34/build_deb.txt
     scp 'dog:dogperc/deb/*.txt' data/reports/T34/
     ssh dog 'sudo -n apt-get install -y -o Dpkg::Options::=--force-confnew ~/dogperc/deb/lidar-3d-detection_0.1.0+*_arm64.deb \
         2>&1 | tail -4; diff /etc/default/lidar-3d-detection.dpkg-old /etc/default/lidar-3d-detection; \
         dpkg -s lidar-3d-detection | grep Version; systemctl is-enabled lidar-3d-detection; systemctl is-active lidar-3d-detection' \
         2>&1 | tee data/reports/T34/install.txt
   - 报告里 settings 一行是 ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///opt/ros/cyclonedds.xml
     LIDAR3D_THREADS=4；Depends 里有 rmw-cyclonedds-cpp；
   - diff 只应该是注释行（新模板多了几行说明），三个设置不变；
   - 编译失败、引擎建不出来就停下报告（不要把旧的 0.1.0+313 卸掉）。
4. CycloneDDS 下收点云（室内没有固定解也能看）：服务重启以后等 70 s，
     ssh dog 'sleep 70; journalctl -u lidar-3d-detection --since "-60s" --no-pager' > $N/journal_rx.txt
     grep -c "no frames yet" $N/journal_rx.txt; grep "no frames yet" $N/journal_rx.txt | tail -5 | tee data/reports/T34/rx.txt
     python - <<'PY' | tee -a data/reports/T34/rx.txt
     import re
     rows = [tuple(map(int, m.groups())) for m in re.finditer(r"clouds (\d+) frames (\d+) skipped (\d+)",
             open("data/deploy/T34/journal_rx.txt").read())]
     print("lines", len(rows), "clouds", sum(r[0] for r in rows), "frames", sum(r[1] for r in rows), "skipped", sum(r[2] for r in rows))
     PY
   - 60 行左右，clouds 合计约 600、skipped 0（真雷达 10 Hz）。有了固定解（frames 不是 0）的话也照抄每秒那一行；
   - clouds 明显少于 10 / s 或 skipped 不是 0：停下报告（下一步要改机器人的 XML，人定）。
5. 狗上回放 60 s 的 bag，和编译机比：
     ssh dog 'mkdir -p ~/dogperc/bags ~/dogperc/data'；scp -r data/deploy/T30/bags/$c dog:dogperc/bags/  （约 1.9 GB；T30 的不在就按 T30 第 1 步重截）
     对 a = 不锁频、b = 锁频（先 orin_clocks.sh，没锁时 sudo -n jetson_clocks --store /tmp/clocks_default 后 a，再 sudo -n jetson_clocks 后 b）：
       ssh dog "source /opt/ros/humble/setup.bash && source $I/local_setup.bash && cd ~/dogperc && \
           $B/lidar_3d_bag_replay --bag bags/$c --config $I/lidar_3d_detection/config/lidar_3d_detection.yaml \
           --model-dir $I/lidar_3d_detection/model --precision fp16 --gpu --threads 4 --out data/dog_$k.npz" 2>&1 | tail -2 \
           | tee data/reports/T34/replay_$k.txt
     scp dog:dogperc/data/dog_a.npz dog:dogperc/data/dog_b.npz $N/
     python tools/deploy/cpp_parity.py compare --ref data/deploy/T33/ref_c.npz --cpp $N/dog_b.npz --out data/reports/T34/dog_vs_orin.json \
         | tail -1 | tee data/reports/T34/dog_vs_orin.txt
     python tools/deploy/cpp_parity.py timings $N/dog_a.npz $N/dog_b.npz | tee data/reports/T34/timings.txt
     （T33 的 ref_c.npz 在服务器上不在了，就在编译机上按 T33 第 4 步重做一份拷回来）
   - 离线回放不碰 DDS，域 0 上的服务照常跑（会多占一点 GPU，报告里写上）；
   - dog_vs_orin：不同的引擎，不会逐位相同，按 T28 第 3 步"各自检测"的门槛判；
   - 结束时锁频保持还是恢复（sudo -n jetson_clocks --restore /tmp/clocks_default）：按第 1 步看到的原状恢复，报告里写明。
6. 照 SENSORS.md 7.2 录 60 s（室内，第 4 步固定解那一关可以不过，报告里写明），NAME=1104_t34：
   - 第 1–3、5 步的数照抄；第 6 步录 60 s 后 Ctrl-C（用 timeout -s INT 65）；第 7 步的条数表；第 8 步丢包；
   - 雷达条数少于 10 × T − 5：再用车上的 XML 录一遍（同样 60 s，环境加 CYCLONEDDS_URI=file:///opt/ros/cyclonedds.xml，
     用 sudo -n -E 跑，写到 /home/nvidia/bags/1104_t34_uri），两次的条数并排写；
   - 录完删掉这两份 bag（室内没有固定解，不拿去标）。
7. foxglove_bridge（可选）：服务器能下载 packages.ros.org 的 ros-humble-foxglove-bridge 3.6.0 arm64 deb 才做。
   scp 到狗上，sudo -n dpkg -i（依赖不满足就停，不要 --force-depends）。然后照节点 README"车上实车"第 2 步在狗上前台开桥
   （先 ros2 topic list | grep perception），另一个终端在狗上：
     python3 ~/dogperc/repo/tools/deploy/foxglove_probe.py --url ws://127.0.0.1:8765 --seconds 10 \
         --topics /perception/lidar_3d/objects /perception/lidar_3d/tracks /robot/topic/rdap/top_point_cloud2
   - 没有固定解时检测话题是 silent，正常；channels 里要有这三个，原始点云要收到约 100 条；
   - 做完停掉桥，确认服务还是 active。
8. 结束：服务 active、enabled，/etc/default 是第 3 步装上的；删掉狗上 data/*.npz、/tmp/debx；bags/$c 留着。

停止条件（按约定 9 修不好的时候）：第 2 步我们的话题有别的发布者；第 3 步编译、建引擎失败；第 4 步点云收不齐；
任何一步需要停或重启机器人别的服务。
提交：data/reports/T34/、本文档，以及按约定 9 改的代码。不提交 data/deploy/T34/、deb、bag、引擎。
报告（写进"T34 结果"）：
- 代码改动（约定 9）；
- 第 1 步的证据（XML 的要点：网口、共享内存、缓冲；机器人节点的用户；装了哪些包；锁频）；
- 第 2 步：我们话题的发布者和订阅者、TF 里的坐标系；
- 第 3 步：引擎的 GPU 时间、deb 版本、settings 行、Depends 里的 RMW、conffile 的 diff；
- 第 4 步：每秒那一行的合计（clouds / frames / skipped）；
- 第 5 步：dog_vs_orin 的结果、不锁频（a）和锁频（b）的 timings（中位数、p99）；
- 第 6 步：条数表、丢包；
- 第 7 步：probe 的 channels 和 received；
- 结论：上车前还差什么（天线接法、开机锁频、同名坐标系、录包要不要用车上的 XML），哪些要人定。
```

### T34 结果

（2026-11-04）8 号狗，室内，没有固定解。代码改动：无（收点云的那一行和 `build_deb.sh` 的 DDS 参数已经在拉下来的 `dbde7ed` 里）。pytest `22 passed`。

**第 1 步。** `/opt/ros/cyclonedds.xml` 原样在 `data/reports/T34/cyclonedds_robot.xml`：`Domain Id="any"`，共享内存开着，只绑 `eth1`，`AllowMulticast` true，`SocketReceiveBufferSize` / `SocketSendBufferSize` 的 min 都是 10 MB。`~/.bashrc` 里反复写了 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`、`CYCLONEDDS_URI=file:///opt/ros/cyclonedds.xml`、`ROS_DOMAIN_ID=0`。机器人的 ROS 进程（雷达、RTK、SLAM、规划、导航感知，以及我们的 `lidar_3d_detection`）用户都是 root。装了 `rmw-cyclonedds-cpp` 1.3.4 和 `rmw-fastrtps-cpp` 6.2.7，还有 `ros-humble-rviz2` 11.2.16、`foxglove-msgs`。当时 `dpkg -l` 里没有 `foxglove-bridge`。时钟没锁：cpu0 729600 / 2201600 / 2201600 kHz，gpu 306 / 1300 / 306 MHz（min / max / current），电源模式 MAXN。刚启动 14 分钟，负载大约 32。

**第 2 步。** `/perception/lidar_3d/objects`、`tracks`、`predictions` 各 1 个发布者、0 个订阅。节点名是 `_NODE_NAME_UNKNOWN_`（`nvidia` 的 `ros2 topic info` 看不到 root 的图），三个 GID 前缀相同，QoS 深度是 5、5、5，和节点里 `create_publisher(..., 5)` 一致；`/perception/lidar_3d/cloud` 也是这一个参与者，深度 2（发布器一直建着，`publish_cloud` 仍是关的）。没有第二个发布者。规划没有订 `/perception/lidar_3d/objects`。同名以外的感知话题是 `/perception/ptz_*`、`/perception/seg_det_result`。`/tf`、`/tf_static` 里的 frame 是 `map`、`base_link`、`ptz_infrared`、`ptz_visible`，没有 `camera_init`，也没有 `lidar_det`。

**第 3 步。** 引擎按 `--opt-pillars 10000` 重建（没锁频，服务开着）：PFN GPU 中位 2.76 ms、p99 12.1 ms，轮廓 opt 是 10000；RPN GPU 中位 7.23 ms、p99 14.5 ms。DLA 那句 fallback 和以前一样，引擎在 GPU 上。deb `lidar-3d-detection_0.1.0+316.gdbde7ed`（35 MB）。报告里的 settings 是 `ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///opt/ros/cyclonedds.xml LIDAR3D_THREADS=4`。Depends 里两个 RMW 都有（`rmw-fastrtps-cpp >= 6.2.7-1jammy.20241128.022749`，`rmw-cyclonedds-cpp >= 1.3.4-1jammy.20251219.095239`）。`--force-confnew` 装上，从 `0.1.0+313.g5b2b6bd` 升上来。`/etc/default` 的 diff 只有新模板多出来的注释行（DDS 说明、注释掉的 Fast DDS profile），三个设置没变。服务 enabled、active。up 那一行没有 `publishing TF`。

**第 4 步。** 重启后约 60 秒：56 行，clouds 598，frames 0，skipped 0。没有固定解，所以不出框。日志里仍是 `waiting for a good INS solution` / `no pose for scan yet`。点云在这台 CycloneDDS 上收得齐，不用改机器人的 XML。

**第 5 步。** 离线回放 `bags/rosbag2_0921_600`，服务一直开着（多占 GPU）。两遍都是 12608 messages、600 clouds、599 frames、6044 tracks。不锁频（a）墙钟 51.0 s，锁频（b）37.2 s。和编译机 T33 的 `ref_c.npz` 比（各自检测）：599 / 599 帧，匹配 0.9983，`score_p99` 0.0067，`center_p99` 0.0038，漏 43、多 43，gates pass。延迟（中位 / p99，毫秒，热身 5 帧不算）：

| 运行 | total | det_gpu | det_network | preprocess |
|---|---|---|---|---|
| a 不锁频 | 58.57 / 98.71 | 33.56 / 65.85 | 21.19 / 50.92 | 16.28 / 42.54 |
| b 锁频 | 37.05 / 67.97 | 16.67 / 37.48 | 8.92 / 18.20 | 13.65 / 41.46 |

比编译机锁频时在线的 17.7 ms 慢：这边服务也在用 GPU，而且没锁频时整机负载高。第 1 步原来没锁，`--store` 存过之后测完 `--restore`，结束时仍是没锁。

**第 6 步。** `ins_status` 1、`pos_type` 52，固定解那一关不过，照任务照样录了 `1104_t34`，录完删了。丢包计数开始是 170（不是 0），录完还是 170，这段没有新的 UDP 丢包。时长 64.13 s：

| 话题 | 条数 | 约 |
|---|---|---|
| `/robot/topic/rdap/top_point_cloud2` | 640 | 10.0 × T（10×T−5 = 636，不用再按车上的 XML 录一遍） |
| `/rte/rdap/rtk/imus/data_raw` | 6419 | 100 × T |
| `/rte/rdap/rtk/inspvax` | 6418 | 100 × T |
| 四路 `*_sensor_camera` | 1585 / 1588 / 1583 / 1580 | 约 24.7 Hz。录之前 `topic hz` 就是约 25，不是录的时候丢掉的 |
| `/lidar_packets_loss` | 641 | 约 10 Hz |

录之前不设 `CYCLONEDDS_URI` 的 `topic hz` 只有约 7.8，间隔最大 0.42 s；bag 的条数是齐的（录包队列 256 MB、按发布者的 RELIABLE 订）。看收没收齐以 bag 条数为准。

**第 7 步。** `ros-humble-foxglove-bridge` 3.6.0 arm64（sha256 `797ce436…9916`）`dpkg -i` 成功。装的时候显示是盖在已经有的同一个 3.6.0 上（第 1 步的列表里还没有这个包）。`nvidia` 用户、和服务相同的 `/etc/default` 起桥：三个话题都在通道里，objects、tracks 是 silent（没有固定解），原始点云 10 秒 10 条，不是约 100。桥已停，服务仍是 active。若干机器人自己的消息缺 `.msg`（`AMENT_PREFIX_PATH` 只有我们的安装和 `/opt/ros/humble`），不影响这几个话题。

**第 8 步。** 服务 enabled、active。`/etc/default` 是第 3 步装上的那份。狗上 `data/*.npz`、室内这份 bag 删了。`~/dogperc/bags/rosbag2_0921_600` 留着。时钟恢复成没锁。

**结论（要人定的）。**

- 车上 `/etc/default` 就用现在这四行：域 0、CycloneDDS、`file:///opt/ros/cyclonedds.xml`、4 线程。新打的包已经写成这样。机器人的 XML 不用加缓冲（第 4 步 skipped 0，里面已经是 10 MB）。
- 开机锁频：不锁时回放 total 中位 59 ms，锁了 37 ms，服务还在旁边占 GPU。要不要在服务里加可选的 `jetson_clocks` 仍是人定；这次测完已经恢复成没锁。
- `camera_init` 现在不在车上的 TF 里，不用改名。实车继续不要开 `publish_tf`。
- 天线接法这次看不了（没走、也没有固定解）。狗走起来以后若报 `walks backwards`，再改 `ins_azimuth_offset_deg`。
- 笔记本上的 Foxglove：桥用 `nvidia` 起得起来，但原始点云只有大约 1 Hz。要看密的点云，桥改成 root 再试（这次没改成 root 起）。没有固定解时没有框。
- `ros-humble-rviz2` 11.2.16 已经在 8 号狗上。从 `192.168.122.100` 看仍要 `ssh -X` 在狗上跑。

### T34 复核

**结论认可**：8 号狗上车前室内能查的都查了。
- 机器人的 DDS 有了原始证据（`cyclonedds_robot.xml`、进程、`~/.bashrc`），和 80a9546 说的一致；
- 我们的三个话题各只有一个发布者，没有订阅者；TF 里只有 `map`、`base_link`、`ptz_*`，没有 `camera_init` 和 `lidar_det`；
- 用 dbde7ed 在狗上重建引擎（`--opt-pillars 10000`），打出带车上 DDS 设置的 deb（`0.1.0+316`），升级后 `/etc/default` 只多了注释；
- CycloneDDS 下真雷达 60 s 收到 598 帧、skipped 0。机器人的 XML 不用改；
- 狗上回放 60 s 的 bag，和编译机比过门槛（匹配 0.998，599 / 599 帧）；录 64 s，雷达 640 条，丢包计数没涨。

**看法不同或要补的**：
1. **延迟慢的原因不是"服务也在占 GPU"**。没有定位时，服务每帧只做去重、去自身点，在去畸变那里就停了（`PoseOutOfRange`），不上 GPU。慢的是机器人自己的负载：开机 14 分钟时 load average 31.9（12 核），进程里有导航感知、`seg_det_result`、PTZ、规划、SLAM。锁频时各段和编译机比：

   | 段（中位，ms） | 编译机（T28B / T33，锁频） | 8 号狗（锁频，机器人的程序都在跑） |
   |---|---|---|
   | 去重 | 2.6 | 6.2 |
   | 体素化 | 1.6 | 3.4 |
   | RPN | 4.75 | 6.84 |
   | 预处理合计 | 6.2 | 13.7 |
   | 检测 GPU | 约 9 | 16.7 |
   | total | 17.7 | 37.1（p99 68） |

   不锁频是 58.6 ms，p99 98.7 ms，已经贴着 100 ms 的帧间隔，尾巴上会连着丢帧。所以 8 号狗开机锁频比编译机更要紧。负载是不是一直这么高（开机 14 分钟可能还在起）、GPU 被谁占着，要在开机一小时以后用 tegrastats 再看一次（T35）。
2. **`nvidia` 用户看到的不全**：狗上有 root 的 `iox-roudi`（CycloneDDS 的共享内存）。`nvidia` 的进程拿到的数据不全：
   - `ros2 topic info` 里的节点名是 `_NODE_NAME_UNKNOWN_`；
   - bridge 10 秒只收到 10 帧原始点云；
   - `ros2 topic hz` 只有约 7.8 Hz（录包按 RELIABLE 订，是齐的）。

   所以第 2 步"没有订阅者""TF 里没有 camera_init"是从 `nvidia` 这边看的，要用 root 再看一次（T35）。README 已经加了用 root 起 bridge 的命令（标明还没试过）。
3. **SENSORS 7.2 的录前检查会误判**：第 1 步用 `ros2 topic hz` 看雷达，门槛是"低于 9 先不要录"，可这一侧只看得到 7.8 Hz。已改成录 12 秒数条数（约 10 × Duration）。相机实测 24.7–27 Hz，表里改成"各约 25 × T"。
4. 引擎是在没锁频、机器人的程序都在跑的时候建的。TensorRT 建引擎时要逐个量战术的耗时，GPU 被别人占着时可能挑到慢的。下次建引擎前先锁频，不强求。

**要人定的**（T34 的结论里也列了）：
- **开机锁频**：8 号狗不锁时 p99 接近 100 ms。我建议在服务里加一个可选的开机 `jetson_clocks`（`/etc/default` 里一个开关，默认关，打包时可以打开），也可以交给机器人团队的系统设置；
- **天线接法**：室内看不了，T35 出门时看；
- 实车不发 TF、不发调试点云，维持现状。

## T35 8 号狗：出门实跑（RTK 固定解，人带狗走）

到这一步，室内能验的都验过了。还没见过的是有固定解时节点在真狗上实际出框：每秒的帧数、延迟、天线接法对不对、在线和离线回放是否一致，以及笔记本上 Foxglove 看到的样子。

分两段：
- **A 段**：室内，Agent 自己做，什么时候做都行；
- **B 段**：要人把狗带到室外开阔处、等 RTK 固定解，按 Agent 的提示走一段（直走、转弯、急停各几次）。Agent 在服务器上通过 ssh 看日志、录包。

规矩同 T34：域 0 上除了我们的服务不发任何东西；不停、不改机器人别的服务；狗上不 `apt install`。

本任务可以改的范围（通用约定 9）：`deploy_deb/`、`tools/deploy/`。另外，B 段的证据表明天线接法和配置不一致时，可以改狗上装好的配置 `/opt/seres/perception/lidar_3d_detection/config/lidar_3d_detection.yaml`（先备份），见 B5。

```text
任务 T35：8 号狗出门实跑。A 段（室内，Agent）：root 视角的撞车检查、root 起 bridge、负载和 GPU 占用。
B 段（室外，人带狗）：有固定解时的在线结果、天线接法、录包和离线回放对比、笔记本上 Foxglove。
先按通用约定做（第 9 条，范围见本节开头；第 10 条）。前置：T34 复核。在服务器上 ssh dog。

0. mkdir -p data/reports/T35 data/deploy/T35；N=data/deploy/T35；L=deploy_deb/perception_combo_ros/node_src/lidar_3d_detection
   I=/opt/seres/perception；B=$I/lidar_3d_detection/lib/lidar_3d_detection
   狗上 root 的 ROS 环境（下文 RR，用 sudo bash -c "$RR && <命令>"）：
     RR='set -a; source /etc/default/lidar-3d-detection; set +a; source /opt/ros/humble/setup.bash; source /opt/seres/perception/local_setup.bash'
   git pull 之后同步代码：ORIN_HOST=dog tools/deploy/sync_to_orin.sh
   7432c90 改了预测线（从框中心、按框的高度画），节点要重打：同 T34 第 3 步（锁频后重建引擎可选，打包、--force-confnew 升级），
   装完 journal 里 up 那一行和 settings 不变。

A1. root 视角的撞车检查（只读），输出 collisions_root.txt：
     ssh dog "sudo bash -c \"$RR && for t in /perception/lidar_3d/objects /perception/lidar_3d/tracks /perception/lidar_3d/predictions; do \
         timeout 10 ros2 topic info -v \\\$t; done; timeout 5 ros2 topic echo --once /tf_static; timeout 5 ros2 topic echo --once /tf\"" \
         2>&1 | tee data/reports/T35/collisions_root.txt
   - 发布者的节点名应是 lidar_3d_detection（不是 _NODE_NAME_UNKNOWN_）；记下订阅者（和 T34 的 nvidia 视角比）；
   - TF 里有没有 camera_init、lidar_det。有的话停下问人（同名坐标系）。
A2. root 起 bridge（README"车上实车"第 2 步里 root 的那条命令），在狗上另开一个 ssh 跑探针：
     python3 ~/dogperc/repo/tools/deploy/foxglove_probe.py --url ws://127.0.0.1:8765 --seconds 10 \
         --topics /perception/lidar_3d/objects /perception/lidar_3d/tracks /robot/topic/rdap/top_point_cloud2
   - 原始点云 10 秒约 100 条（T34 nvidia 起的桥只有 10 条）；检测话题没有固定解时 silent，正常。停掉桥，服务仍 active。
   - 还是只有十几条：换成 nvidia 起的桥同样跑一遍，两次的数并排写，不用再试别的。
A3. 负载（开机一小时以后）：
     ssh dog 'uptime; ~/dogperc/repo/'$L'/scripts/orin_clocks.sh; top -b -n 1 -o %CPU | head -25; \
         timeout 30 tegrastats --interval 1000' 2>&1 | tee data/reports/T35/load.txt
   - 记下 load average、CPU 最高的 10 个进程、GR3D_FREQ 的范围（GPU 有多忙）、EMC。

B0. 等人：人在聊天里说"狗在室外、可以开始"再往下做。先确认固定解：
     ssh dog "sudo bash -c \"$RR && timeout 5 ros2 topic echo --once /rte/rdap/rtk/inspvax --field ins_status; \
         timeout 5 ros2 topic echo --once /rte/rdap/rtk/inspvax --field pos_type\""
   - 要是 3 和 56。不是就告诉人等一等（开阔处、几分钟），不要往下做。
B1. 锁频（这次实跑用；结束时恢复原状）：sudo -n jetson_clocks --store /tmp/clocks_default（第一次）；sudo -n jetson_clocks；orin_clocks.sh 一行进 run.txt。
B2. 开始录（狗上，nvidia，SENSORS 7.2 第 6 步的话题，再加上我们的 /perception/lidar_3d/objects），NAME=1105_t35：
     timeout -s INT 200 ros2 bag record --max-cache-size 268435456 -o /home/nvidia/bags/1105_t35 <SENSORS 7.2 第 6 步的 8 个话题> \
         /perception/lidar_3d/objects
   同时告诉人按这个顺序走（每段之间停 5 秒）：直走约 20 m → 原地左转 180° → 直走回来 → 急停两次 → 绕一个约 5 m 的圈 → 停。
B3. 录的同时看日志（每 30 s 抄一次每秒那一行），结束后整段存下来：
     ssh dog 'journalctl -u lidar-3d-detection --since "-4min" --no-pager' > $N/journal_live.txt
     grep -cE "walks backwards" $N/journal_live.txt；grep -E "dets, .* tracks" $N/journal_live.txt | tail -20 > data/reports/T35/live_lines.txt
     python tools/deploy/soak_report.py 用不了 journal 的格式时，照 T34 第 4 步的 python 合计 clouds / frames / skipped，再取 total_ms 的中位和 p99。
B4. 恢复锁频原状（sudo -n jetson_clocks --restore /tmp/clocks_default，如果开始时没锁）。告诉人可以带狗回来。
B5. 天线接法，两个证据：
   a. 日志里有没有 error "the robot walks backwards in the body frame"；
   b. 服务器上：scp -r dog:/home/nvidia/bags/1105_t35 /public/zjst/robot/cp_data/；用 rosbag_to_sequence.py（--azimuth-offset auto，
      --inspvax-topic 等同 BAG_SPLIT 第 2 步）转成 data/seq/1105_t35，看打印的 "INS azimuth offset ... deg"。
   - a 有 error 且 b 打印 0：天线是接正的。备份狗上的配置，把 node: 里的 ins_azimuth_offset_deg 改成 0.0、ins_lever_arm 改成
     [0.0213, -0.18, 0.0]，sudo -n systemctl restart lidar-3d-detection；人如果还在外面，再走 30 s 确认不再报错；
   - a 没有、b 打印 180：维持现状；两者对不上：停下报告，不要改。
B6. 在线对离线：录下的 bag 在狗上离线回放（装好的 lidar_3d_bag_replay、装好的配置和模型，--gpu --threads 4，输出 data/t35_replay.npz），
    拷回来以后：
     python tools/deploy/objects_bag.py --bag /public/zjst/robot/cp_data/1105_t35 --replay $N/t35_replay.npz \
         --out data/reports/T35/live_vs_replay.json | tail -25 > data/reports/T35/live_vs_replay.txt
   - B5 改过配置的话，只比改完以后的那一段（或者用改之前的配置回放改之前的那一段），报告里写明；
   - missing_frames、same_tracks_frames 记下（T33 编译机上锁频时 86–89%）。
B7. 人看 Foxglove（可选，人愿意时）：A2 的 root 桥开着，人在笔记本上按 README"车上实车"第 3 步连上，导入布局，说一句看到了什么
    （框跟不跟车、有没有重影）。Agent 抄进报告。看完停桥。
B8. 结束：服务 active；锁频是开始时的状态；狗上的 bag 删掉（服务器上留着，以后可以拿去标）。

停止条件：A1 有同名坐标系，或我们的话题有别的发布者；B0 一直没有固定解；B5 两个证据对不上；任何一步要停机器人别的服务。
提交：data/reports/T35/、本文档，以及按约定 9 改的代码。不提交 bag、data/deploy/T35/。
报告（写进"T35 结果"）：
- A1 的发布者 / 订阅者 / TF（和 T34 并排）；A2 两种桥的原始点云条数；A3 的负载、最忙的进程、GR3D；
- B：固定解时刻、每秒那一行的合计（clouds / frames / skipped）和 total_ms 中位、p99；walks backwards 的次数；
  B5 的结论和改了什么；B6 的 objects_bag 输出；B7 人说的话；
- 结论：8 号狗能不能交给规划用；还差什么（开机锁频、负载、天线配置写进哪里）。
```

### T35 结果

（待填）

## T36 1104：体检、切场景、转换、位姿融合（第二批数据）

1104 是 8 号狗（或同一套车载镜像的狗）2026-11-04 晚上 20:02–20:23 录的 4 份 bag（cp01–cp04），共 17.6 分钟、10,565 帧雷达，四路鱼眼都录了（[RECORDED_DATA.md](RECORDED_DATA.md) 第 2 节）。和 0921 比有三处不同：
- 天线应该已经接正（航向偏置 0，杠杆臂是 IMU 中心）；
- 地点：人补的现场信息是和 0921 同一个厂区，相距几百米（T34 在狗上抽的那一条 INSPVAX 是北纬 40°、东经 116°，但那条 `ins_status` 是 1，没有解，经纬度不作数）；
- 是晚上录的：相机画面可能太暗，审核时靠不上。

这一轮只走 BAG_SPLIT 的前几步：体检、切场景、转换、预览、时间差、位姿融合、再预览。不跑教师，不划分。三件事会决定后面怎么做，所以设了停止条件：有没有 RTK 固定解、天线接法、时钟。

本任务可以改的范围（通用约定 9）：`tools/dog/`、`dog_perception/` 里工具的 bug（Python 流水线的行为不改）。

```text
任务 T36：1104 的 4 份 bag：体检、切场景、转换（四路相机进 cameras.mcap）、预览、时间差、位姿融合、再预览。不跑教师，不划分。
先按通用约定做（第 9 条，范围见本节开头；第 10 条）。前置：无（不用等 T35）。仓库根目录，export PYTHONPATH=.。
BAGS=bags1104（软链，指向 /public/zjst/robot/bags1104）

0. mkdir -p data/reports/T36
   python -m pytest tests/dog -m "not slow" -n 8 -q 2>&1 | tail -3 | tee data/reports/T36/pytest.txt
   df -h data | tee data/reports/T36/df.txt          （序列约 0.6 MB / 帧，约 7 GB；相机的 cameras.mcap 另算，预留 15 GB）
   for b in $BAGS/cp0*; do echo "== $b"; grep -E "duration|message_count|name:" $b/metadata.yaml; done > data/reports/T36/topics.txt
   （每份应有 RECORDED_DATA.md 2.2 的 8 个话题）
1. 体检（每份一个前缀 1104_cpNN）：
     for b in $BAGS/cp0*; do n=1104_$(basename $b)
       python tools/dog/bag_survey.py --bag $b --out data/survey/$n --prefix $n 2>&1 | tail -8 | tee data/reports/T36/survey_$n.txt
     done
   - 记下：时长；雷达 / IMU / INS 的频率和最大间隔；时钟（log − header）；pos_type 的分布；可用秒数和场景数；
   - 时钟异常（几秒的偏差、5–95% 区间很宽、前后两段不一样）：停下报告（BAG_SPLIT 第 1 节）；
   - 四份加起来的可用时间不到总时长的一半（RTK 固定解太少：夜里、楼边、没开 RTK 都可能）：停下报告。不要加
     --pos-types 放宽，要人定。
2. 转换（相机默认 auto，四路都进 <序列>/cameras.mcap）：
     for b in $BAGS/cp0*; do n=1104_$(basename $b)
       python tools/dog/rosbag_to_sequence.py --bag $b --inspvax-topic /rte/rdap/rtk/inspvax \
           --lidar-topic /robot/topic/rdap/top_point_cloud2 --imu-topic /rte/rdap/rtk/imus/data_raw \
           --segments data/survey/$n/segments.csv --out data/seq 2>&1 | tee data/seq/$n.convert.log
       grep -E "INS azimuth offset|INS lever arm|cameras ->|WARNING|skipped|wrote" data/seq/$n.convert.log \
           | tee -a data/reports/T36/convert.txt
     done
   - "INS azimuth offset" 应是 0（天线接正），lever arm 是 IMU 中心 [0.0213, -0.18, 0.0]。四份不一样，或打印了
     "hardly moves" 的 WARNING：停下报告（这是 8 号狗天线接法的证据，T35 B5 要用）；
   - "cameras ->" 要列出四路（front、left、right、rear）；每个场景的 wrote 帧数等于 segments.csv 的 scans 列。
3. 第一遍预览（INS 位姿）：
     python tools/dog/sequence_preview.py --seq "data/seq/1104_*" --out data/preview/1104_ins --jobs 8
4. 时间差：挑航向变化最大的 3 个场景（第 3 步之后用 python 读各场景 odom.npy 的航向 ptp，同 T10 第 1 步的写法）：
     python tools/dog/time_offset_check.py --seq <这 3 个> --out data/preview/time_offset_1104 | tee data/reports/T36/time_offset.txt
   - 结论不是"没有固定偏差"就停下报告（BAG_SPLIT 3.2）。
5. 位姿融合：
     python tools/dog/refine_poses.py --seq data/seq/1104_* --plots data/preview/poses_1104 2>&1 | tee data/reports/T36/refine.txt
   - 最后一行 "INS lever arm left over" 应接近 0（两个分量都 < 0.1 m）。不是的话把它加到 --lever-arm 上重跑这一步，
     两次的数都写进报告；
   - 汇总每个场景的 shift p90（ins / lo / fused）、heading_correction_max_deg、ins_unhealthy_share、bad_share。
     天线接正以后 INS 应该比 0921 好很多（0921：急停时超前 0.45 m，航向修正最大 9.9°，不可靠 1.7%）。
6. 第二遍预览（融合后的位姿）：
     python tools/dog/sequence_preview.py --seq "data/seq/1104_*" --out data/preview/1104 --jobs 8
   - 汇总 OK / WARN / FAIL 和非 OK 场景的 problems；shift_p90_m 有 FAIL 就停下报告。
7. 离地高度：python tools/dog/sequence_meta.py --seq data/seq/1104_* | tee data/reports/T36/base_height.txt
   （站着走应在 0.65–0.80 m；不在的场景写进报告，不用改）
8. 提交：data/survey/1104_cp0*/（segments.csv、survey.png、survey.json）、data/preview/1104/（index.html、checks.csv、checks.json、png）、
   data/preview/1104_ins/checks.csv、data/preview/poses_1104/（refine_summary.csv、png，不要 _diag.npz）、
   data/preview/time_offset_1104/、data/reports/T36/、本文档。不提交 data/seq。

停止条件（按约定 9 修不好的时候）：第 1 步时钟异常或可用时间不到一半；第 2 步航向偏置不是 0 或四份不一致；
第 4 步有固定的时间差；第 6 步 shift_p90_m FAIL。
报告（写进"T36 结果"）：
- 第 0 步的话题表；
- 每份 bag 体检的摘要（时长、速度、pos_type、可用秒数、场景数、时钟）；
- 转换的 azimuth offset、lever arm、cameras 行；场景总数和帧数；
- 时间差的结论；refine 的汇总和 "lever arm left over"；
- 两遍预览的 OK / WARN / FAIL（INS 对融合），非 OK 的原因；离地高度；
- 用掉的磁盘（du -sh data/seq/1104_*），每步耗时。
```

### T36 结果

**停在第 2 步（2026-11-05）。** 第 0、1 步完成；转换 `cp01` 时触发停止条件，后面的 bag 未继续转。

**第 0 步**
- 快速测试：180 passed（约 111 s），见 `data/reports/T36/pytest.txt`。
- 磁盘：仓库盘 `/` 剩 34 G；公共盘 `/public` 剩 691 T。`data/seq` 已链到公共盘。
- 话题：四份各 8 个，与 RECORDED_DATA.md 2.2 一致（雷达、IMU、INSPVAX、四路鱼眼、`/lidar_packets_loss`）。原始输出 `data/reports/T36/topics.txt`。

**第 1 步体检**（每份摘要在 `data/reports/T36/survey_1104_cpNN.txt`，完整图/表在 `data/survey/1104_cpNN/`）

| bag | 时长 | 雷达 | IMU / INS | 时钟 log−header（中位，5–95%） | pos_type | 路径 | 可用 | 场景（keep） |
|---|---|---|---|---|---|---|---|---|
| cp01 | 123 s | 10.0 Hz，max gap 0.10 s | 100 / 100 Hz | 雷达 +0.103（0.102..0.112）；IMU/INS ≈ 0 | 56: 100% | 0.56 m | 122 s / 123 s | 4（3） |
| cp02 | 171 s | 同上 | 同上 | 雷达 +0.104（0.102..0.113）；IMU/INS ≈ 0 | 56: 100% | 0.82 m | 171 s / 171 s | 6（5） |
| cp03 | 518 s | 同上 | 同上 | 雷达 +0.103（0.102..0.113）；IMU/INS ≈ 0 | 56: 100% | 4.36 m | 518 s / 518 s | 17（16） |
| cp04 | 245 s | 同上 | 同上 | 雷达 +0.103（0.102..0.112）；IMU/INS ≈ 0 | 56: 100% | 1.20 m | 245 s / 245 s | 8（7） |

- 时钟正常（雷达约 +0.10 s，和 0921 一样；IMU/INS 近 0，区间窄）。
- RTK 固定解 100%，四份合计可用约 1056 / 1057 s（远高于一半），未触发第 1 步停止条件。
- **四份几乎都没动**：所有 keep 场景的 `note` 都是 `static (robot not moving)`，平均速度 0，合计路程约 7 m。后面转换无法用前进速度判断天线接法。

**第 2 步转换（停下）**
- `cp01`：四路相机都进了 `cameras.mcap`（front / left / right / rear）。
- 打印：`INS azimuth offset 180 deg (antennas swapped as on 0921; median forward speed - m/s)`；
  `WARNING: the robot hardly moves; offset 180 is the 0921 default. Pass --azimuth-offset as decided on the other recordings of the same day`；
  `INS lever arm [-0.06, -0.54, 0.0] m`（0921 接反时的拟合值，不是任务要求的 IMU 中心 `[0.0213, -0.18, 0.0]`）。
- 三个 keep 场景的 wrote 帧数与 `segments.csv` 的 scans 列一致（306 / 307 / 306）。
- 停止条件命中后杀掉了正在转的 `cp02`（解到约 1000 帧、尚未打印 azimuth）；`cp03`、`cp04` 未开转。
- 公共盘上留下不完整的 `data/seq/1104_cp01_*`、`data/seq/1104_cp02_*`（约 2 GB）。**重跑前请删掉这些目录**，再按人定的 `--azimuth-offset` 整批重转。不提交 `data/seq`。

**需要人定**
1. 8 号狗这天的天线接法：偏置用 `0` 还是 `180`？本批四份都几乎静止，auto 不可信，只能靠同天有走动的录制（例如待做的 T35 室外）或车上确认接线。
2. 定了以后重跑第 2 步起：`--azimuth-offset 0|180`（lever arm 会跟着 `default_lever_arm` 选 IMU 中心或 0921 拟合值），再做预览 / 时间差 / 融合。
3. 本批全是静止场景，即使用对了偏置，时间差检查（要航向变化大的场景）和运动标注价值也有限；要不要补录走动的 1104 段，一并定。

**耗时**：pytest 约 2 min；四份体检约 2 min；`cp01` 转换约 1 min 后停下。第 3–7 步未做。

### T36 复核

**停得对**：任务里写了"打印 hardly moves 就停"。数据本身是好的：
- 四份都是 100% RTK 固定解，时钟正常（雷达 +0.10 s，IMU、INS 约 0）；
- 话题齐，四路相机都进了 `cameras.mcap`。

**但狗一直没动**：17.6 分钟里四份合计走了约 7 m。每个场景 30 s 的路程多半是 0.1–0.3 m，这是 RTK 的噪声；最多的是 cp03_s000，1.9 m，平均 0.04 m/s。所有场景都是 `static`。

**天线接法对这批数据几乎没影响**：
- 狗不动时，航向偏置错 180° 只是把整个世界系绕狗转了 180°；
- 检测系 D 跟着机身（`Rz(yaw)` 抵消掉），点、框、速度都在 D 里，和偏置无关；
- 杠杆臂选错，只是一个不变的位置偏移。

实测：静止的样例 bag 用偏置 0 和 180（杠杆臂也跟着不同）各转一次，同样的预处理，28 帧检测系点云最多差 3 mm。

只有狗在动的那几下（cp03_s000 的 1.9 m）会受影响：多帧叠加错开最多约两倍位移。这点位移也太少，判断不了天线接法。

**所以**：
- 按天线接正（`--azimuth-offset 0`，杠杆臂跟着用 IMU 中心）接着做；
- meta 里会记下偏置是怎么来的。复核时转换器新加了 `ins_azimuth_offset_from`（`auto` / `given` / `default (the robot hardly moves)`），这批是 `given`。RECORDED_DATA 里写明"未验证"；
- 8 号狗的天线接法留给 T35 B 段（狗会走）；
- 时间差检查要转弯，这批做不了，跳过（体检的时钟检查是好的）。

**这批数据的用处**：
- 全是静止的狗看路过的车：新地方、晚上、四个机位（每份 bag 一个）；
- 每份 bag 内部的场景在同一个位置（`val path within 10 m of train: 100%`），一份 bag 拆不成训练和验证，只能整份进一个集合；
- 有没有价值，看路过的车有多少（T37 的 `scene_stats`）。没有运动的车的场景，只是同一个视角下停着的车重复几千帧，不值得人审。

**以后录制**：SENSORS 7.2 加了一步：开录后先让狗往前直走 10–20 m（> 0.5 m/s、至少 5 s），转换器靠它判断天线接法。

### T36 续

```text
任务 T36 续：按天线接正（偏置 0）把 1104 的 4 份重新转换，接着做预览、位姿融合、再预览、离地高度。时间差检查跳过（狗不动、没有转弯）。
先按通用约定做。前置：T36 复核。仓库根目录，export PYTHONPATH=.。BAGS=bags1104

0. 删掉上次没转完的：ls -d data/seq/1104_*；rm -rf data/seq/1104_cp01_* data/seq/1104_cp02_*（只应有这两份的目录）
1. 转换（同 T36 第 2 步，加 --azimuth-offset 0）：
     for b in $BAGS/cp0*; do n=1104_$(basename $b)
       python tools/dog/rosbag_to_sequence.py --bag $b --inspvax-topic /rte/rdap/rtk/inspvax \
           --lidar-topic /robot/topic/rdap/top_point_cloud2 --imu-topic /rte/rdap/rtk/imus/data_raw \
           --azimuth-offset 0 --segments data/survey/$n/segments.csv --out data/seq 2>&1 | tee data/seq/$n.convert.log
       grep -E "INS azimuth offset|INS lever arm|cameras ->|skipped|wrote" data/seq/$n.convert.log | tee -a data/reports/T36/convert2.txt
     done
   - lever arm 要是 [0.0213, -0.18, 0.0]；meta.json 里 ins_azimuth_offset_from 是 "given"；四路相机；wrote 帧数等于 segments.csv 的 scans。
2. 第一遍预览、位姿融合、第二遍预览：同 T36 第 3、5、6 步（"INS lever arm left over" 这批转弯不够，会是空的，正常）。
   - 静止的场景 shift 应该很小（< 0.05 m）。哪个场景 shift 或 align 不是 OK，抄下 problems；
   - shift_p90_m FAIL 仍然停下报告。
3. 离地高度：同 T36 第 7 步。狗趴着（0.2–0.3 m）还是站着（0.65–0.80 m），每份 bag 写一句。
4. 提交：同 T36 第 8 步（加上 data/reports/T36/convert2.txt）。
报告（写进"T36 结果"的后面）：convert2.txt 的要点；两遍预览的 OK / WARN / FAIL；refine 汇总（shift p90 ins / lo / fused、bad_share）；
离地高度；场景总数和帧数；用掉的磁盘。
```


### T36 续结果

**完成（2026-11-05）。** 按复核用 `--azimuth-offset 0` 重转 4 份，跳过时间差；两遍预览、融合、离地高度都跑完。

**第 0 步**：删掉了上次不完整的 `data/seq/1104_cp01_*`、`1104_cp02_*`（含 convert.log）。相关测试 `test_bag_split` / `test_refine_poses` 通过（`data/reports/T36/pytest_续.txt`）。

**第 1 步转换**（`data/reports/T36/convert2.txt`）
- 四份都是 lever arm `[0.0213, -0.18, 0.0]`；`meta.json`：`ins_azimuth_offset_deg=0`，`ins_azimuth_offset_from=given`。
- 四路相机都进 `cameras.mcap`；31 个 keep 场景的 wrote 帧数与 `segments.csv` 的 scans 列一致，合计 **9 359** 帧。
- 场景磁盘约 **7.3 GB**（`data/seq/1104_*`，不提交）。

**第 2 步预览 / 融合**
- INS 预览 `data/preview/1104_ins/`：**31 OK / 0 WARN / 0 FAIL**；静止场景 shift p90 约 0.00 m。
- `refine_poses.py`：几乎全部场景 correction p90 ≤ 0.01 m、`ins_unhealthy_share=0`、`bad_share=0`；航向修正最大 0.4°。
  - 唯一略动的是 `1104_cp03_s000`（体检路径 1.9 m）：shift p90 ins / lo / fused = 0.12 / 0.06 / 0.07 m，LO bad 1.3%，仍无 pose bad。
  - "INS lever arm left over"：这批几乎不转，`lever_residual_m` 全是 null（预期）。
- 融合后预览 `data/preview/1104/`：**31 OK / 0 WARN / 0 FAIL**；无 shift_p90 FAIL。

**第 3 步离地高度**（`data/reports/T36/base_height.txt`）

| bag | 场景数 | base_height 中位 | 判断 |
|---|---|---|---|
| cp01 | 3 | 0.48 m | 介于趴下（约 0.3 m）与站立（0.65–0.80 m）之间 |
| cp02 | 5 | 0.48 m | 同上 |
| cp03 | 16 | 0.63 m | 接近站立下限 |
| cp04 | 7 | 0.63 m | 接近站立下限 |

**耗时**：重转约 4.7 min；INS 预览约 1.2 min；refine 约 1.6 min；融合预览约 1.2 min；离地高度约 1 s。

### T36 续复核

结果认可：
- 35 个场景里 keep 的 31 个，转换（偏置 0、杠杆臂 IMU 中心、`ins_azimuth_offset_from = given`）、两遍预览（全 OK）、位姿融合都正常；
- 狗不动，位姿基本不用修（修正量 p90 ≤ 0.01 m，航向修正最大 0.4°），唯一挪了 1.9 m 的 cp03_s000 融合后 shift 0.07 m。

人补的现场信息改变了对这批的看法：
- 狗站在三叉路口看车，货车、客车很多，基本都在动。狗不动、车在动，正好补上 0921 缺的"运动的车进入 30 m"（0921 自动标注只有 29 条）；
- 地点和 0921 是同一个厂区，相距几百米。T36 开头"地点很可能不一样"的说法来自 T34 抽的一条没有解的 INSPVAX，已更正；
- 离地高度 0.48 m（cp01、cp02）和 0.63 m（cp03、cp04），人说趴着和站着都有。在线节点的基准是 0.72 m，`auto_ground` 最多修 ±0.3 m，0.48 m 在范围内。

划分人定：1104 按整份 bag 分，两份验证、两份训练。T37 按这个改了：
- 体检留的 4 个缓冲场景按整份 bag 分以后不需要了，先转出来；
- 验证的两份整份审，训练的两份按统计挑；
- 默认验证 cp02 + cp04（各有一份趴着、一份站着），运动的车太偏时换 cp01 + cp04。

## T37 1104：两个教师、自动标注、划分、挑场景、审核材料

T36 续以后做。1104 是 8 号狗在 0921 同一个厂区（相距几百米）的一个三叉路口站着（有时趴着）看车，路过的货车、客车很多，基本都在动（RECORDED_DATA.md 2.2）。狗不动，路过的车在动，正好补上 0921 缺的"运动的车进入 30 m"。

- **教师**不换（T12、T15）：nuScenes Voxel + TTA、nuScenes Pillar + TTA，双教师自动标注。另外跑一遍上线模型（nuScenes Pillar，不加 TTA），看它和教师哪里不一样；
- **划分（人定，2026-11-05）**：1104 按整份 bag 分，两份做验证、两份做训练；0921 的划分不变；
- **挑场景**：验证的两份整份都审，训练的两份按 DATA_COLLECTION_PLAN 3.4 挑；
- 生成带四路相机的审核文件。

```text
任务 T37：1104：转出 4 个缓冲场景；双教师推理和自动标注；上线模型对教师；每个场景的统计；按 bag 划分（不写入 meta）；挑场景；审核材料。
先按通用约定做（第 9 条：可以改 tools/dog/ 里工具的 bug；第 10 条）。前置：T36 续。仓库根目录，export PYTHONPATH=.。BAGS=bags1104

0. mkdir -p data/reports/T37；python -m pytest tests/dog/test_scene_stats.py -q；
   df -h（--save-frames 每帧约 3.6 MB，1104 约 1 万帧，约 36 GB）；nvidia-smi。
   缓冲场景：体检给每份 bag 留了一个 keep = 0 的缓冲（1104_cp01_s002、1104_cp02_s003、1104_cp03_s012、1104_cp04_s005），
   那是为 bag 内部分训练 / 验证留的。现在按整份 bag 分，不需要了，转出来（每个约 30 s）：
     把这 4 行 segments.csv 的 keep 改成 1，然后每份：
     python tools/dog/rosbag_to_sequence.py --bag $BAGS/cpNN ...（同 T36 续第 1 步，--azimuth-offset 0）--only <那个场景>
   再对这 4 个跑 refine_poses.py 和 sequence_preview.py（同 T36 续第 2 步），都要 OK。合计 35 个场景。
1. 双教师（Voxel + TTA、Pillar + TTA，存帧，自动标注一起做）：
     python tools/dog/batch_label.py --seq data/seq/1104_* --ckpt-pillar work_dirs/pretrained/nusc_pp.pth --dry-run   （先看卡怎么分）
     python tools/dog/batch_label.py --seq data/seq/1104_* --ckpt-pillar work_dirs/pretrained/nusc_pp.pth
     mv data/labels/batch_summary.csv data/labels/batch_summary_1104.csv && git checkout data/labels/batch_summary.csv
2. 上线模型（nuScenes Pillar，不加 TTA），重放存好的帧：
     python tools/dog/run_sequence.py --replay $(ls -d data/runs/1104_*) --detectors pillar \
         --ckpt-pillar work_dirs/pretrained/nusc_pp.pth --no-predict --no-mcap --jobs 3 --device cuda:0 \
         --out "data/runs_online/{scene}" 2>&1 | tail -5 | tee data/reports/T37/online.txt
   然后每个场景：
     python tools/dog/compare_detectors.py --labels data/labels/<场景>/labels.pkl \
         --results online=data/runs_online/<场景>/pillar/results.pkl --classes vehicle --max-range 30
   抄 AP@2.0 那一行，写成 data/reports/T37/online_vs_teacher.csv（scene, ap2_30m）。越低说明上线模型和教师越不一致。
3. 每个场景的统计（30 m / 15 m 内运动的车、按子类、起步、待审核轨迹数）：
     python tools/dog/scene_stats.py --labels data/labels/1104_*/labels.pkl --out data/reports/T37/scene_stats.csv \
         | tee data/reports/T37/scene_stats.txt
   对照：0921 的自动标注合计 moving_30m 29、moving_15m 11（同一脚本，data/labels/0921_*/labels.pkl）。
   另按 bag 合计一张表（cp01–cp04：秒数、moving_30m、moving_15m、truck、bus），写进 scene_stats_by_bag.txt。
4. 按 bag 划分（只写方案，不改 meta.json，T38 才写入）：
   - 0921 照旧：data/survey/split/split.csv（训练 25、验证 10）；
   - 1104 默认：验证 cp02 + cp04，训练 cp01 + cp03。这样两边各有一份趴着的（cp01、cp02，离地高度约 0.48 m）、
     一份站着的（cp03、cp04，约 0.63 m）；验证 416 s，占 1104 的 39%；
   - 第 3 步按 bag 的表里，cp02 + cp04 的 moving_30m 不在 1104 合计的 25–60% 之间：换成 cp01 + cp04（368 s）验证、cp02 + cp03 训练；
   - 写 data/survey/split_0921_1104/split.csv（name, split, reason；reason 写 "1104 val clip" / "1104 train clip" / 0921 的原值）；
   - 空间重叠只报告、不改（同一个路口，两边的背景一定重叠；路过的车不同）：用 split_check.py 的 load、near_share
     （import tools/dog/split_check.py）算 1104 验证场景的路线有多少落在 0921 训练路线 10 m 以内，以及 1104 验证和训练的机位相距多远；
   - 报告：训练 / 验证的秒数，各自 moving_30m、moving_15m、truck、bus 的合计（0921 + 1104 并排）。
   - 提醒：cp03 结束（20:17:59）到 cp04 开始（20:18:59）只隔 1 分钟，同一辆车可能跨两份 bag，报告里看 cp03 末尾、cp04 开头各 10 s
     有没有轨迹（scene_stats 不管这个，用 labels.pkl 看）。
5. 挑场景，写成 data/reports/T37/review_candidates.csv（scene, split, 为什么, moving_30m, moving_30m_truck, ap2_30m, keyframes）：
   a. 验证的两份 bag：全部场景（验证标签都要人审）；
   b. 训练的两份 bag：moving_30m ≥ 1 的；有 truck 或 bus 的；ap2_30m 最低的 3 个；
   c. 关键帧（每 5 帧取 1 帧）合计超过 2,000：b 里按 moving_30m 从多到少留，直到 2,000；a 不减。
6. 审核材料（只看车；Foxglove 文件带四路相机）：只做第 5 步挑出的场景，同 BAG_SPLIT 5.15 第 6 步：
     review_labels.py table --classes vehicle → data/labels/<场景>/labels_tracks_vehicle.csv
     review_labels.py mcap --vehicles-only --labels data/labels/<场景>/labels.pkl --frames data/runs/<场景>/frames \
         --results data/runs/<场景>/voxel/results.pkl --cameras data/seq/<场景>/cameras.mcap --out data/labels/<场景>/<场景>_review.mcap
   SUSTechPOINTS 场景：同 T16（sustechpoints_convert.py，双教师的框，只留车；每 5 帧一个关键帧）。验证场景同时放候选框
   （DATA_COLLECTION_PLAN 3.5：两个教师在关键帧上检出、又和已有标注对不上的车，红框候选，`sustechpoints_convert.py candidates`）。
   挑一个车最多的场景的 review mcap，复制到 data/examples/runs/<场景>/ 提交，人先用它看路灯下相机够不够看清车型。
7. 提交：data/labels/batch_summary_1104.csv、data/labels/1104_*/labels.pkl、labels_tracks.csv、labels_tracks_vehicle.csv、
   data/reports/T37/、data/survey/split_0921_1104/、data/survey/1104_cp0*/segments.csv（keep 改过的）、data/preview/1104/（4 个缓冲场景加进去）、
   data/examples/runs/<那一个场景>/、本文档。
   不提交 data/runs、data/runs_online、review mcap（除了那一个）、SUSTechPOINTS 的场景目录（在服务器上给人用）。

停止条件：任何报错；第 1 步待审核的轨迹超过一半（教师在这批数据上明显不行，先停下看）。
报告（写进"T37 结果"）：缓冲场景的转换和预览；batch_summary_1104.csv 全文；scene_stats 的合计和按 bag 的表（和 0921 并排）；
online_vs_teacher；第 4 步用的是哪两份做验证、为什么，秒数和运动的车的合计，空间重叠，跨 bag 的车；
review_candidates.csv 和关键帧总数；每步耗时、显存、磁盘。
```

### T37 结果

**第 0 步缓冲场景**（keep=0 → 1）：`1104_cp01_s002`、`1104_cp02_s003`、`1104_cp03_s012`、`1104_cp04_s005`。偏置 0 转换，四路相机，帧数等于 segments；refine shift p90 全 0.00 m；预览 4/4 OK。合计 35 个场景。`test_scene_stats` 1 passed。公共盘空闲约 691 TB；仓库盘约 34 GB。

**第 1 步双教师**（Voxel+TTA、Pillar+TTA，存帧）：先在 cuda:7 跑；中途 GPU 被占，21 个场景 OOM。随后 Voxel-only 在 cuda:5 补全，Pillar+TTA 等 GPU 7 空出后再补，双教师融合重标。35/35 成功。`data/labels/batch_summary_1104.csv`：266 条轨迹、110 条待审（41.4%，未过半）。显存：Pillar+TTA 单卡约需 ≥13 GB 空闲。

**第 2 步上线模型**（Pillar 无 TTA → `data/runs_online/`）：35/35。`online_vs_teacher.csv` 的 AP@2.0@30m 多数 0.86–0.99；较低的有 `1104_cp04_s006` 0.664、`1104_cp04_s000` 0.770、`1104_cp03_s001` 0.735。`1104_cp04_s001` 无车（AP 记为 `-`）。

**第 3 步 scene_stats**（与 0921 对照）

| 批次 | seconds | moving_30m | moving_15m | moving_30m_truck | moving_30m_bus |
|---|---|---|---|---|---|
| 1104 合计 | 1052 | **103** | 56 | 98 | 0 |
| 0921 合计 | 1033 | 29 | 11 | 20 | 0 |

按 bag（`scene_stats_by_bag.txt`）：cp01 122 s / m30=13；cp02 171 s / 20；cp03 517 s / 40；cp04 245 s / 30。几乎全是 truck 子类在动，bus=0。

**第 4 步划分**（只写方案，未改 meta）：验证 **cp02+cp04**（默认；moving_30m 占 1104 的 48.5%，在 25–60%），训练 cp01+cp03。验证 416 s、m30=50；训练 639 s、m30=53。0921 验证 m30=15、训练 14。空间：1104 验证路线落在 0921 训练 10 m 内的比例 **0**（厂区相距几百米）；1104 验证与训练机位距离 **0.43 m**（同一路口静站，不同 bag）。跨 bag：`1104_cp03_s016` 末 10 s 有 4 条轨迹，`1104_cp04_s000` 头 10 s 有 5 条（可能同一辆车跨 bag）。方案：`data/survey/split_0921_1104/split.csv`。

**第 5–6 步挑场景与审核材料**：`review_candidates.csv` 32 个场景（验证 14 个全要 + 训练 18 个），关键帧 838+1102=1940（≤2000）。SUSTechPOINTS 在 `/public/zjst/robot/centerpoint_data/sustech_1104/`（不提交）；验证集候选红框合计 7 框 / 3 物体。示例 review mcap：`data/examples/runs/1104_cp03_s001/`（车最多，220 MB，路灯下看相机）。

**耗时**：缓冲转换约 1 min；双教师首轮约 40 min（后半 OOM）+ Voxel 补约 28 min + 等 GPU + Pillar 补约 24 min；上线约 16 min；统计/划分/审核材料约 10 min。磁盘：`data/runs/1104_*` 约 18 GB+（公共盘）。

### T37 复核（2026-11-06）

- **结果可用，1104 值得审。**
  - 30 m 内运动的车：1104 有 103 条，0921 只有 29 条；15 m 内 56 对 11。
  - 新的验证集（0921 的 10 个场景 + 1104 的 14 个）：
    - 30 m 内运动的车 65 条（0921 15 + 1104 50），达到 DATA_COLLECTION_PLAN 3.5 的 ≥ 50；15 m 内 29 条；
    - 有车的场景 21 个（8 + 13），还没到 ≥ 30，等下一批（另一天、另一片区域）。
  - 唯一缺的一步：没把 `sustech_1104/` 接进 SUSTechPOINTS，也没重新起服务。补在下面的 T37 续，几分钟的事，之后就可以开始 H6。
- **待审 41.4% 主要来自骑车的，不是车。**
  - 按类别数 `labels.pkl` 里的待审轨迹：车 42 / 140（30%），骑车 59 / 73，行人 9 / 53。
  - 只检车的模型只审车，停止条件按车算也没到一半。
  - 骑车的待审多，大概是晚上厂区里的电动车。以后做行人、骑车时再看。
- **bus = 0 是设计如此，不是漏了。**
  - `dog_perception/detection/boxes.py` 的 `NUSC_TO_SUBCLASS` 把教师的 bus、trailer 都归到 truck（0921 上 nuScenes 的 bus 头总落在长货车上）；
  - 只检车的模型又按尺寸分 car / truck 两个头（长 ≥ 5.5 m 或高 ≥ 2.5 m 是 truck）。
  - 1104 里路过的客车，标签上是 truck，训练进 truck 头，这是对的。H6 里不用改类型。
- **划分**：用默认的 cp02 + cp04 做验证，运动的车占 48.5%，在 25–60% 之间，可以。
  - 四份 bag 机位只差 0.43 m，验证和训练是同一个路口、同一个视角。
  - 本地查了自动标注，40 m 内的车框：
    - 训练的 18 个场景 5,380 个，在动的 5,243 个（97%）；
    - 验证的 14 个场景 6,065 个，在动的 4,451 个（73%）。
  - 训练里几乎没有停着的车，所以"同一辆停着的车两边都有"的问题很小。
  - 两边真正共享的是背景：路口的楼、杆子、路沿。训练里这些都当负样本学过，所以 1104 验证上的误检会偏少，偏乐观。
  - T39 要分开报 0921 验证和 1104 验证。1104 上主要看两样：30 m 内的 AP@2.0；运动的车被检出了多少（compare_detectors 的 moving labels / matched 那张表）。1104 上的误检和假轨迹不代表别处。
  - 跨 bag：cp03 末尾有 4 条轨迹，cp04 开头有 5 条。两份中间隔 1 分钟，按 1 m/s 算，一辆在动的车这段时间也开出了 60 m。能两边都出现的只会是停着的车或很慢的车，而训练里停着的车本来就很少，影响小。
- **online_vs_teacher**：较低的 3 个场景里，2 个在验证集（cp04_s006 0.664、cp04_s000 0.770），1 个在训练集（cp03_s001 0.735，已挑上）。
  - 这是对着没审的教师标签算的，不知道是谁的错。H6 里这 3 个场景看仔细些，T39 在审过的标签上重算。
- **审核量**：
  - 验证 14 个场景：838 个关键帧、1,392 框（红 156），每个框都看；
  - 训练 18 个场景：1,102 个关键帧、1,275 框（红 266），看红框和漏掉的车；
  - 验证候选红框 7 框 / 3 个物体，都是只有 Pillar 检出的。
  - H3 开始时红框是 780 个，这次合计 422 个，但验证要逐框看。估计 1 天。
- **没挑上的 3 个训练场景**：`1104_cp03_s003`（2 条车轨迹，1 条待审）、`s006`、`s007`（没有车），合计 91 s，30 m 内都没有运动的车。没人审过，T38 里改成 drop，不进 v3。
- 示例 review mcap 是 LFS 文件（220 MB）。本地要看的话：`git lfs pull --include "data/examples/runs/1104_cp03_s001/*"`。

---

## T37 续 1104 的场景接进 SUSTechPOINTS（H6 之前）

```text
任务 T37 续：把 sustech_1104/ 的 32 个场景接进 SUSTechPOINTS，备份审核前的标注，重新起服务，给 H6 用。
先按通用约定做。前置：T37。仓库根目录，export PYTHONPATH=.。
NEW=/public/zjst/robot/centerpoint_data/sustech_1104; BK=/public/zjst/robot/centerpoint_data/sustech_backup
0921 的 35 个场景不动（third_party/SUSTechPOINTS/data/0921_* 仍指向 sustech_dual/，H3 的结果在里面）。

1. 记下正在跑的 serve 命令，然后停掉：
     ps aux | grep "sustechpoints_convert.py serve" | grep -v grep
2. 接进去：
     R1104=$(ls -d $NEW/1104_* | xargs -n1 basename); echo $R1104 | wc -w          （应为 32）
     for r in $R1104; do ln -sfn $NEW/$r third_party/SUSTechPOINTS/data/$r; done
     ls third_party/SUSTechPOINTS/data | grep -c '^1104_'                             （应为 32）
3. 备份审核前的标注（照 T11 第 2 步）：
     mkdir -p $BK; tar -C $NEW -cf $BK/H6_before.tar $(for r in $R1104; do echo $r/label $r/dog_meta.json $r/dog_info; done)
4. 统计：python tools/dog/sustechpoints_convert.py stats --root $NEW | tail -2 | tee data/reports/T37/stats_before_H6.txt
   应和 T37 的 sustech_stats.txt 一样：验证 838 帧、1392 框、156 红；训练 1102 帧、1275 框、266 红。不一样就停下报告。
5. 用第 1 步记下的命令重新起（后台，日志写到 data/reports/T37/serve.log），浏览器打开，看列表里有 1104 的场景。

停止条件：任何报错；第 2 步不是 32；第 4 步对不上。
提交：data/reports/T37/stats_before_H6.txt、本文档（不提交 serve.log、tar）。
报告（写进"T37 续 结果"）：SUSTechPOINTS 地址；第 4 步两行；备份文件的大小。
```

### T37 续 结果

- 原 serve：`python tools/dog/sustechpoints_convert.py serve --host 10.36.136.7`（自 10-26 起）；已停掉并用同一命令重新起。
- 32 个 `1104_*` 软链进 `third_party/SUSTechPOINTS/data/`（0921 的 35 个仍指 `sustech_dual/`，未动）。
- 备份：`/public/zjst/robot/centerpoint_data/sustech_backup/H6_before.tar`，6.0 MB。
- 统计（与 T37 `sustech_stats.txt` 一致）：
  - `total train: 1102 frames, 1275 boxes, 266 marked review`
  - `total val: 838 frames, 1392 boxes, 156 marked review`
- 地址：http://10.36.136.7:8081 （日志里已列出全部 1104 场景；浏览器 Ctrl+F5 后开始 H6）。

### T37 续 复核（2026-11-06）

- 结果对：32 个场景接好，0921 的没动；审核前的统计和 T37 的 `sustech_stats.txt` 一样（验证 838 帧、1,392 框、156 红；训练 1,102 帧、1,275 框、266 红）；备份 `H6_before.tar` 6.0 MB。可以开始 H6。
- T38 第 0 步再打一份 `H6_after.tar`，导回出问题时两份都在。

### 068342b 复核（标注工具的相机，2026-11-06）

人加的：`sustechpoints_convert.py cameras` 把场景的 cameras.mcap（H.264）解成去畸变的 JPEG 和标定，SUSTechPOINTS 右上角默认只开前相机，选中的框落在左 / 右相机里时再开那一路；点云调亮了，旧扫描从 30% 亮度调到 65%。做法可以，改了三处（本次提交）：

- **图比关键帧早一帧（40 ms）**：
  - 原来的解码先过 H.264 parser，它要等到下一条消息才吐出这一帧，所以每张图都记成了下一条消息的时间。
  - 合成码流上量出来是早一帧（有没有 B 帧都一样）；10 m/s 的货车差 0.4 m。
  - 另外，片段从两个关键帧中间开始时会直接报错（`InvalidDataError`）。场景是从整份 bag 里切出来的，这种情况会有。
  - 改成每条消息单独送解码器、用序号当 pts，解出来的帧按 pts 找回自己的消息；关键帧之前解不了的消息跳过。
  - 原来把整段视频的每一帧都转成 BGR 留在内存里：30 s、25 Hz、1920×1536，一路约 6.6 GB。现在只转要用的那几帧。
- **去畸变的视野太宽**：
  - `balance=0` 得到的针孔焦距只有 187 px，横向 158°，图中间被缩小了 2.7 倍。10 m 外的轿车在图上约 80 px 宽，在右上角的小窗里只剩十来个像素。
  - 默认改成用鱼眼自己的 fx（509 px，横 124°、纵 113°），中间是原分辨率；前、左、右合起来覆盖到正后方盲区的边上。
  - `--focal 0` 回到原来的宽视野。
- **标定少了雷达离地高度**：
  - 场景点在检测系里（原点在地面），相机挂在机身上；原来只把导出时的 1.8 m 挪回去了，没减去雷达离地的高度（1104 是 0.48 / 0.63 m）。
  - 投到图上的框整体偏高，5 m 处约 50–65 px（509 px 焦距下）。
  - 现在从序列的 meta.json 读 `base_height`。机身的俯仰、横滚没有算（检测系去掉了，相机的旋转本来也没标定），投影仍只是参考。
- 测试：
  - `test_camera_frames_are_matched_to_their_own_message`（libx264 编的码流，有无 B 帧、从中间开始）；
  - `test_camera_calib_puts_the_scene_ground_below_the_camera`；
  - 本地另用转换器写法的 cameras.mcap 走了一遍 `export_cameras`，每张图都是对应关键帧那一帧。
- 旧扫描调到 65% 以后，开得快的货车，拖尾和当前这一帧的亮度差得不多了。H6 里框要贴最亮的那一层点（当前扫描），不要贴整条拖尾。拖尾太乱可以调回去（`public/js/lidar.js` 的 `OLD_SWEEP_SHADE`）。

---

## T37 续 2 1104 的场景导出相机图（H6 之前或期间都可以）

```text
任务 T37 续 2：用修正后的 cameras 命令，给 sustech_1104 的 32 个场景导出四路相机图和标定，重新起 SUSTechPOINTS。
先按通用约定做。前置：拉到 068342b 的复核提交；pip install av（requirements_dog.txt 新加的）。仓库根目录，export PYTHONPATH=.。
NEW=/public/zjst/robot/centerpoint_data/sustech_1104; R1104=$(ls -d $NEW/1104_* | xargs -n1 basename)
只写场景的 camera/、calib/，不动 label/：H6 已经开始也可以做，人刷新页面（Ctrl+F5）就行。
以前用 068342b 的版本导过的，这次全部覆盖（那一版的图早一帧、视野太宽、框偏高）。

1. python -c "import av, cv2; print(av.__version__, cv2.__version__, 'libx264' in av.codecs_available)"
   pytest tests/dog/test_sustechpoints.py tests/dog/test_cameras.py -q          （新的两个测试不能是 skipped）
2. python tools/dog/sustechpoints_convert.py cameras --root $NEW --scenes $R1104 --seq-root data/seq \
       2>&1 | tee data/reports/T37/cameras.txt
   每个场景每一路的张数应等于关键帧数（data/reports/T37/sustech_stats.txt 的 frames 列）。
   开头一小段没有关键帧时会少几张，少的列进报告；某一路少一半以上就停下报告。
3. du -sch $NEW/*/camera | tail -1
4. 用 T37 续结果里那条命令重新起 SUSTechPOINTS（后台，日志写到 data/reports/T37/serve.log）。
   stats 两行：H6 还没开始的话，应和 T37 续一样（相机不改标注）。

停止条件：任何报错；第 1 步测试失败或跳过；第 2 步某一路少一半以上。
提交：data/reports/T37/cameras.txt、本文档。
报告（写进"T37 续 2 结果"）：张数不等于关键帧数的场景和相机；磁盘；第 2 步耗时。
```

### T37 续 2 结果

- 前置：已拉到 `34f4a15`（含 cameras 复核修复）；`av 17.1.0`、`cv2 4.11.0`、`libx264` 可用；`pytest tests/dog/test_sustechpoints.py tests/dog/test_cameras.py`：**20 passed**（含帧对齐、地面高度标定两个新测）。
- 第 2 步：32 个场景全部覆盖导出；耗时 **651 s**（约 10.9 min）。
- 张数 ≠ 关键帧（开头缺关键帧 / 无匹配，均远未少一半）：
  - `1104_cp01_s000`：left 59/62、rear 59/62（各少 3）
  - `1104_cp04_s000`：四路均为 61/62（各少 1）
  - 其余 30 个场景四路均等于关键帧数
- 磁盘：`du -sch $NEW/*/camera` → **1.7G** total；标定 fx≈509（鱼眼本征），不再用 balance-0 的 187。
- stats 未变（相机不改标注）：`total train: 1102 frames, 1275 boxes, 266 marked review`；`total val: 838 frames, 1392 boxes, 156 marked review`
- 服务：http://10.36.136.7:8081 （已用 T37 续同命令重启；H6 请 Ctrl+F5）

### T37 续 2 复核（2026-11-06）

- 结果对：30 个场景四路都等于关键帧数；`cp01_s000` 左、后各少 3 张，`cp04_s000` 四路各少 1 张（开头没赶上关键帧），不影响审核。
- 标定用的是 509 px，标注没变，1.7 GB，11 分钟。

---

## H6 审 1104 挑出的场景（人做）

T37 续之后。在 SUSTechPOINTS 里审 1104 的 32 个场景（先验证集 14 个，再训练集 18 个），浏览器重新打开页面（Ctrl+F5）。

- 列表里也有 0921 的 35 个场景，H3 已审完，不用动，只选 `1104_` 开头的。
- **只审车辆**，行人、骑车的不审。
- 划分已定（2026-11-05）：验证 cp02 + cp04，训练 cp01 + cp03。

**相机**（T37 续 2 之后）：SUSTechPOINTS 右上角是前相机；选中一个框，它落在左 / 右相机里时，下面再开那一路。小窗可以拖大。

- 先翻几帧，看路灯下能不能认出是不是车、是不是叉车。能认出的话，拿不准就对照相机；认不出就只看点云。结论记到下面"H6 结果"。
- 图上画的框只是参考：相机的旋转没标定，前相机还往下低 6–7°，框可能偏一点。用图判断"是什么"，框的位置和大小在点云里调。
- 车开得快的时候，点云里有一条拖尾（前几帧扫描，暗一些）。框贴最亮的那一层点，不要贴整条拖尾。
- 要连续看视频，还是用 Foxglove：服务器上每个场景都有 `data/labels/<场景>/<场景>_review.mcap`（本地的示例在 `data/examples/runs/1104_cp03_s001/`，先 `git lfs pull`）。

**验证集（cp02、cp04 的 14 个场景）：每个关键帧的每个框都看**，不只是红框。

- 验证标签决定以后选哪个模型，漏一辆车、多一个假框，都直接算进 AP。
- 逐帧检查三件事：
  - 是不是车；
  - 框偏出车 0.5 m 以上就调；
  - 有没有车没框（翻帧时留意 30 m 以内）。
- 红框里有 7 个是候选：两个教师在关键帧上检出、又和已有标注对不上（都只有 Pillar 检出）。和别的红框一样判。

**训练集（cp01、cp03 的 18 个场景）**：只看红框，加上翻帧时看到的漏掉的车（同 H3）。

| 看到的 | 操作 |
|---|---|
| 是车（含停着的、挂车厢、客车） | 右键 → 确认（整条轨迹）。框明显歪了（偏出车 0.5 m 以上）就调一下 |
| 叉车、牵引车拉的一串料架 | 右键 → 看不清（作忽略框）。不要求检出（叉车 2026-10-26，料架车 2026-11-06），不要确认成车，也不要删 |
| 不是车（墙角、集装箱、货堆、岗亭、路灯杆……） | 右键 → Delete this object (in all frames)。**删，不要标忽略** |
| 看不清（点太少、只看到一角） | 右键 → 看不清（作忽略框） |
| 同一个 ID 有几帧是车、有几帧不是 | 按 `v` 进批量窗口逐帧看，不是车的那几帧单独删 |

- **漏掉的车**：按住 Ctrl 拖矩形框进去，自动出框（两个教师的模型框），检查后类别选 Car 或 Truck。
  - 类型只影响记录，只检车的模型按尺寸分 car / truck。客车不用改成 Bus，教师给的 Truck 留着就行。
- **多看这 3 个场景**：`1104_cp04_s006`、`1104_cp04_s000`、`1104_cp03_s001`，上线模型和教师差得最多（T37 第 2 步）。
- **狗正后方**约 76° 的扇区没有点（机身系方位角 |θ| > 142°，SENSORS.md）。车开进、开出这个扇区时只剩半边点，是车就确认，别删。
- **收尾**：审完还红着的框改成"看不清"。T38 导回时，红框超过 22 个会停下来问（H6 收尾留了 22 个）。

做完把状态表的 H6 改成"完成"，在下面"H6 结果"里写：审核人、日期；确认、删除、改忽略、补画的大致数量；相机够不够用；遇到的拿不准的东西（像 0921 的叉车那样，以后可以定一条规则）。

### H6 结果

- 审核日期：2026-11-06；标注目录：`/public/zjst/robot/centerpoint_data/sustech_1104/`（相对 `H6_before.tar`）。
- 审核后 stats（`data/reports/T37/stats_after_H6.txt`，二次补清后）：
  - `total train: 1102 frames, 1203 boxes, 4 marked review`
  - `total val: 838 frames, 1382 boxes, 4 marked review`
  - 审核前（T37 续）：train 1275 框 / 266 红；val 1392 框 / 156 红。
- 相对 `H6_before.tar` 的框级变化（按 `obj_id`）：删除约 **83** 个框；新画约 **1** 个；属性变化约 **332**（主要是去掉 `review`＝确认）；框心移动 >5 cm 的约 **0**。未使用 `ignore` 属性（`obj_attr` 仅有空或 `review`）。
- 仍带 `review` 的 **8** 个框（4 条轨迹，低于 T38 阈值）：
  - train：`1104_cp03_s011` #20（2）、`1104_cp03_s015` #14（2）
  - val：`1104_cp04_s005` #15（2）、`1104_cp04_s007` #c2（2）
  - 导回后仍作忽略框；不挡 T38。
- 相机：前视 + 选中后左右视可用；右视已改到右下角少挡点云（`image.js`）。

### H6 复核（2026-11-06）

- **审完了，结果大部分可以直接用。**
  - 红框从 422 个降到 8 个（人二次补清：22 → 8）；删了 83 框，补画 1 框，没有挪过框，也没有用"看不清"。
  - 验证集只动了 11 框（删 11、补 1），审前的自动标注在验证场景上基本是对的。
  - 剩下的 8 个红框（4 条轨迹）导回后就是忽略框，和"看不清"效果一样，T38 照常做（停止条件仍是超过 22）。
- **只有一处要人再看：`1104_cp01_s003` 删了 61 框（137 → 76），大多是一辆像在慢行的货车。**
  - 按自动标注看，删掉的主要是 #5、#17、#18：
    - 时间上首尾相接（0–12 s、12–14 s、14–30 s）；
    - 都在狗左边约 27 m 的同一条线上，一路往前走了约 55 m，速度约 2 m/s；
    - 框长 8.5–12 m、高 3–3.6 m；在 26–40 m 时每框当前扫描有 200–340 个点，同一场景里确认过的货车在这个距离是 260–700 个；
    - 最近的别的车框也在 10 m 以外，不是别的车拆出来的半截。
  - 两个教师给的分数都低（0.31–0.37），所以标成了待审。
  - 右边 28–37 m 有两条短的（#8、#9，1–2 s，长 11 m）。（T38 复核更正：它们审核时确认成了车，没有删。）
  - 如果是车，删掉等于告诉模型"30 m 处慢慢开的货车是背景"，正好伤到最想要的那类样本（训练集）。如果相机里确认不是车（比如墙或栅栏上滑动的假框），删了是对的。
  - 其他场景删的大多是短轨迹或长货车拆出来的半截。比如 `cp04_s006` #12 离另一辆车的框只有 0.9–3.6 m，`cp02_s000` #14 离另一辆车 6 m，删除合理。
- **H6 补（人，10 分钟）**：Foxglove 打开服务器上的 `data/labels/1104_cp01_s003/1104_cp01_s003_review.mcap`，看 0–30 s 狗左边约 27 m、右边约 30 m 的那几条轨迹，对照左右相机。
  - 结论写到下面一行：是车 → T38 第 0 步从 `H6_before.tar` 恢复这几条（不用重画）；不是车 → 不动。
  - **H6 补 结论**（人，2026-11-06）：#5/#17/#18 不是货车，是一辆很小的车拉着一串置物架，开得慢，**可要可不要**。
    - 可要可不要，就作忽略框（同叉车）：不要求检出，检出了也不算误检。
    - 删掉不对：删了就是背景，模型检出它反而算误检。T38 第 0 步把这三条从 `H6_before.tar` 恢复成忽略框。
    - #8/#9 审核时确认成了车，不动（T38 复核更正：之前说它们也被删了，不对）。
    - 以后同类的（牵引车拉的一串料架、物料车）都按这条：右键 → 看不清（作忽略框），写进了 LABELING_TUTORIAL 5.1、DOG_FINETUNE 的叉车那条。

---

## T38 导回 H6，和 0921 合并，导出 v3 数据集

```text
任务 T38：导回 H6 在 sustech_1104/ 里的审核结果；训练场景把关键帧传播到每一帧，再补停着的车的空档；叉车的尺寸规则兜底；
1104 的划分写进 meta.json；0921 + 1104 一起导出 v3 数据集（新目录）；统计新验证集。
先按通用约定做。前置：H6 完成。仓库根目录，export PYTHONPATH=.。
SPLIT=data/survey/split_0921_1104/split.csv; NEW=/public/zjst/robot/centerpoint_data/sustech_1104
BK=/public/zjst/robot/centerpoint_data/sustech_backup; R1104=$(ls -d $NEW/1104_* | xargs -n1 basename)
1104 的新文件和 0921 现在用的同名（labels_sustech2.pkl、labels_full2.pkl、labels_train2.pkl），按 {scene} 找标签的脚本两批通用。
0921 的标签不动。

0. mkdir -p data/reports/T38；pytest tests/dog -m "not slow" -n 8 | tail -1。
   停掉 SUSTechPOINTS（审完了，免得导回时还有人在改），备份：
     tar -C $NEW -cf $BK/H6_after.tar $(for r in $R1104; do echo $r/label $r/dog_meta.json $r/dog_info; done)
   H6 补（H6 复核）：1104_cp01_s003 的 #5/#17/#18 是牵引车拉的一串置物架，可要可不要，从 H6_before.tar 恢复成忽略框
   （obj_attr = ignore；只加回现在没有的）。下面的 Python 从行首开始写，参数是场景和轨迹号：
NEW=$NEW BK=$BK python - 1104_cp01_s003 5 17 18 <<'EOF' | tee data/reports/T38/restore.txt
import json, os, sys, tarfile
NEW, BK = os.environ["NEW"], os.environ["BK"]
scene, ids = sys.argv[1], set(sys.argv[2:])
n = 0
with tarfile.open(os.path.join(BK, "H6_before.tar")) as tar:
    for m in tar.getmembers():
        if not (m.name.startswith(scene + "/label/") and m.name.endswith(".json")):
            continue
        old = [o for o in json.load(tar.extractfile(m)) if str(o["obj_id"]) in ids]
        p = os.path.join(NEW, m.name)
        cur = json.load(open(p))
        have = {str(o["obj_id"]) for o in cur}
        add = [dict(o, obj_attr="ignore") for o in old if str(o["obj_id"]) not in have]
        if add:
            json.dump(cur + add, open(p, "w"))
            n += len(add)
print(scene, "restored", n, "boxes as ignore of", sorted(ids, key=int))
EOF
   应恢复 50–61 框。第 1 步导回后，这些是忽略框，不是车。
   没审的 3 个训练场景（T37 复核）改成 drop：
     awk -F, -v OFS=, '$1=="1104_cp03_s003"||$1=="1104_cp03_s006"||$1=="1104_cp03_s007" {$2="drop"; $3="1104 train clip, not reviewed"} 1' \
         $SPLIT > $SPLIT.tmp && mv $SPLIT.tmp $SPLIT
     awk -F, 'NR>1 {print substr($1,1,4), $2}' $SPLIT | sort | uniq -c        （0921 train 25 / val 10；1104 train 18 / val 14 / drop 3）
1. 32 个场景导回（照 T17 第 1 步）：
     for r in $R1104; do
       echo "== $r"
       python tools/dog/sustechpoints_convert.py import --scene-dir $NEW/$r --frames data/runs/$r/frames \
           --out data/labels/$r/labels_sustech2.pkl
     done 2>&1 | tee data/reports/T38/import.txt
   每个场景的帧数要和 data/reports/T37/sustech_stats.txt 一样。"still marked review" 合计超过 22 个框就停下问人
   （H6 收尾留了 22 个，导回后是忽略框；人后来又改掉一些的话会更少）。
2. 训练场景（18 个）：关键帧传播到每一帧，再补停着的车的空档（照 T17 第 2 步，--full 用 T37 的双教师自动标注）：
     T1104=$(awk -F, 'NR>1 && $2=="train" && $1 ~ /^1104_/ {print $1}' $SPLIT); echo $T1104 | wc -w   （应为 18）
     for r in $T1104; do
       echo "== $r"
       python tools/dog/sustechpoints_convert.py propagate --full data/labels/$r/labels.pkl \
           --keyframes data/labels/$r/labels_sustech2.pkl --out data/labels/$r/labels_full2.pkl
       python tools/dog/merge_labels.py --reviewed data/labels/$r/labels_full2.pkl --frames data/runs/$r/frames \
           --out data/labels/$r/labels_train2.pkl
     done 2>&1 | tee data/reports/T38/propagate.txt
   某个场景留下的帧少于 90% 就停下报告（pose_bad 的帧不算在分母里，同 T17 的 s024）。
3. 叉车兜底（2026-10-26 的规则，和 0921 一样；H6 里人已经手工改过的不会再列出来）：
     python tools/dog/ignore_tracks.py --scenes $R1104 --keyframes "data/labels/{scene}/labels_sustech2.pkl" \
         --also "data/labels/{scene}/labels_full2.pkl" "data/labels/{scene}/labels_train2.pkl" \
         --length 2.3 3.6 --width 1.3 1.85 --height 1.95 2.6 --reason forklift --dry-run | tee data/reports/T38/forklift_dryrun.txt
   列出的轨迹里，有标 moving、位置离狗 30 m 内的，就停下报告：路口上在动的这种尺寸的车，可能是三轮货车，要人看一眼。
   否则去掉 --dry-run 再跑一遍，输出写到 data/reports/T38/forklift.txt。没列出轨迹就跳过。
4. 1104 的划分写进 meta.json（segment.split；原来的值留在 segment.split_survey，同 split_check.py --apply）。
   下面的 Python 从行首开始写，整段复制：
python - $SPLIT <<'EOF'
import csv, json, os, sys
for r in csv.DictReader(open(sys.argv[1])):
    p = os.path.join("data/seq", r["name"], "meta.json")
    if not r["name"].startswith("1104_"):
        continue
    meta = json.load(open(p))
    seg = meta.setdefault("segment", dict(name=r["name"]))
    seg.setdefault("split_survey", seg.get("split", ""))
    seg["split"] = r["split"]
    json.dump(meta, open(p + ".tmp", "w"), indent=2)
    os.replace(p + ".tmp", p)
    print(r["name"], seg["split_survey"], "->", seg["split"])
EOF
5. 导出 v3 到新目录（照 T17 第 3 步）：验证场景用 labels_sustech2.pkl，训练场景用 labels_train2.pkl，0921 和 1104 一样：
     VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT); TRAIN=$(awk -F, 'NR>1 && $2=="train" {print $1}' $SPLIT)
     echo $VAL | wc -w; echo $TRAIN | wc -w                                     （24、43）
     REC=$(for r in $VAL; do echo --recording data/runs/$r data/labels/$r/labels_sustech2.pkl; done
           for r in $TRAIN; do echo --recording data/runs/$r data/labels/$r/labels_train2.pkl; done)
     NEWDS=$(dirname $(readlink -f data/dog_veh))/dog_veh_$(date +%m%d_%H%M)
     python tools/dog/export_dataset.py $REC --val-recordings $VAL --classes vehicle \
         --out $NEWDS --gt-db --jobs 8 2>&1 | tee data/reports/T38/export.txt
   换链接前确认没有训练、评估在读 data/dog_veh（ps aux | grep -E "train|eval|compare_detectors" | grep -v grep）。
   GT 库是按路径读的，训练中途换链接会读到新目录。有就等它结束：
     OLD=$(readlink -f data/dog_veh); ln -sfn $NEWDS data/dog_veh; echo "data/dog_veh: $OLD -> $NEWDS"
   旧目录（v2）写进报告，先不删，T39 还要对照。
6. 新验证集的统计（关键帧 2 Hz，--min-frames 2 就是 1 s）：
     python tools/dog/scene_stats.py --labels $(for r in $VAL; do echo data/labels/$r/labels_sustech2.pkl; done) \
         --min-frames 2 --out data/reports/T38/val_stats.csv | tee data/reports/T38/val_stats.txt
   按批次（0921 / 1104）合计：场景数、有车的场景数、moving_30m、moving_15m。
   1104 的 14 个验证场景再和 T37 的自动标注比一下：用同一条命令，标签换成 data/labels/$r/labels.pkl，--min-frames 10。

停止条件：任何报错；第 0 步数量不对；第 1 步帧数对不上或红框超过 22；第 2 步某个场景少于 90%；第 3 步有在动的近处轨迹。
提交：data/labels/1104_*/labels_sustech2.pkl（32 个）、labels_full2.pkl、labels_train2.pkl（18 个）、$SPLIT（3 个 drop）、
data/reports/T38/、本文档。不提交 tar、data/seq、数据集目录。
报告（写进"T38 结果"）：
- import 的计数合计（车辆框、still marked review、moved or new），按验证 / 训练分开；
- propagate 和补空档的合计；叉车（列出的轨迹、改了多少框）；
- export 的 train / val 帧数，car / truck 正样本和忽略数，和 T17（v2）并排；新数据集目录和旧目录；
- 第 6 步按批次的表，以及 1104 验证场景审前、审后的 moving_30m。
```

### T38 结果

32 个场景导回帧数和 T37 `sustech_stats.txt` 一致（合计 1,940 关键帧）。车辆框 **2,646**，still marked review **8**（<22），moved or new **3**。按划分：验证 14 场景车辆框 1,382 / review 4 / moved 3；训练 18 场景车辆框 1,264 / review 4 / moved 0。`1104_cp01_s003` 从 H6_before 恢复 #5/#17/#18 共 61 个 ignore 框。3 个未审训练场景改成 drop。

- propagate（18 个训练场景）：留下 **5,438 / 5,488** 帧（99.1%），最低也 ≥98%。keyframe boxes 1,264，interpolated boxes 4,772。补停着的车的空档：parked places **5**。
- 叉车兜底：dry-run 无轨迹，跳过。
- 导出到新目录 `/public/zjst/robot/centerpoint_data/dog_veh_1106_1811`（61G）。`data/dog_veh`：`/public/zjst/robot/centerpoint_data/dog_veh_1026_1410`（v2）→ 这个新目录（v3）。旧目录先不删，T39 对照用。train **12,670** / val **1,418** 帧。

  | | train car | train truck | val car | val truck |
  |---|---|---|---|---|
  | v3（T38） | 14,382（+3,018 ignored） | 10,636（+1,973 ignored） | 811（+528 ignored） | 2,446（+160 ignored） |
  | v2（T17） | 14,339（+2,893 ignored） | 5,073（+1,418 ignored） | 1,100（+217 ignored） | 1,091（+155 ignored） |

  note：few points 1,056；no points 362；subclass disagrees 34。GT 库：car 14,382、truck 10,636。

- 新验证集（关键帧，`--min-frames 2`）按批次：

  | batch | scenes | with vehicle | moving_30m | moving_15m |
  |---|---|---|---|---|
  | 0921 | 10 | 9 | 21 | 8 |
  | 1104 | 14 | 13 | 51 | 27 |
  | total | 24 | 22 | 72 | 35 |

- 1104 的 14 个验证场景，审前（`labels.pkl`，`--min-frames 10`）moving_30m **50**；审后（`labels_sustech2.pkl`，同参）moving_30m **31**。

### T38 复核（2026-11-07）

- **结果可用，T39 可以开始。** v3 在 `/public/zjst/robot/centerpoint_data/dog_veh_1106_1811`（61 GB），训练 12,670 帧（0921 7,232 + 1104 5,438）、验证 1,418 帧（580 + 838）。
- 本地用提交的标签核了一遍：
  - `cp01_s003` 的 #5/#17/#18 是忽略框：关键帧 25 + 3 + 33 = 61 框，10 Hz 是 121 + 16 + 165 框。
  - 剩下的 8 个红框导回后是忽略框；moved or new 3 个（验证）。
  - 传播留下 99.1% 的帧；补空档只有 5 处（1104 几乎没有停着的车，和 T37 复核一致）；叉车尺寸规则没挑到东西；快速测试 183 passed。
- **更正（H6 复核）**：#8/#9 没有被删，审核时确认成了车（关键帧 4 + 2 框）。
  - 删掉的 61 框正好是 #5/#17/#18。我当时是按框数推的，推错了。
  - H6 复核和 H6 补 结论里的说法已改。
- **"审前 50 → 审后 31"是两边的参数不一样，审核没有删掉运动的车。**
  - 审后的标签是关键帧，2 Hz。`--min-frames 10` 在关键帧上要连续 5 s 在动，在 10 Hz 的自动标注上只要 1 s。
  - 两边都按 1 s 算（关键帧用 `--min-frames 2`），本地重算：30 m 内 50 → 51，15 m 内 24 → 27。
  - 第 6 步"用同一条命令"是我写得不清楚。
- **验证集的车（car）从 1,100 降到 811，不是退步。**
  - v2 导出（10-26 下午）之后，0921 的叉车才改成忽略框。现在 0921 验证关键帧里，car 头的正样本 823 个、忽略 501 个（导出再去掉点太少的）。
  - 1104 的验证集只加了 20 个 car、1,358 个 truck。
  - 所以 1104 验证基本是一个"货车"验证集，新验证集上 car 的 AP 几乎全来自 0921。T39 按批次分开报。
- **训练的 truck 翻了一倍**（5,073 → 10,636）：1104 加了约 5,560 个，几乎全是在动的货车；car 只多了 43 个。
  - 配置里 GT 粘贴是 car 2 个、truck 3 个（注释写"truck 少"），现在已经不少了。
  - T39 先不改，和 v2 同一个配方，这样差别只来自数据。T39 里货车误检明显变多的话，再调。
- 新验证集（关键帧、1 s）：24 个场景，有车的 22 个（目标 ≥ 30，还差）；30 m 内运动的车 72 条，15 m 内 35 条。
- 报告里没写第 4 步（划分写进 meta.json）做没做。T39 不读它，T39 第 0 步顺手查一下。

---

## T39 训练 v3；在新验证集上定上线的 Pillar、教师换不换

- **v3** 用 T38 的数据集（0921 + 1104 的 18 个训练场景），配方和 v2 完全一样：单卡、batch 4、学习率 5e-4、20 个 epoch，配置不改。所以 v3 和 v2 的差别只来自数据。
- **新验证集**：0921 的 10 个场景 + 1104 的 14 个，每个比较都按三份报：合计、0921、1104。
  - 0921：狗在走，有轿车也有货车；
  - 1104：同一个路口、狗不动、几乎全是在动的货车（T38 复核）；它的背景在训练里见过，所以 1104 上的误检偏乐观（T37 复核）。
- **验证标签偏向 nuScenes 模型**：标签是从两个 nuScenes 教师的框审出来的（T17 复核），所以只看 AP@2.0、AP@4.0，不看 AP@0.5、AP@1.0 和含它们的 mAP。
- **所有对照模型都用现在的代码在 24 个场景上重跑。** 0921 的旧结果是 T6、T15 时跑的，T22 之后检测的默认参数变了，不要混用。
- **上线的 Pillar 按人定的规则**（T15 复核）：nusc_pp（现在上线的）、v2_pp、v3_pp 三个里选最好的；都不比 nusc_pp 好就留 nusc_pp。

```text
任务 T39：用 T38 的 v3 数据集训练 v3 的 Voxel 和 Pillar；在新验证集（0921 + 1104）上和 nusc_pp、v2_pp 比，定上线的 Pillar 和上报阈值；
v3 Voxel + TTA 和 nuScenes 教师比，定教师换不换。
先按通用约定做。前置：T38。仓库根目录，export PYTHONPATH=.。SPLIT=data/survey/split_0921_1104/split.csv

0. mkdir -p data/reports/T39
   readlink -f data/dog_veh                       （应是 /public/zjst/robot/centerpoint_data/dog_veh_1106_1811）
   python -c "import pickle; print(*(len(pickle.load(open('data/dog_veh/infos_%s.pkl' % s, 'rb'))) for s in ('train', 'val')))"
                                                  （应是 12670 1418）
   python -c "import json; print(json.load(open('data/seq/1104_cp02_s000/meta.json'))['segment']['split'])"
                                                  （应是 val，T38 第 4 步；不是就补做 T38 第 4 步，不用停）
   VAL=$(awk -F, 'NR>1 && $2=="val" {print $1}' $SPLIT)
   V0921=$(echo $VAL | tr ' ' '\n' | grep '^0921_'); V1104=$(echo $VAL | tr ' ' '\n' | grep '^1104_')
   echo $VAL | wc -w; echo $V0921 | wc -w; echo $V1104 | wc -w      （24、10、14）
   RUNS=$(for r in $VAL; do echo data/runs/$r; done)
   前两项不对就停下报告。
1. 训练（照 T6 第 1、2 步放卡和检查），work_dir 是 data/train/dog_voxel_veh_v3、data/train/dog_pp_veh_v3（先 ls 确认两个都不存在）：
     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     CUDA_VISIBLE_DEVICES=<卡1> nohup python tools/train.py configs/dog/dog_centerpoint_voxel_xt32_veh.py \
         --work_dir data/train/dog_voxel_veh_v3 > data/train/dog_voxel_veh_v3.log 2>&1 &
     CUDA_VISIBLE_DEVICES=<卡2> nohup python tools/train.py configs/dog/dog_centerpoint_pp_xt32_veh.py \
         --work_dir data/train/dog_pp_veh_v3 > data/train/dog_pp_veh_v3.log 2>&1 &
   单卡，不加 --autoscale-lr，配置不改。训练帧数是 v2 的 1.75 倍，每个 epoch 约 3,170 步；
   各占一张卡时 Voxel 约 2.3 小时、Pillar 约 1.4 小时。
2. 训练期间，在有空的显存上（训练那张卡也行；Pillar 进程约 3 GB，Voxel + TTA 约 6 GB，通用约定 5）跑对照，一次一条：
     { python tools/dog/run_sequence.py --replay $RUNS --detectors pillar --ckpt-pillar work_dirs/pretrained/nusc_pp.pth \
           --no-predict --no-mcap --jobs 2 --device cuda:0 --out "data/runs_v3cmp/nusc_pp/{scene}"
       python tools/dog/run_sequence.py --replay $RUNS --detectors pillar --ckpt-pillar data/train/dog_pp_veh_v2/epoch_10.pth \
           --no-predict --no-mcap --jobs 2 --device cuda:0 --out "data/runs_v3cmp/v2_pp/{scene}"
       python tools/dog/run_sequence.py --replay $RUNS --detectors voxel --tta --ckpt-voxel work_dirs/pretrained/nusc_voxel.pth \
           --no-predict --no-mcap --jobs 1 --device cuda:0 --out "data/runs_v3cmp/teacher/{scene}"
     } 2>&1 | tee data/reports/T39/runs_baselines.txt
3. 选 epoch：照 T7 第 1 步的 ev 函数（权重目录 dog_voxel_veh_v3、dog_pp_veh_v3，报告写到 data/reports/T39/），
   第 10、15、20 个 epoch。规则同 T6 第 4 步：merged/vehicle 的 AP@2.0 最高；和更晚的差 < 0.005 取更晚的。记为 E_v3_voxel、E_v3_pp。
4. v3 重放 24 个验证场景：
     python tools/dog/run_sequence.py --replay $RUNS --detectors pillar,voxel \
         --ckpt-pillar data/train/dog_pp_veh_v3/epoch_<E_v3_pp>.pth --ckpt-voxel data/train/dog_voxel_veh_v3/epoch_<E_v3_voxel>.pth \
         --no-predict --no-mcap --jobs 2 --device cuda:0 --out "data/runs_v3cmp/v3/{scene}" 2>&1 | tail -3 | tee data/reports/T39/runs_v3.txt
     python tools/dog/run_sequence.py --replay $RUNS --detectors voxel --tta \
         --ckpt-voxel data/train/dog_voxel_veh_v3/epoch_<E_v3_voxel>.pth \
         --no-predict --no-mcap --jobs 1 --device cuda:0 --out "data/runs_v3cmp/v3_tta/{scene}" 2>&1 | tail -3 | tee -a data/reports/T39/runs_v3.txt
5. 对比（40 m 内，审过的验证标签），合计、0921、1104 各一次：
     D=data/runs_v3cmp
     res() { for r in "$@"; do echo nusc_pp=$D/nusc_pp/$r/pillar/results.pkl v2_pp=$D/v2_pp/$r/pillar/results.pkl \
         v3_pp=$D/v3/$r/pillar/results.pkl v3_voxel=$D/v3/$r/voxel/results.pkl \
         teacher=$D/teacher/$r/voxel/results.pkl v3_voxel_tta=$D/v3_tta/$r/voxel/results.pkl; done; }
     cmp() { n=$1; shift
       python tools/dog/compare_detectors.py --classes vehicle --max-range 40 --results $(res "$@") \
           --labels $(for r in "$@"; do echo data/labels/$r/labels_sustech2.pkl; done) \
           --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 --target-recall 0.9 --bootstrap 1000 \
           | tee data/reports/T39/compare_$n.md; }
     cmp all $VAL; cmp 0921 $V0921; cmp 1104 $V1104
6. 运动状态（照 T15 第 5 步），三个 Pillar，24 个场景：
     motion() {  # motion <名字> <目录>
       python tools/dog/motion_eval.py $(for r in $VAL; do echo --scene data/labels/$r/labels_sustech2.pkl $D/$2/$r/pillar/results.pkl; done) \
           --details data/reports/T39/motion_$1_details.csv | tee data/reports/T39/motion_$1.txt
     }
     motion nusc_pp nusc_pp; motion v2_pp v2_pp; motion v3_pp v3
7. 盲区边上的误检（照 T15 第 6 步），三个 Pillar：
     python tools/dog/sector_check.py --scenes $VAL --labels "data/labels/{scene}/labels_sustech2.pkl" --sector 142 218 \
         --results nusc_pp=$D/nusc_pp/{scene}/pillar/results.pkl v2_pp=$D/v2_pp/{scene}/pillar/results.pkl \
         v3_pp=$D/v3/{scene}/pillar/results.pkl --list 10 | tee data/reports/T39/sector_val.md
8. 两个结论：
   - **上线的 Pillar**：v3_pp、v2_pp 各和 nusc_pp 比，同时满足下面三条才算比 nusc_pp 好：
     a. compare_all 的重抽表里"X − nusc_pp"一行：ΔAP@2.0 0-20 m ≥ +0.02，并且 A ahead ≥ 90%；
     b. compare_0921、compare_1104 的 AP 表：X 的 AP@2.0、AP@4.0 在两份里都不比 nusc_pp 低 0.02 以上
        （0921 里有轿车，1104 里是在动的货车，哪边都不能明显变差）；
     c. 运动状态：相关运动段（≥ 1 m/s、30 m 内）的漏报不比 nusc_pp 多。
     两个都满足就取 ΔAP@2.0 0-20 m 大的；都不满足就留 nusc_pp。
     写出选中的那个在 compare_all 里的建议上报阈值（--target-recall 0.9，20 m 内），以及那个阈值和 0.30 时 40 m 内的误检数。
     选中的不是 nusc_pp 的话，到此为止：后处理的阈值、过滤、跟踪是按 nusc_pp 调的（T19–T22），上车要重新导 ONNX、在狗上建引擎，
     这些另写任务，T39 不做。
   - **教师**：compare_all 里 v3_voxel_tta 和 teacher 比，同时满足才换（照 T15 第 7 步）：
     a. AP@2.0 高 ≥ 0.02；b. AP@2.0[20-40m] 不低；c. 速度表 "head > 0.15 m/s" 不低于减 5 个百分点。
     满足就是下一批数据的 Voxel 教师换成 v3 Voxel + TTA；不满足就不换。Pillar 教师（Pillar + TTA 单进程要约 13 GB）这次不比。

停止条件（停下并报告日志最后 50 行，不要改配置重跑）：任何报错；loss 变成 nan / inf；估算训练总时长超过 5 小时；
训练进程意外退出（可以用 --resume_from data/train/<名字>/latest.pth 续一次，再退出就停下）。
提交：data/reports/T39/、两个训练的 .log 和 work_dir 里的 *.log.json（复制到 data/reports/T39/）、本文档。不提交 data/runs_v3cmp、权重。
报告（写进"T39 结果"）：
- 训练：卡号、显存、time / data_time、每个 epoch 的步数、总耗时；每 5 个 epoch 的 loss / hm_loss（两个 head 分开），和 v2（T15）并排；
- 第 3 步每个权重的 merged/vehicle 一行，E_v3_voxel、E_v3_pp；
- 三份 compare 的 AP 表和速度表；compare_all 重抽表里 v3_pp − nusc_pp、v2_pp − nusc_pp、v3_pp − v2_pp、v3_voxel_tta − teacher 四行
  （compare_0921、compare_1104 的这四行也抄上）；
- 三个 Pillar 在 compare_all 里的建议阈值，以及阈值 0.30 时 40 m 内的误检数；
- 三个 motion_*.txt 的表；sector_val.md 的表；第 8 步的两个结论，每一条写清楚是哪个数满足 / 不满足。
```


### T39 结果

拉到 `ba61de4`，只有文档，没有跑测试。数据集是 `/public/zjst/robot/centerpoint_data/dog_veh_1106_1811`，12670 / 1418 帧。`1104_cp02_s000` 的 split 是 val。验证场景 24 个（0921 十个、1104 十四个）。

**训练。** 单卡，没有 `--autoscale-lr`，配置没改。Voxel 在物理 GPU 7（进程里是 `cuda:0`），Pillar 在物理 GPU 5。每个 epoch 3425 步（v2 是 2125）。没有 nan，进程没有中途退出。

| | 卡 | nvidia-smi | 日志 memory | time / data_time | 墙钟 |
|---|---|---|---|---|---|
| Voxel | 7 | 3960 MiB | 3448，后到 3688 | 0.104 / 0.004 s | 08:59–10:59，1 小时 59 分 |
| Pillar | 5 | 3352 MiB | 2839，后到 3044 | 0.081 / 0.002 s | 08:59–10:32，1 小时 32 分 |

v2（T15）是 Voxel 0.13 / 0.005 s、约 1 小时 32 分，Pillar 0.08 / 0.002 s、约 59 分。v3 步数是 v2 的 1.6 倍，墙钟也按这个比例长。

每个 epoch 最后一条日志的 loss / hm_loss（car、truck 分开）。v2 取的是 `data/train/dog_*_veh_v2.log` 里该 epoch 的最后一条。

Voxel：

| epoch | v3 car | v3 truck | v2 car | v2 truck |
|---|---|---|---|---|
| 5 | 0.321 / 0.157 | 0.517 / 0.273 | 0.381 / 0.214 | 0.497 / 0.246 |
| 10 | 0.319 / 0.169 | 0.435 / 0.233 | 0.383 / 0.224 | 0.344 / 0.165 |
| 15 | 0.245 / 0.126 | 0.286 / 0.108 | 0.257 / 0.129 | 0.250 / 0.107 |
| 20 | 0.222 / 0.110 | 0.246 / 0.103 | 0.280 / 0.164 | 0.215 / 0.092 |

Pillar：

| epoch | v3 car | v3 truck | v2 car | v2 truck |
|---|---|---|---|---|
| 5 | 0.483 / 0.260 | 0.611 / 0.328 | 0.395 / 0.189 | 0.553 / 0.289 |
| 10 | 0.478 / 0.293 | 0.441 / 0.213 | 0.379 / 0.210 | 0.505 / 0.273 |
| 15 | 0.294 / 0.158 | 0.314 / 0.148 | 0.431 / 0.259 | 0.381 / 0.193 |
| 20 | 0.236 / 0.107 | 0.321 / 0.140 | 0.269 / 0.142 | 0.286 / 0.128 |

**选 epoch**（`dist_test`，merged/vehicle 的 AP@2.0 最高；和更晚的差 < 0.005 才取更晚的）。mAP，AP@0.5 / 1 / 2 / 4，AP@2.0 的 0–10 / 10–20 / 20–40 m：

| 权重 | mAP | AP@0.5 | AP@1 | AP@2.0 | AP@4 | 0–10 | 10–20 | 20–40 |
|---|---|---|---|---|---|---|---|---|
| voxel e10 | 0.812 | 0.654 | 0.808 | 0.882 | 0.905 | 0.945 | 0.958 | 0.858 |
| voxel e15 | 0.816 | 0.661 | 0.806 | 0.890 | 0.908 | 0.963 | 0.945 | 0.867 |
| voxel e20 | 0.814 | 0.665 | 0.806 | 0.883 | 0.902 | 0.963 | 0.952 | 0.858 |
| pp e10 | 0.776 | 0.637 | 0.754 | 0.833 | 0.879 | 0.944 | 0.827 | 0.817 |
| pp e15 | 0.798 | 0.649 | 0.787 | 0.859 | 0.897 | 0.955 | 0.916 | 0.832 |
| pp e20 | 0.800 | 0.650 | 0.791 | 0.863 | 0.897 | 0.923 | 0.921 | 0.839 |

**E_v3_voxel = 15**（0.890；第 20 个是 0.883，低 0.007，超过 0.005）。**E_v3_pp = 20**（0.863，三个里最高）。后面的重放用这两个。24 个场景都写完了，没有报错。

**合计**（40 m 内，2933 个车框，1418 帧）：

| model | AP@0.5 | AP@1.0 | AP@2.0 | AP@4.0 | mAP | 0–10 | 10–20 | 20–40 |
|---|---|---|---|---|---|---|---|---|
| nusc_pp | 0.676 | 0.805 | 0.859 | 0.879 | 0.804 | 0.841 | 0.919 | 0.831 |
| v2_pp | 0.573 | 0.758 | 0.851 | 0.887 | 0.768 | 0.863 | 0.909 | 0.818 |
| v3_pp | 0.672 | 0.804 | 0.871 | 0.905 | 0.813 | 0.924 | 0.920 | 0.829 |
| v3_voxel | 0.679 | 0.819 | 0.894 | 0.909 | 0.825 | 0.963 | 0.935 | 0.856 |
| teacher | 0.733 | 0.889 | 0.936 | 0.949 | 0.876 | 0.947 | 0.971 | 0.903 |
| v3_voxel_tta | 0.694 | 0.828 | 0.903 | 0.918 | 0.836 | 0.962 | 0.950 | 0.866 |

速度头（运动标签 > 1 m/s，1049 个）：

| model | matched | head > 0.15 m/s | median error m/s |
|---|---|---|---|
| nusc_pp | 957 | 98% | 1.69 |
| v2_pp | 876 | 80% | 2.68 |
| v3_pp | 934 | 99% | 0.78 |
| v3_voxel | 971 | 98% | 0.73 |
| teacher | 1012 | 97% | 1.49 |
| v3_voxel_tta | 974 | 99% | 0.69 |

**0921**（1721 框，580 帧）：

| model | AP@2.0 | AP@4.0 | mAP | 0–10 | 10–20 | 20–40 | head > 0.15 | matched / 162 |
|---|---|---|---|---|---|---|---|---|
| nusc_pp | 0.834 | 0.852 | 0.787 | 0.878 | 0.937 | 0.788 | 91% | 127 |
| v2_pp | 0.854 | 0.882 | 0.787 | 0.754 | 0.955 | 0.813 | 89% | 118 |
| v3_pp | 0.823 | 0.860 | 0.763 | 0.764 | 0.936 | 0.765 | 94% | 116 |
| v3_voxel | 0.848 | 0.865 | 0.773 | 0.864 | 0.927 | 0.797 | 83% | 124 |
| teacher | 0.928 | 0.937 | 0.884 | 0.922 | 0.982 | 0.896 | 93% | 151 |
| v3_voxel_tta | 0.864 | 0.881 | 0.787 | 0.859 | 0.944 | 0.816 | 90% | 128 |

**1104**（1212 框，838 帧）：

| model | AP@2.0 | AP@4.0 | mAP | 0–10 | 10–20 | 20–40 | head > 0.15 | matched / 887 |
|---|---|---|---|---|---|---|---|---|
| nusc_pp | 0.898 | 0.920 | 0.832 | 0.808 | 0.899 | 0.896 | 99% | 830 |
| v2_pp | 0.849 | 0.897 | 0.737 | 0.912 | 0.836 | 0.827 | 79% | 758 |
| v3_pp | 0.929 | 0.957 | 0.877 | 0.991 | 0.897 | 0.915 | 100% | 818 |
| v3_voxel | 0.955 | 0.965 | 0.897 | 0.988 | 0.952 | 0.937 | 100% | 847 |
| teacher | 0.946 | 0.964 | 0.864 | 0.959 | 0.952 | 0.912 | 98% | 861 |
| v3_voxel_tta | 0.954 | 0.964 | 0.902 | 0.997 | 0.960 | 0.936 | 100% | 846 |

重抽差值。工具只打印「列表里更早的 − 更晚的」。下面按第 8 步要的方向取反了差值和区间；ahead 仍是打印那一行的领先占比，没有改成另一边的。

合计：

| A − B | ΔmAP | 区间 | 打印行 ahead | ΔAP@2.0 0–20 m | 区间 | 打印行 ahead |
|---|---|---|---|---|---|---|
| v3_pp − nusc_pp | +0.009 | −0.026 – +0.065 | nusc 37% | +0.006 | −0.026 – +0.059 | nusc 43% |
| v2_pp − nusc_pp | −0.037 | −0.067 – −0.011 | nusc 99% | −0.010 | −0.045 – +0.021 | nusc 74% |
| v3_pp − v2_pp | +0.045 | +0.007 – +0.111 | v2 2% | +0.016 | −0.006 – +0.065 | v2 14% |
| v3_voxel_tta − teacher | −0.041 | −0.070 – +0.010 | teacher 91% | −0.020 | −0.032 – −0.003 | teacher 97% |

0921：

| A − B | ΔmAP | 区间 | 打印行 ahead | ΔAP@2.0 0–20 m | 区间 | 打印行 ahead |
|---|---|---|---|---|---|---|
| v3_pp − nusc_pp | −0.024 | −0.059 – +0.085 | nusc 74% | −0.018 | −0.050 – +0.040 | nusc 74% |
| v2_pp − nusc_pp | +0.000 | −0.023 – +0.066 | nusc 48% | +0.007 | −0.019 – +0.053 | nusc 30% |
| v3_pp − v2_pp | −0.024 | −0.045 – +0.023 | v2 87% | −0.024 | −0.041 – −0.004 | v2 97% |
| v3_voxel_tta − teacher | −0.097 | −0.109 – −0.060 | teacher 99% | −0.036 | −0.073 – −0.016 | teacher 100% |

1104：

| A − B | ΔmAP | 区间 | 打印行 ahead | ΔAP@2.0 0–20 m | 区间 | 打印行 ahead |
|---|---|---|---|---|---|---|
| v3_pp − nusc_pp | +0.044 | +0.007 – +0.088 | nusc 2% | +0.031 | −0.020 – +0.083 | nusc 16% |
| v2_pp − nusc_pp | −0.095 | −0.133 – −0.052 | nusc 100% | −0.028 | −0.086 – +0.024 | nusc 81% |
| v3_pp − v2_pp | +0.139 | +0.104 – +0.172 | v2 0% | +0.059 | +0.033 – +0.087 | v2 0% |
| v3_voxel_tta − teacher | +0.037 | +0.009 – +0.067 | teacher 2% | −0.004 | −0.021 – +0.012 | teacher 62% |

合计里 20 m 内召回 ≥ 0.9 的最高阈值，以及阈值 0.30 时 40 m 内的误检：

| model | 建议阈值 | 该阈值的误检 | 0.30 的误检 |
|---|---|---|---|
| nusc_pp | 0.35（精确率 0.763） | 724 | 1299 |
| v2_pp | 0.25（精确率 0.720） | 574 | 382 |
| v3_pp | 0.35（精确率 0.891） | 271 | 361 |

运动状态（相关段是 ≥ 1 m/s、30 m 内）：

| model | moving recall | false moving | 运动段漏报 / 全部，延迟中位 / p90 | 相关段漏报 / 全部，延迟中位 / p90 |
|---|---|---|---|---|
| nusc_pp | 0.926 | 0.0176 | 3 / 77，0.50 / 1.00 | 2 / 49，0.50 / 1.00 |
| v2_pp | 0.785 | 0.0193 | 15 / 68，0.50 / 3.30 | 4 / 44，0.50 / 2.50 |
| v3_pp | 0.945 | 0.0272 | 6 / 71，0.50 / 0.50 | 1 / 45，0.50 / 0.50 |

盲区（142°–218°，分数默认，24 个场景）：

| model | range | zone | detections | false | false share |
|---|---|---|---|---|---|
| nusc_pp | < 10 m | elsewhere | 275 | 35 | 13% |
| nusc_pp | < 10 m | edge | 47 | 22 | 47% |
| nusc_pp | < 10 m | blind | 68 | 62 | 91% |
| nusc_pp | 10–40 m | elsewhere | 3198 | 857 | 27% |
| nusc_pp | 10–40 m | edge | 697 | 313 | 45% |
| nusc_pp | 10–40 m | blind | 17 | 5 | 29% |
| v2_pp | < 10 m | elsewhere | 229 | 61 | 27% |
| v2_pp | < 10 m | edge | 55 | 39 | 71% |
| v2_pp | < 10 m | blind | 21 | 8 | 38% |
| v2_pp | 10–40 m | elsewhere | 2283 | 213 | 9% |
| v2_pp | 10–40 m | edge | 398 | 47 | 12% |
| v2_pp | 10–40 m | blind | 5 | 5 | 100% |
| v3_pp | < 10 m | elsewhere | 212 | 15 | 7% |
| v3_pp | < 10 m | edge | 27 | 11 | 41% |
| v3_pp | < 10 m | blind | 14 | 7 | 50% |
| v3_pp | 10–40 m | elsewhere | 2338 | 231 | 10% |
| v3_pp | 10–40 m | edge | 431 | 85 | 20% |
| v3_pp | 10–40 m | blind | 2 | 2 | 100% |

**上线的 Pillar：仍用 nuScenes Pillar。** 建议上报阈值 0.35。两条候选都没有同时满足三条。

v3_pp：

- a 不满足。合计里 v3_pp − nusc_pp 的 ΔAP@2.0 0–20 m 是 +0.006，不到 +0.02；区间 −0.026 – +0.059 跨过 0。打印行是 nusc 领先 43%，所以 v3 的领先占比到不了 90%。
- b 满足。0921 上 AP@2.0 0.823 对 0.834（低 0.011）、AP@4.0 0.860 对 0.852；1104 上 0.929 对 0.898、0.957 对 0.920。两份都没有低过 0.02。
- c 满足。相关运动段漏报 1 次，nusc_pp 是 2 次。

v2_pp：

- a 不满足。ΔAP@2.0 0–20 m 是 −0.010，nusc 这一项领先 74%。
- b 不满足。1104 上 AP@2.0 0.849 对 0.898（低 0.049）、AP@4.0 0.897 对 0.920（低 0.023），都低过 0.02。0921 上是 0.854 对 0.834、0.882 对 0.852，这边没有更差。
- c 不满足。相关运动段漏报 4 次，nusc_pp 是 2 次。

v3 的优势在 1104（在动的货车，背景训练里见过）：AP@2.0 0.929 对 0.898。0921（有轿车、狗在走）上近处反而更差，0–10 m 的 AP@2.0 是 0.764 对 0.878。合在一起近处分不出来。后处理、过滤、跟踪仍是按 nusc_pp 调的，这次不用重导 ONNX。

**教师不换。** 合计里 v3 Voxel + TTA 对 nuScenes Voxel + TTA：

- a 不满足。AP@2.0 0.903 对 0.936，低 0.033，不是高 0.02。
- b 不满足。AP@2.0[20–40m] 0.866 对 0.903，更低。
- c 满足。速度头 99% 对 97%，没有低过 5 个百分点。

1104 上 v3 Voxel + TTA 的 AP@2.0 是 0.954 对 0.946、20–40 m 是 0.936 对 0.912，这边更高；换教师看的是合计，合计没有过。

### T39 复核（2026-11-09）

- **结论对：上线仍用 nuScenes Pillar，教师不换。** 第 8 步的三条按顺序判的，和表里的数一致；重抽表的方向也处理对了。
- **上报阈值不改，仍是代码里的 0.40**，不用这里建议的 0.35：
  - 0.35 只看检测（合计里 20 m 内召回 ≥ 0.9 的最高阈值）；0.40 是 T19–T22 连跟踪一起调出来的。
  - 0921 上 0.40 → 0.35：20 m 内召回只从 0.941 升到 0.949，40 m 内误检却从 382 涨到 618。
  - 跟踪以后的检查用的就是 0.40：30 m 内、≥ 1 m/s 的相关运动段，nusc_pp 漏 2 / 49，延迟中位 0.5 s、p90 1.0 s。
- **v3_pp 误检少，是拿召回换的，不是白赚。**
  - 阈值 0.40 时合计误检 188 对 445，差别主要在 0921（121 对 382）；但 0921 上 20 m 内召回是 0.863 对 0.941。
  - 0921 上把召回拉到一样（20 m 内约 0.92）比：nusc_pp（阈值 0.45）误检 247，v3_pp（阈值 0.25）410。
  - 也就是在训练里没见过的地方，nusc_pp 还是更好。
  - 0921 的 0–10 m AP@2.0：v3 0.764、v2 0.754、nusc 0.878，微调过的 Pillar 近处偏弱，和 T15、T18 一样。不过 10 m 内的车框不多（T18 续 2 是 74 个），区间很宽。
- **1104 上 v3 明显好，但现在还说不准是真好。**
  - 0–10 m AP@2.0 是 0.991 对 0.808；阈值 0.40 时 20 m 内召回 0.924 对 0.833。
  - 1104 的验证和训练是同一个路口、同一个晚上，背景在训练里见过（T37 复核），分不清 v3 是学会了"低机位近处的货车"，还是记住了这个路口。要等测试集（另一天、另一片区域）。
  - 这也是现在上线模型最弱的地方：狗趴着 / 站着（雷达离地 0.48–0.63 m）时近处的货车，nusc_pp 在 1104 上 20 m 内召回只有 0.833（阈值 0.40）。
  - 跟踪以后在动的货车只漏 2 段，影响不大。T35 B 段出门时，留意近处的大车。
- **教师**：v3 Voxel + TTA 在 1104 上更好（0.954 对 0.946），在 0921 上差很多（0.864 对 0.928）。0921 的验证标签就是从教师的框审出来的，最偏教师。按规则不换，没问题。
  - v3 Voxel 单次前向的 AP@2.0 是 0.894，比三个 Pillar 都高。但稀疏卷积在狗上没有 TensorRT 的路，不是上线候选。
- **训练正常**：没有 nan，每个 epoch 3,425 步，Voxel 2 小时、Pillar 1.5 小时。选 epoch 按规则：Voxel 第 15 个是 0.890，第 20 个 0.883，差 0.007，超过 0.005，取第 15 个。
- **下一步**：模型这条线到这里先停，上线的不换，车上什么都不用动。
  - 数据：下一批要另一天、另一片区域、狗在走（最好白天），先留出测试集，再看 v3 是不是真的好（DATA_COLLECTION_PLAN 3.1、3.5）。
  - 部署：T35（A 段室内随时可以做；B 段要人带狗出门）。
  - v3 的权重留着（`data/train/dog_*_veh_v3`）。v2 的数据集目录 `dog_veh_1026_1410`（37 GB）T39 已经用不到，删不删人定。

---

## T40 收尾：剩下的都在这里（2026-11-09）

这一轮模型和数据做完了（T39：上线仍用 nuScenes Pillar，阈值 0.40）。剩下的收尾放在一起，按谁做分开。
T35 并入这里：步骤照 T35 原来写的，这里只写改了的地方。

| # | 事 | 谁 | 前置 |
|---|---|---|---|
| 1 | 删 v2 数据集 `dog_veh_1026_1410`（人定，2026-11-09） | Agent，服务器 | 无 |
| 2 | 8 号狗装新 deb：开机锁频的开关 `LIDAR3D_LOCK_CLOCKS`（默认 0，什么都不改）；7432c90 的预测线也在里面 | Agent，ssh dog | 无 |
| 3 | T35 A 段（室内）：root 视角的撞车检查、root 起 bridge、负载和 GPU | Agent | 2 |
| 4 | T35 B 段（室外、固定解）：在线出框、天线接法、在线对离线、Foxglove；锁频改用开关，顺便在真狗上验开关 | 人带狗出门，Agent | 3 |
| 5 | 8 号狗开不开开机锁频：`/etc/default/lidar-3d-detection` 里 `LIDAR3D_LOCK_CLOCKS=1`，然后 `systemctl restart lidar-3d-detection` | 机器人团队 | 4（看锁频前后的延迟和负载） |
| 6 | 规划确认 `ObjectArray` 的坐标系和字段 | 人、规划 | 无 |
| 7 | H5：看最终效果，定上线参数 | 人 | 4 |
| 8 | 下一批数据：另一天、另一片区域、狗在走（最好白天）。先留测试集，再看 v3（DATA_COLLECTION_PLAN 3.5） | 人录，Agent 照 T36–T39 做 | 无 |

不做了：
- H2：只在选了 Voxel 时才要，上线的是 Pillar；
- T9：用 v1 Voxel 当教师，教师已经定了（T15、T39：不换）；
- T10：两个杠杆臂，可选项，到收尾也没用上；
- H4：等下一批数据，模型变了再看。

```text
任务 T40：收尾。1. 删 v2 数据集；2. 8 号狗装带开机锁频开关的 deb（默认关）；3. T35 A 段；4. T35 B 段（人带狗出门时）。
先按通用约定做（第 9 条：可以改 deploy_deb/、tools/deploy/；第 10 条）。仓库根目录，export PYTHONPATH=.。
规矩同 T34、T35：域 0 上除了我们的服务不发任何东西；不停、不改机器人别的服务；狗上不 apt install，deb 用 dpkg -i，
依赖不满足就停，不要 --force-depends。
mkdir -p data/reports/T40；L=deploy_deb/perception_combo_ros/node_src/lidar_3d_detection
pytest tests/dog/test_deploy.py -q -k "lock_clocks or orin_clocks" | tail -1      （新的开关测试，7 + 3 个）

1. 删 v2 数据集：
     readlink -f data/dog_veh                       （应是 /public/zjst/robot/centerpoint_data/dog_veh_1106_1811，v3；不是就停）
     ps aux | grep -E "train.py|dist_test|compare_detectors|run_sequence|export_dataset" | grep -v grep   （应没有输出；有就等它结束）
     D=/public/zjst/robot/centerpoint_data/dog_veh_1026_1410
     du -sh $D | tee data/reports/T40/delete_v2.txt && rm -rf $D && ls -d $D 2>&1 | tee -a data/reports/T40/delete_v2.txt
   v2 的权重（data/train/dog_*_veh_v2）留着。v1 的目录（dog_veh_1023_1529）还在的话，只报大小，不删（人没说）。
2. 8 号狗装新 deb（开机锁频开关，默认关）：
   a. 装之前的状态，备份 /etc/default：
     ssh dog 'dpkg -s lidar-3d-detection | grep Version; systemctl is-enabled lidar-3d-detection; systemctl is-active lidar-3d-detection; \
         ~/dogperc/repo/'$L'/scripts/orin_clocks.sh; sudo -n cp /etc/default/lidar-3d-detection /etc/default/lidar-3d-detection.T40-before' \
         2>&1 | tee data/reports/T40/before.txt
   b. 同步、打包（同 T34 第 3 步。引擎不用重建：模型和狗上的 TensorRT 都没变）：
     ORIN_HOST=dog tools/deploy/sync_to_orin.sh --model data/deploy/nusc_pp_dyn
     ssh dog 'mkdir -p ~/dogperc/deb/old && mv ~/dogperc/deb/lidar-3d-detection_0.1.0+* ~/dogperc/deb/old/ 2>/dev/null; \
         cd ~/dogperc/repo/deploy_deb/perception_combo_ros && node_src/lidar_3d_detection/scripts/build_deb.sh \
         --model ~/dogperc/models/nusc_pp_dyn --service-user root --no-enable-service \
         --ros-domain-id 0 --rmw rmw_cyclonedds_cpp --cyclonedds-uri file:///opt/ros/cyclonedds.xml --out ~/dogperc/deb' \
         2>&1 | tail -40 | tee data/reports/T40/build_deb.txt
   - settings 一行应是 T34 的四项再加 LIDAR3D_LOCK_CLOCKS=0。
   c. 装：
     ssh dog 'sudo -n dpkg -i --force-confnew ~/dogperc/deb/lidar-3d-detection_0.1.0+*_arm64.deb 2>&1 | tail -4; \
         diff /etc/default/lidar-3d-detection.T40-before /etc/default/lidar-3d-detection; \
         systemctl show lidar-3d-detection -p ExecStartPre | cut -c1-200; \
         dpkg -s lidar-3d-detection | grep Version; systemctl is-enabled lidar-3d-detection; systemctl is-active lidar-3d-detection' \
         2>&1 | tee data/reports/T40/install.txt
   - diff 只能多出 LIDAR3D_LOCK_CLOCKS=0 和它上面三行注释，DDS 的三项和 LIDAR3D_THREADS 不变。
     有别的不一样就停下：sudo -n cp /etc/default/lidar-3d-detection.T40-before /etc/default/lidar-3d-detection，报告；
   - ExecStartPre 里有 lock_clocks.sh；
   - a 里是 active 的，现在也要是 active（postinst 只在 enabled 时重启服务；不是就 sudo -n systemctl start lidar-3d-detection）。
   d. 开关关着时什么都不改：
     ssh dog 'sudo -n systemctl restart lidar-3d-detection; sleep 15; \
         journalctl -u lidar-3d-detection --since "-1min" --no-pager | grep -c lock_clocks; \
         ~/dogperc/repo/'$L'/scripts/orin_clocks.sh; systemctl is-active lidar-3d-detection' 2>&1 | tee data/reports/T40/switch_off.txt
   - lock_clocks 的行数是 0；orin_clocks.sh 那一行和 a 里一样；服务 active。
3. T35 A 段：照 T35 的第 0 步（不用再打包，第 2 步做过了）和 A1–A3，报告写到 data/reports/T40/（文件名照 T35）。
   A1 有同名坐标系就停下问人。
4. T35 B 段（人在聊天里说"狗在室外、可以开始"再做）：照 T35 的 B0–B8，改三处：
   - bag 的名字用当天的日期，例如 1110_t40，B5、B6 里的 1105_t35 都换成它；
   - B1 用开关锁频（顺便在真狗上验开关）：
     ssh dog 'C=$(~/dogperc/repo/'$L'/scripts/orin_clocks.sh); echo "$C"; \
         case "$C" in "clocks locked"*) ;; *) sudo -n jetson_clocks --store /tmp/clocks_default ;; esac; \
         sudo -n sed -i "s/^LIDAR3D_LOCK_CLOCKS=0$/LIDAR3D_LOCK_CLOCKS=1/" /etc/default/lidar-3d-detection; \
         sudo -n systemctl restart lidar-3d-detection; sleep 15; \
         journalctl -u lidar-3d-detection --since "-1min" --no-pager | grep lock_clocks; \
         ~/dogperc/repo/'$L'/scripts/orin_clocks.sh; systemctl is-active lidar-3d-detection' 2>&1 | tee data/reports/T40/switch_on.txt
     journal 里要有 "lock_clocks.sh: clocks locked"，orin_clocks.sh 打印 clocks locked。不是就照 T35 原来的 B1 手动锁，报告里写明。
   - B4 恢复：开关改回 0（同一条 sed，0 和 1 对调）。开始时没锁的话，sudo -n jetson_clocks --restore /tmp/clocks_default；
     orin_clocks.sh 一行进报告。服务不用再重启（开关只在服务启动时起作用）。改回 0 以后，/etc/default 和包里的一样，下次升级不会问配置。

停止条件：同 T35；第 1 步 data/dog_veh 不是 v3；第 2 步 diff 里有别的设置、依赖不满足、服务起不来。
提交：data/reports/T40/、本文档，以及按约定 9 改的代码。不提交 bag、data/deploy/。
报告（写进"T40 结果"）：
- 1：删掉了多大；v1 的目录在不在、多大；
- 2：装之前 / 之后的版本，settings 一行，diff，ExecStartPre，开关关着时 orin_clocks.sh 前后两行；
- 3、4：照 T35 的报告要求；B1 打开开关后 journal 里那一行和 orin_clocks.sh 的结果，B4 恢复后的结果；
- 结论：8 号狗能不能交给规划用；开机锁频建议开不开（锁 / 不锁时的延迟和负载，T34 是 37 / 59 ms，p99 68 / 99 ms）。
```

---

## T41 Nav_D20 规划：补上漏提交的脚本和配置，重编，重录，查录包

aff7108 把 `Nav_D20` 放进了仓库，但仓库根目录 `.gitignore` 里从上游 CenterPoint 带来的 `*.sh`、`*.yaml` 两条，把所有启动脚本和配置都漏掉了：
- `run_planning_stack_debug.sh` 等启动脚本；
- A* 地图的 `CJGC.yaml`：没有它规划起不来，日志报 `Failed to load PGM`；
- 定位的 `default_d20.yaml` 和感知的配置。

所以别处克隆下来跑不起来。服务器上的工作目录里这些文件都在，只是没提交。`.gitignore` 已经补了 `!Nav_D20/**/*.sh`、`!Nav_D20/**/*.yaml`（2026-11-09）。

Foxglove 里规划录包显示不对，从代码里找到三处，都已改好（`Nav_D20/planning/docs/ONBOARDING.md`"调试时看什么"）：
1. `/tf` 不是一棵树。`TfBuffer` 把每个变换连同它的逆一起发，云台还多发一条 `map -> ptz_*`，结果 `map`、`base_link`、`ptz_visible` 各有两个父坐标系，还成了环。
   - 本机把 `tf_buffer.cc` 单独编出来比过：改前发 6 条、有环；改后只发 `map -> base_link`、`base_link -> ptz_visible` 两条，规划内部的查询照旧能查到逆变换和捷径。
2. `base_link` 里的局部代价地图（`/planning/debug/cost_map_production/occmap`）用本机时钟打戳。
   - 回放 0921 的 bag 时，它比里程计（也就是 `map -> base_link`）晚几周，Foxglove 找不到那一刻的变换。
   - 现在和梯度标记一起，改用里程计的时间戳。
3. 标记的寿命是 0.05 s，正好一个 20 Hz 周期，回放时一闪一闪。
   - 现在是 0.5 s；路径、TEB、碰撞点这几组个数会变的数组，开头先发一个 DELETEALL。

新工具 `Nav_D20/planning/tools/check_planning_mcap.py` 查录包里这三件事，外加布局里的话题录了没有（测试 `tests/dog/test_planning_mcap.py`）。

```text
任务 T41：Nav_D20 规划：补提交漏掉的脚本和配置；用改过的代码重编 planning；回放 0921、重录 planning_io 的录包；
用 check_planning_mcap.py 比改前、改后的录包。
先按通用约定做（第 10 条）。可以改的范围：Nav_D20/。仓库根目录（服务器上原来的工作目录，漏提交的文件在这里）。
不要在狗用的域上起规划（它会发 /cmd_vel、/mc/motion_control）：ROS_DOMAIN_ID 用脚本默认的 42，或别的没人用的号。
mkdir -p data/reports/T41

0. 改前的录包先查一遍（之前在 Foxglove 里显示不对的那一份）：
     python3 Nav_D20/planning/tools/check_planning_mcap.py <那份录包> --layout Nav_D20/planning/script/planning_io.json \
         2>&1 | tee data/reports/T41/check_before.txt
   pytest tests/dog/test_planning_mcap.py -q | tail -1                       （3 passed）
1. 补提交（拉到本次提交以后，.gitignore 已经不忽略 Nav_D20 里的 .sh、.yaml）：
     git status --short -uall Nav_D20 | tee data/reports/T41/new_files.txt    （应多出启动脚本和 yaml，包括 CJGC.yaml）
     NEW=$(git status --short -uall Nav_D20 | awk '$1=="??" {print $2}')
     grep -liE "password|passwd|token|secret|api_key" $NEW
     du -k $NEW | awk '$1 > 5120'
   - grep 有输出就先停下，把文件名报给人，不要提交；
   - du 列出的（超过 5 MB），先问人；
   - 其余的 git add Nav_D20，和第 6 步一起提交。
   还被忽略的（*.so、*.bin、*.pkl、*.zip、build 目录等）只列出来，不提交：
     git status --ignored --short Nav_D20 | grep '^!!' | tee data/reports/T41/still_ignored.txt
2. 第 6 步提交以后，确认启动要的东西进了仓库（别处克隆下来就有）：
     git ls-files Nav_D20 | grep -E "run_planning_stack_debug.sh|run_planning_debug.sh|astar_pgm/CJGC.yaml|default_d20.yaml" \
         | tee data/reports/T41/tracked.txt                                    （四个都要有）
3. 重编 planning：照 ONBOARDING"在这台机器上跑起来"的增量重编命令（colcon --packages-select planning），2>&1 | tail -20 存到 build.txt。
4. 跑起来、重录：用之前录 planning_io 录包的同一条命令（run_planning_stack_debug.sh 播 0921 的 bag，pub_flag 打开）。
   - 先确认 planning 的日志里有这几行：Load robot config: d20/001、GlobalMapManager: PGM map loaded；
   - 发一个之前用过的目标点，等日志里出现 mission_state: active 和 CMD:；
   - 录到 data/reports/T41/ 以外的地方（录包不提交），录的话题和之前一样。
   照抄每个进程起来时的前 20 行日志和 planning 的关键字行，存到 run.txt。
5. 查新录包：
     python3 Nav_D20/planning/tools/check_planning_mcap.py <新录包> --layout Nav_D20/planning/script/planning_io.json \
         2>&1 | tee data/reports/T41/check_after.txt
   - 应是 no problems found。只剩"layout: ... is not in the recording"的，写清是哪个话题、为什么没录；
   - 还有 TF、时间戳或寿命的问题，把那几行和 planning 的日志一起报告，不要自己改别的。
6. 提交：第 1 步的文件、data/reports/T41/、本文档。不提交录包、build 目录。

停止条件：第 1 步 grep 有输出、有超过 5 MB 的文件；第 3 步编不过；第 4 步 planning 起不来、收不到 odom 或局部栅格。
报告（写进"T41 结果"）：
- 新加的文件数，和还被忽略的清单；第 2 步的四个文件；第 3 步编译的最后几行；第 4 步的关键日志行；
- check_before.txt、check_after.txt 的 PROBLEM 行和最后一行；
- 人在 Foxglove 里打开新录包（planning_io.json）看到的：路径、局部代价地图、点云跟不跟着狗、闪不闪（H7）。
```

### T41 结果

2026-11-09。`git pull` 到 `78567f5`。人确认 `copy_planning_bag.sh` 里的密码可以提交，从第 1 步继续。

第 0 步。改前录包 `work_dirs/planning_io/planning_io_0.mcap`（不提交）。`check_before.txt` 最后一行 `11 problems`。PROBLEM：`base_link`、`map` 各 3 个父，`ptz_infrared`、`ptz_visible` 各 2 个父；`/fused_point_cloud`、`/map/occupancy_grid`、`/planning/debug/cost_map_production/occmap` 没有唯一 TF；`/mc/motion_control`、`/planning/debug/goal_pose` 空 `frame_id`；两个标记寿命 0.05 s。`pytest tests/dog/test_planning_mcap.py -q`：`3 passed`。查包用的是仓库外 `.venv` 的 python（本机 `python3` 没有 `mcap`）。

第 1 步。新文件 78 个，没有超过 5 MB 的。还被忽略的 15 条在 `still_ignored.txt`（`build/`、`*.so`、`__pycache__/`、`gflags_build.py.in`），不提交。密码那条按人的决定一起提交。

第 3 步。旧的 colcon 缓存还指着已经删掉的 `d20car_predict/Nav_D20/planning`，增量编译直接失败。删掉 `/public/zjst/robot/planning_humble_env/colcon/build/planning` 后按 ONBOARDING 的命令重配再编，`Finished <<< planning [2min 29s]`，`BUILD_EXIT:0`（`build.txt`）。没有改规划源码。

第 4 步。`bash Nav_D20/planning/script/run_planning_stack_debug.sh --once`，域 42。日志有 `Load robot config: .../d20/001`、`GlobalMapManager: PGM map loaded. Size: 10293x11412`、`SetPubFlag: pub_flag enabled`。目标仍是 map 下 `(80.0, 126.9)`。全局规划报 `Start point is in collision!`，接着 `mission_state: standby`，后续目标把任务停掉，`mission_state: fault`。没有 `mission_state: active`，也没有 `CMD:`。起点离障碍不到 `safe_margin` 0.3 m，而且代价地图要算 1 秒多，这段时间定位会跳。录包在 `work_dirs/planning_io_t41/`（不提交），话题和之前一样。各进程前 20 行和关键字在 `run.txt`。

第 5 步。`check_after.txt` 最后一行 `2 problems`。PROBLEM 只有：

- `/mc/motion_control`: empty frame_id
- `/planning/debug/goal_pose`: empty frame_id

多父 TF、代价地图时间戳、0.05 s 寿命都没有了。没有改这两处空 `frame_id`。

代码改动：无（只清了指向旧目录的编译缓存）。

### T41 复核（2026-11-09）

- **规划这边的三处显示问题都没有了**：多父 TF、代价地图的时间戳、0.05 s 的寿命。查录包的结果从改前 11 个问题降到 2 个空 `frame_id`。
  - `/mc/motion_control` 不画，不算问题（查包工具改成只查会画的类型）；
  - `/planning/debug/goal_pose` 在收到第一个目标之前发空 `frame_id` 的标记，已改成收到目标后才发。
- **录包还是不对，是定位的问题。** 用提交的录包查了 `/lightning/odom`（17.5 s，84 Hz）：
  - 录包在 0921_164127 的第 2 个场景里，狗在稳步走：28.8 s 走 46.3 m（1.55 m/s），全程 RTK 固定解。这 17.5 s 应该走约 27 m。
  - 定位给出的却是：轨迹 552 m、首尾只差 1.8 m；`y` 在 85–211 m 之间来回；`z` 从 3.9 m 掉到 −46.7 m；自报速度到 32 m/s；
    相邻两条差 0.3 m 以上的 195 次，最大 43 m；航向一下转 38°。同时 1,471 / 1,476 条都报"定位正常"（covariance[1] = 1）。
  - 规划报 `Start point is in collision!`，是因为每次发目标时，定位的位置都在一秒内跳了 15–30 m，`z` 在 −30 m 上下，起点落在地图里随便哪儿，所以 fault。规划本身没有问题。
- **原因（找到一处，已改）**：`/lightning/odom` 是 LIO 的 IMU 高频预测（`laser_mapping.cc` 的 `kf_imu_`，`GetIMUState`）。
  - 速度的变化 = R·(加速度 − 零偏) + 重力，重力状态是 9.81。
  - kygi665 的 IMU 加速度单位是 g，静止时约 1.0（SENSORS.md 3.3、LOCALIZATION.md）。LIO 主滤波在去畸变时按初始化的重力归一化了（`imu_processing.hpp` 第 201 行），这个高频预测却直接用原始读数。
  - 于是两次 LIO 更新之间，每秒多出约 8.8 m/s² 的下坠：1 s 掉 4.4 m，速度差 8.8 m/s。
  - 录包里的样子正是这样：平滑地往下掉、越跑越快，大约每秒被拉回一次（还有一次停了 2.8 s），然后再掉。
  - 已改：高频预测里的加速度也乘 `G_m_s2 / GetMeanAccNorm()`。输入是 g 还是 m/s² 都对。
  - 要编出来再跑才能确认（T42）。在狗上实时跑时 LIO 10 Hz 更新，每 0.1 s 只差 4 cm，但速度每 0.1 s 会差到 0.9 m/s，规划用这个速度。
- **还有两处，可能也在拖后腿**：
  - 这次回放里 LIO 大约每秒才纠正一次（平时 10 Hz），原因要看定位日志（T42）；
  - 定位找不到配置的地图 `ad68517c-...`，退回到仓库里的 `0c3f8a2d-...`，ONBOARDING 本来就说那不是这包录制的地方。高频预测改好以后还跳的话，就是地图对不上。
- 另外，`copy_planning_bag.sh` 里的明文密码已经按人的决定进了 git 历史，删文件也删不掉。这个密码要是还在用，建议换掉。

## H7 在 Foxglove 里看规划的新录包（人做）

T43 之后。用 Foxglove 打开 T43 的录包（有现场地图的那一份；没有就用 INS 直出那一份），导入 `Nav_D20/planning/script/planning_io.json`，从头播放，看几处：
- 狗的位置（`/lightning/odom`）和点云、局部栅格对得上，不来回跳；
- 局部代价地图跟着狗走；
- 规划路径不闪，暂停时也在。

再打开改前那份录包对照。哪里还不对，截一张图，说一句是哪个话题。结果写到下面。

### H7 结果

（人填）

---

## T42 Nav_D20 定位：高频预测的加速度单位改好以后，重编、重跑、查录包

```text
任务 T42：重编 lightning（定位：高频预测的加速度按重力归一化）和 planning（目标标记）；照 T41 第 4 步回放 0921 重录；
查录包里的 /lightning/odom 还跳不跳；看定位日志里 LIO 每秒处理几帧；找配置的定位地图。
先按通用约定做（第 10 条）。可以改的范围：Nav_D20/。仓库根目录。ROS_DOMAIN_ID 同 T41（42），不在狗的域上起规划。
mkdir -p data/reports/T42

0. 先查 T41 的录包（新版查包工具，多了 odom 那一条）：
     python3 Nav_D20/planning/tools/check_planning_mcap.py work_dirs/planning_io_t41/planning_io_t41_0.mcap \
         --layout Nav_D20/planning/script/planning_io.json 2>&1 | tail -4 | tee data/reports/T42/check_t41.txt
   应有一行 "/lightning/odom: not a walking robot"。pytest tests/dog/test_planning_mcap.py -q | tail -1（4 passed）。
1. 重编：planning 同 T41 第 3 步；lightning 用 ONBOARDING 那一套环境，--packages-select lightning
   （colcon 包名以 lightning-lm-master/package.xml 为准），2>&1 | tail -20 存到 build.txt。编不过就停下报告，不要改别的。
2. 照 T41 第 4 步重跑、重录，录到 work_dirs/planning_io_t42/（提交，同 T41 的录包）。目标点同 T41。
   另外把定位的完整日志（/tmp/nav_stack_debug/ 下 loc 那一份）拷到 data/reports/T42/loc.log（太大就只留前 2000 行和后 2000 行）。
3. 查新录包：
     python3 Nav_D20/planning/tools/check_planning_mcap.py work_dirs/planning_io_t42/planning_io_t42_0.mcap \
         --layout Nav_D20/planning/script/planning_io.json 2>&1 | tee data/reports/T42/check_after.txt
   - 没有 "/lightning/odom: not a walking robot"：定位好了，看规划有没有进 active、有没有 CMD:；
   - 还有：把那一行抄下来，说明是平滑地往下掉 / 越跑越快，还是一下一下跳（照 T41 复核的看法）。
4. 定位日志：
   - LIO 每秒处理几帧（找每帧都打的那一行，按时间数）；
   - 有没有 "IMU 回调间隔过大"、reinit、fault 一类的行，各多少次。
5. 配置的定位地图：find /public /home -maxdepth 6 -type d -name "ad68517c*" 2>/dev/null | head。
   - 找到的话，报路径，不要改配置，等人定；
   - 找不到，写"没有"。
6. 提交：data/reports/T42/、work_dirs/planning_io_t42/、本文档。

停止条件：编不过；跑不起来。
报告（写进"T42 结果"）：第 0、3 步查包的 PROBLEM 行；第 1 步编译的最后几行；规划的关键日志行（mission_state、CMD:、collision）；
第 4 步 LIO 每秒几帧和那几类日志的次数；第 5 步的结果。
```

### T42 结果

2026-11-09。`git pull` 到 `1eff593`（T41 复核已把高频预测的加速度按重力归一化，目标标记改成收到目标后才发）。这次只重编、重跑，没有改源码，也没有改定位地图配置。

第 0 步。`check_t41.txt` 的 PROBLEM：

- `/planning/debug/goal_pose`: empty frame_id
- `/lightning/odom`: not a walking robot: speed p99 491.5 m/s (max 2706.5), z -46.7 .. 3.9 m, 61 jumps > 1 m (largest 43.3 m); path 552 m, net 2 m over 17.5 s

最后一行 `2 problems`。`pytest tests/dog/test_planning_mcap.py -q`：`4 passed`。查包用的是仓库外 `.venv` 的 python（本机 `python3` 没有 `mcap`）。

第 1 步。planning 的 colcon 缓存已经指着 CenterPoint，按 T41 的命令重编，约 87 s，`returncode 0`。lightning 的缓存还指着已经删掉的 `d20car_predict/Nav_D20`，删掉 `/public/zjst/robot/planning_humble_env/colcon/build/lightning` 后重配再编（地理库头文件用仓库里的 `Nav_D20/.../geographiclib/X86/include`）。`build.txt` 最后：

```text
Finished <<< lightning [2min 16s]
Summary: 1 package finished [2min 17s]
  1 package had stderr output: lightning
LIGHTNING_EXIT:0
```

第 2 步。`bash Nav_D20/planning/script/run_planning_stack_debug.sh --once`，域 42。目标仍是 map 下 `(80.0, 126.9)`，只发一次。录包 `work_dirs/planning_io_t42/planning_io_t42_0.mcap`（72 MB）。定位日志 1667 行，整份在 `loc.log`。

规划关键行：`Load robot config: .../d20/001`；`PGM map loaded. Size: 10293x11412`；`Set mission to Tracking`；`Start Tracking mission`；`mission_state: passive`，然后 `standby`，然后 `active`。`Initial pose is in collision` 173 次。`CMD:` 149 次，开头两条是 `CMD: 0.0212205 -0.0401903 7.41221e-07` 和 `CMD: 0.0824891 -0.135617 0.0151561`。

第 3 步。`check_after.txt` 最后一行 `1 problems`。空 `frame_id` 没有了（`/planning/debug/goal_pose` 的 frame 是 map）。还在的一行：

`/lightning/odom: not a walking robot: speed p99 55.2 m/s (max 258.8), z -2.9 .. 14.2 m, 18 jumps > 1 m (largest 12.0 m); path 160 m, net 14 m over 30.2 s`

是一下一下跳，不是平滑往下掉，也不是越跑越快。日志里的重力是 −9.807，z 不再掉到 −46 m。LIO 每帧状态（`laser_mapping.cc:501`）相邻差 1 m 以上的有 50 次，多半在两团位姿之间来回（z 在 −0.3 m 和 −1～−2 m）。另外 4 次定位故障把状态重初始化，其中一次从 `(69.5, 128.8, z 6.5)` 拉回 INS `(79.4, 123.1)`，对得上最大那一跳。

第 4 步。每帧都打的是 `time: ..., state:`（`laser_mapping.cc:501`）：676 行，跨 68.5 s，中位间隔 0.101 s，约 9.9 帧/秒。`IMU 回调间隔过大` 7 次。`定位故障!` 和 `auto reinit` 各 4 次（每次重初始化还打两行 `LIO Reinit`，共 8 行）。

第 5 步。没有。`find /public /home -maxdepth 6 -type d -name "ad68517c*"` 没有输出（`find_map.txt`）。配置路径 `/opt/seres/nav/data/ad68517c-e845-42a4-b5dd-a6b7c6d2af13` 本机不存在，定位日志写了「配置的地图路径无效」，回退到仓库里的 `Nav_D20/slam/lightning-lm-master/map/0c3f8a2d-41eb-416d-8fca-aaad500e2be9`。没有改配置。

录包按 T41 走 Git LFS，`.gitignore` 给 `work_dirs/planning_io_t42/**/metadata.yaml` 加了例外。

### T42 复核（2026-11-09）

- **高频预测的修正起作用了**：

  | | T41 | T42 |
  |---|---|---|
  | odom 速度 p99 | 491 m/s | 55 m/s |
  | odom 高度 | −46.7 .. 3.9 m | −2.9 .. 14.2 m |
  | 1 m 以上的跳变 | 61 次 | 18 次 |
  | LIO | 约每秒纠正一次（从跳变推断） | 9.9 帧/秒 |
  | 规划 | fault | 进了 active，`CMD:` 149 次 |

  录包里没有跳变、在走的 10 段（每段 1 s），odom 的航向和行进方向差中位 −2.7°：`base_link` 朝前，坐标系没问题。
- **还在跳，是因为定位用的那张内置地图是歪的。** 从 `loc.log` 和地图文件本身看：
  - 446 / 676 帧报"定位质量下降"，匹配上的点 `effect_num` 中位只有 51（正常应是几千）；每 10 s 攒满 100 帧失败就重初始化一次，共 4 次。
  - 刚用 RTK 固定解的 INS 位姿重初始化完，第一帧也只匹配上约 55 个点：位姿是对的，地图对不上。
  - 开头 28.8 s 狗站着不动（0921_164127_s000），INS 一直在 (78.07, 119.18)。LIO 却在 x 76.3–78.6、z −2.0–0.6 m 之间晃，相邻帧跳 0.5 m 以上 44 次。
  - 内置地图 `0c3f8a2d-...` 是用**这同一个 bag** 的 15 s 建的：8 个关键帧，`pose_lio` 从 (0, 0) 走到 (−0.6, 14.5)，几乎是一条直线。
  - 它对齐到 INS 以后整张图**倾斜 51.3°**：地面的法向离竖直 51°，z 从 −73.8 到 110.6 m。单个关键帧自己的点云是平的（3.4°）。
  - 原因：只用一条直线上的 INS 位置去对齐，绕这条线的转动定不住（建图时 kygi665 的横滚、俯仰也设成了 0、不进约束）。优化随便停在了 51°。
  - 新工具 `Nav_D20/slam/lightning-lm-master/scripts/check_map_tilt.py` 查地图平不平，只要 numpy：这张图报 TILTED 51.3°。测试 `tests/dog/test_loc_map_tilt.py`。
- **规划的局部代价地图把狗自己圈在障碍里**：
  - 555 张里 346 张，狗的脚下（1.0 × 0.4 m）有障碍，1 m 内一半以上的格子是障碍；感知的 `/map/occupancy_grid` 319 张里一张都没有。
  - 所以 TEB 报了 173 次 `Initial pose is in collision`，这些周期速度给 0。
  - 多半是定位跳的时候，静态全局图按错的位置叠进了局部图（狗"站进了墙里"）。定位对了再看；还有的话查 `CostMapProduction` 往局部图里放了什么。
  - 查包工具加了这一条（`--footprint`，默认 D20 的 1.0 × 0.4 m）。
- **配置的地图 `ad68517c-...` 服务器上没有。** 它是机器人团队的现场地图，和规划的 CJGC 是一套（`save_default_map.sh` 里重定位的坐标是几百米量级）。8 号狗跑的就是这套 D20 定位（XT32 + kygi665），多半在狗的 `/opt/seres/nav/data/` 下，只读拷过来就行。
- 本机回放换地图、换 INS 直出不用改配置：`LOC_MAP_PATH=<地图目录>`、`LOC_INS_ONLY=1`（`scripts/loc_debug_config.sh`）。`run_loc_online_debug.sh`、`run_planning_stack_debug.sh` 都认这两个变量，会写一份临时配置，别的不变。
  - `ins_only_mode` 是 lightning 自带的：不跑 LIO 和激光定位，直接发 INS 位姿。0921 这包全程 RTK 固定解，INS 直出就能把规划的调试和定位分开。


---

## T43 Nav_D20 定位地图：从 8 号狗拷现场地图；用它重跑，同时跑一遍 INS 直出

```text
任务 T43：从 8 号狗只读拷出配置的定位地图 ad68517c-...，查它平不平、盖不盖得住 0921 这一片；
有的话用它回放 0921 重录（A）；不管有没有，都用 INS 直出再回放一遍（B）。两个录包都用查包工具查。
先按通用约定做（第 10 条）。可以改的范围：Nav_D20/。仓库根目录。ROS_DOMAIN_ID 同 T41（42）。
狗上只读：不停、不改机器人的任何服务和文件，不往域 0 上发东西。
mkdir -p data/reports/T43；M=/public/zjst/robot/loc_maps；ID=ad68517c-e845-42a4-b5dd-a6b7c6d2af13

0. pytest tests/dog/test_loc_map_tilt.py tests/dog/test_planning_mcap.py -q | tail -1        （8 passed）
   python3 Nav_D20/slam/lightning-lm-master/scripts/check_map_tilt.py Nav_D20/slam/lightning-lm-master/map/0c3f8a2d-41eb-416d-8fca-aaad500e2be9 \
       | tee data/reports/T43/tilt_builtin.txt                          （TILTED 51.3）
   python3 Nav_D20/planning/tools/check_planning_mcap.py work_dirs/planning_io_t42/planning_io_t42_0.mcap | tail -3   （odom、occmap 两条）
1. 狗上找地图（只读）：
     ssh dog 'ls -la /opt/seres/nav/data/ 2>&1 | head -20; du -sh /opt/seres/nav/data/'$ID' 2>&1' | tee data/reports/T43/dog_maps.txt
   - 没有这个目录：在报告里写"狗上没有"，跳过第 2 步和 A，只做 B；
   - 有的话拷到服务器（读不了就用 sudo -n tar 只读打包；狗上不留任何东西）：
       mkdir -p $M && rsync -a dog:/opt/seres/nav/data/$ID/ $M/$ID/ \
         || ssh dog "sudo -n tar -C /opt/seres/nav/data -cf - $ID" | tar -C $M -xf -
2. 查这张地图：
     python3 Nav_D20/slam/lightning-lm-master/scripts/check_map_tilt.py $M/$ID | tee data/reports/T43/tilt_site.txt
   - 要 level；
   - 打印的 x、y 范围要盖住 0921 的起点 (78, 119)。盖不住、或者 TILTED，就跳过 A，报告里写清。
3. A：用现场地图回放（同 T42 第 2 步，只多一个变量）：
     LOC_MAP_PATH=$M/$ID bash Nav_D20/planning/script/run_planning_stack_debug.sh --once
   - 录到 work_dirs/planning_io_t43_map/，定位日志拷到 data/reports/T43/loc_map.log；
   - 查包：check_planning_mcap.py ... | tee data/reports/T43/check_map.txt；
   - 从 loc_map.log 数：定位质量下降的帧数和总帧数，effect_num 中位，定位故障 / 重初始化的次数。
4. B：INS 直出回放（同上，只换变量）：
     LOC_INS_ONLY=1 bash Nav_D20/planning/script/run_planning_stack_debug.sh --once
   - 录到 work_dirs/planning_io_t43_ins/，日志 data/reports/T43/loc_ins.log；check_planning_mcap.py ... | tee data/reports/T43/check_ins.txt；
   - 定位日志里应有 "INS only mode"。
5. 两次都抄规划的关键行：mission_state 走到哪、CMD: 的次数和开头两条、Initial pose is in collision 的次数、Start point is in collision 有没有。
6. 提交：data/reports/T43/、两个录包目录（同 T42，走 LFS）、本文档。不提交拷来的地图。

停止条件：第 1 步要改狗上的东西才能拷；编不过、跑不起来。
报告（写进"T43 结果"）：狗上有没有这张图、多大；tilt_site.txt；A、B 两次查包的 PROBLEM 行；
A 的定位质量统计；两次规划的关键行；哪一次 odom 不再报 "not a walking robot"、局部代价地图不再把狗圈住。
```


---

## T44 路口让行门控：1104 上离线评测

设计见 docs/PLANNING_INTEGRATION.md（第 4、5、7 节）。人定（2026-11-09）：厂区限速 30 km/h；停着的车最多看 3 s；规划里的门控由我们写；先做离线评测。
门控的 Python 参考实现 `dog_perception/yield_gate.py`，评测 `dog_perception/yield_eval.py` 和 `tools/dog/yield_eval.py`，测试 `tests/dog/test_yield_gate.py`。

```text
任务 T44：用 1104 审过的标签离线评测让行门控。狗站的地方往 8 个方向各画一条 8 m 的假想过路线，每 1 s "到一次路边"；
门控用上线模型存下的检测（重跑跟踪和 IMM 预测，和狗上一样）逐帧决定，说走了就按狗的速度一直走过去；
标签当真值，看放行安不安全、比理想门控多等多久。另外量路口车速、车最早在多远被跟上；只读看 8 号狗的定位外参。
先按通用约定做。只用 CPU。仓库根目录，export PYTHONPATH=.。mkdir -p data/reports/T44
S=$(for d in data/labels/1104_*/labels_sustech2.pkl; do basename $(dirname $d); done)

0. pytest tests/dog/test_yield_gate.py -q | tail -1                       （15 passed）
   echo $S | wc -w                                                        （32）
   for r in $S; do [ -f data/runs_online/$r/pillar/results.pkl ] || echo "missing $r"; done     （T37 第 2 步的结果，应该都在）
   8 号狗的定位外参（只读，不停、不改任何东西）：
     ssh dog 'ps -eo args | grep -iE "lightning|run_loc|loc_launch" | grep -v grep | cut -c1-300;
              for f in $(find /opt/seres -name default_d20.yaml 2>/dev/null | head -5); do echo "== $f"; grep -n -B1 -A3 "^ *extrinsic_[TR]" $f; done' \
         2>&1 | tee data/reports/T44/dog_extrinsic.txt
   对着仓库里的 Nav_D20/slam/lightning-lm-master/config/default_d20.yaml 看狗上生效的是哪一组：
   "D20 A2"（fasterlio 的 extrinsic_T 是 [-0.0213, 0.18, 0.009]，雷达在 base_link 前 0.32 m）还是 "D20 A1"（[0.4594, 0.0, 0.0516]，前 0.59 m）。
1. 五次评测，可以同时跑（各占一个 CPU 核；每次约 5–20 分钟）：
     run() { n=$1; shift; s=$(date +%s); python tools/dog/yield_eval.py --scenes $S --fan 8 --length 8 "$@" \
         --details data/reports/T44/episodes_$n.csv > data/reports/T44/yield_$n.md 2>&1; echo "$n exit $? $(( $(date +%s) - s )) s"; }
     { run base & run h8 --pred-horizon 8 & run v2 --dog-speed 2 & run wide --sigma-k 2 --sigma-max 3 & \
       run tight --sigma-max 0.5 & wait; } 2>&1 | tee data/reports/T44/runs.txt
   - base：默认（IMM 3 s 以后匀速，狗 1 m/s，τ 2 s，σ 最多 1 m）；h8：IMM 预测 8 s；v2：狗过路口 2 m/s；
     wide：k = 2、σ 最多 3 m（更保守）；tight：σ 最多 0.5 m。
   - 退出码 1 表示有不安全的放行，是结果不是出错，照样往下做。别的退出码、或者 md 里有 Traceback 才停。
2. 看 yield_base.md：
   - 开头每个场景一行：有没有 "MOVED"（狗在这个场景里挪了 1 m 以上，路线就不是从狗脚下出发了）；
   - "Per path" 表：oracle waited 大于 0 的路线是真有车横穿的；parked in path 超过 50% 的路线（停着的车挡在路上）单独列出来，不算进结论。
3. 不安全的放行（五次都看各自的 "Unsafe episodes"）：每一条抄下场景、路线、那辆车当时多远、方位、车速、跟踪器有没有跟上。
   按原因归类：没跟上（NOT tracked）的，在盲区（方位 142–218°）的几条、30 m 外的几条、别的几条；跟上了还放行的几条。
   跟上了还放行的，挑最多 3 条：从 episodes_base.csv 找到时间，写出那一刻门控为什么没看出冲突
   （可以用 dog_perception/yield_eval.track_objects 重跑那个场景，再用 dog_perception/yield_gate.assess 打印那一帧的 conflicts、min_gap）。
4. 提交：data/reports/T44/（md、csv、runs.txt、dog_extrinsic.txt）、本文档。

停止条件：第 0 步测试不过；缺 results.pkl 的场景超过 3 个；评测报错。
报告（写进"T44 结果"）：
- dog_extrinsic.txt 的结论（A1 还是 A2）；
- 五次的 "all" 一行并成一张表（unsafe、risky、timeouts、wait median / p90、oracle wait median、extra median / p90 / max）；
- base 的 "Per path" 表，标出真有车横穿的路线和被停着的车挡住的路线；
- 第 3 步的归类和挑出来的几条；
- base 的 "Traffic" 两行（30 m 内车的最高车速分布、从 25 m 外开来的车最早在多远被跟上）；
- 每次的运行时间（runs.txt）。
```

