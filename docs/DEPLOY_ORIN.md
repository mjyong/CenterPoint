# Orin 部署：模型转换、量化、C++ 移植和 ROS 2 集成

这份文档写部署到 Jetson AGX Orin 的做法：机器和通道、环境、模型转换和 INT8 量化、每一步怎么验收，以及 C++ / ROS 2 的结构。为什么要上 C++、延迟预算怎么定，见 [DOG_PIPELINE.md](DOG_PIPELINE.md) 第 8 节。服务器上要跑的步骤写在 [SERVER_TASKS.md](SERVER_TASKS.md) 的 T25 起。

## 1. 目标和阶段

Python 流水线是唯一的参考实现。每换一样东西（运行时、精度、语言），都要和它对齐，并在审过的 0921 标签上比最终指标。

| 阶段 | 内容 | 服务器任务 | 状态 |
|---|---|---|---|
| P0 | 环境和通道：服务器经本机的反向隧道连到 Orin，查清两边的版本 | T25 | 完成 |
| P1 | 上线的 Pillar 导出为 ONNX，建 TensorRT FP32 / FP16 引擎，和 PyTorch 比（含原 T8） | T26 | 完成，FP16 可用 |
| P2 | INT8 量化（ModelOpt 显式 Q/DQ），和 FP16 比精度和延迟；动态柱数的 PFN | T27 | 完成：INT8 不上；动态 PFN 上线（`nusc_pp_dyn`，FP16） |
| P3 | C++ 核心库，不依赖 ROS：预处理、检测、跟踪、预测（第 7 节） | — | 完成 |
| P4 | Python 和 C++ 逐段对齐（golden 数据加 GoogleTest），GPU 部分和 TensorRT | T28 | 完成：服务器、板上测试全过 |
| P5 | C++ 全链路离线评估：同一批检测时跟踪对齐，各自检测时 `pipeline_eval.py` 打分 | T28 | 完成：10 个验证场景 ID 每帧相同，标签指标逐项相同 |
| P6 | ROS 2 节点和 `ObjectArray` 消息，bag 回放对比（第 8 节） | T29 | 完成：板上编译过；ROS 读 bag 和消息日志逐位相同；在线回放不丢帧。复核加了时间戳倒退时整个重来 |
| P7 | 板上性能和稳定性 | T28B、T29、T30 | 延迟达标（T28B，4 线程）：端到端 17.3 ms（p99 18.2），网络以外 11.4 ms；节点在样例 bag 上 18.9 ms（T29）。稳定性（长时间、10 Hz 不丢帧、和别的进程抢资源）T30 测 |
| P8 | 车上部署包：`deploy_deb`（车上的接口和名字）和 `deploy/` 对齐，板上打 deb，装上就能跑（第 9 节） | T31 | 完成：板上打出 deb（35 MB），装上后和 `deploy/` 逐位相同，服务起停、卸载都对。复核补了 patchelf、版本号、RPATH；加了 Foxglove 实时看 |

已定（2026-10-28）：
- 板上存储不够，只放 2–3 个验证场景的小子集，全量评估在服务器上做；
- 给规划的消息新定义为 `ObjectArray`；
- C++ 先在服务器 x86 GPU 上开发测试，上板用 colcon 原生编译；
- 先独占实测，资源预算以后定。

## 2. 机器和通道

| 机器 | 地址 | 用途 |
|---|---|---|
| 服务器 | `junyongmou@10.36.136.7`，SSH 端口 1222 | 导出、量化、x86 GPU 上的精度评估、C++ 开发测试 |
| 本机 | 插着板子，网卡 `enx4ce17347042b`，192.168.123.55/24 | 只负责反向隧道 |
| Orin | 192.168.123.100（网卡 eno1），SSH 端口 22，用户 seres，密钥登录 | 建 Orin 引擎、测延迟、终验 |

**Orin 的配置**（T25 实测，`data/reports/T25/orin_env.txt`）：
- NVIDIA Jetson AGX Orin Developer Kit，12 核 Cortex-A78AE，内存 61 GiB；
- Ubuntu 22.04.5，L4T R36.4.4；
- CUDA 12.6、cuDNN 9.3.0、TensorRT 10.3.0、OpenCV 4.10.0、ROS 2 Humble；
- Python 3.10：tensorrt 10.3.0（`~/.local`）、numpy 1.26.4、scipy 1.8.0、torch 2.5（NVIDIA 的版本）；没有 onnxruntime；
- gcc 11.4、cmake 3.22、eigen 3.4、yaml-cpp 0.7、gtest 1.11、colcon 0.3；
- `trtexec`：板上原来的 apt 源没有，T25 从 NVIDIA r36.4 仓库装了 `libnvinfer-bin` 10.3.0.30，在 `/usr/src/tensorrt/bin/trtexec`；
- 功耗模式已是 MAXN；
- 根分区剩余约 5.5 GB（T26 拷完两个场景后 4.7 GB）；
- 系统时钟停在 1970 年（没有校时）：bag 回放用 bag 的时间戳不受影响，上车实时跑之前要人校时；
- `l4tbr0` 上的 192.168.55.1 是 down 的，不要用。

**服务器的配置**（T25 实测，`data/reports/T25/server_*.txt`）：
- 8× RTX 4090，驱动 580.126；系统 CUDA 13.0（另有 11.1），**没有 CUDA 12 工具链**；cuDNN 9.11；
- gcc 10.5、cmake 3.22；没有 eigen、yaml-cpp、gtest、ROS；**没有 sudo**；Docker 27.3.1；
- 根分区剩余约 35 GB；
- 项目环境 `/home/engineers/junyongmou/robot/ultralytics/.venv`：torch 2.13（cu130）、spconv 2.3.8，T25 加了
  tensorrt 10.3.0（pip 的 cu12 版，和板上同版本）、polygraphy 0.53.6、nvidia-modelopt 0.47.0、onnxruntime-gpu 1.22；
- 0921 原始 bag：`/zjst/robot/rosbag0921`（两个 rosbag2，共 35 GB），P6 回放用。

**反向隧道**：在本机执行，断线自动重连：

```bash
autossh -M 0 -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -p 1222 -R 5037:192.168.123.100:22 junyongmou@10.36.136.7
```

服务器的 `~/.ssh/config`：

```text
Host orin
    HostName localhost
    Port 5037
    User seres
```

此后服务器上 `ssh orin`、`rsync -e ssh ... orin:` 都直接到板子。

- sudo 密码不写进仓库，执行时由人给；
- 板上只有 `nvpmodel`、`jetson_clocks` 和 apt 安装需要 sudo。

**同步**：`deploy/scripts/sync_to_orin.sh`。
- 拷代码；
- 拷一个模型目录：只拷 ONNX 和元数据，引擎在板上建；
- 拷几个场景的帧（`--stride`、`--limit` 控制数量）；
- 拷参考检测结果（`--dumps`）。
- 默认目录 `~/dogperc/{repo,models,data}`。
- 板上空间小，用完就删。

## 3. 环境

**Orin**：`deploy/scripts/orin_check.sh > orin_env.txt` 记录版本、功耗模式、内存和磁盘、编译工具。T25 时需要的都已经有，没有装任何包。以后标 `MISSING` 的再装：

```bash
sudo apt install libeigen3-dev libyaml-cpp-dev libgtest-dev python3-colcon-common-extensions
sudo apt install python3-libnvinfer        # 若没有 TensorRT 的 Python 绑定；用来跑 build_engines.py 和 compare_backends.py
```

- 板上没有 det3d：导出的检测器（`dog_perception/detection/exported.py`）只要 numpy、scipy 和 TensorRT。
- 测性能前：功耗模式确认是 MAXN（`nvpmodel -q`）；`sudo jetson_clocks` 锁频，重启后失效。人已加免密 sudo（2026-10-29），Agent 用 `sudo -n jetson_clocks`；还要密码时 `-n` 直接报错，就问人。

**服务器**：
- 在项目环境里加 `tensorrt==10.3.*`，和板上同一个大版本、小版本，保证数值可比；
- 再加 `onnx`、`onnxruntime-gpu`（或 `onnxruntime`）、`polygraphy`、`nvidia-modelopt`；
- 装不进项目环境就另建一个，`torch` 后端那一步回项目环境跑。
- 以上 T25 已装好。`orin_check.sh` 在服务器上要指定项目环境：`PYTHON=<venv>/bin/python deploy/scripts/orin_check.sh`。
- pip 的 TensorRT 是 CUDA 12 编的，`exported.py` 先加载 `libcudart.so.12`（`import tensorrt` 已带进来），找不到再用系统的。

**服务器上的 C++ 开发环境**（T26 定）：服务器没有 CUDA 12 工具链，也没有 sudo，但 Docker 免 sudo 就能用 GPU。
- 用 `nvcr.io/nvidia/tensorrt:24.08-py3`：TensorRT 10.3.0、CUDA 12.6、Ubuntu 22.04，和板子一致，已拉下来；
- Docker 数据目录在 `/zjst`，空间充足；
- 在容器里装 Eigen、yaml-cpp、GoogleTest 和 ROS 2 Humble；
- C++ 全部（含 CUDA、TensorRT）先在服务器的 4090 上开发、测试，再到板上原生编译、终验；
- Python 参考、golden 数据和全量评估都在服务器上做。

## 4. 模型转换（P1）

```text
PyTorch 权重 ──export_pillar_onnx.py──► pfn.onnx、rpn_head.onnx、model_meta.json
                                              │
                    build_engines.py ◄────────┤  (在运行它的机器上：服务器 x86 / Orin)
                    pfn.fp16.engine、rpn_head.fp16.engine
                                              │
     ExportedPillarDetector（onnx | trt）◄────┘  和 CenterPointDetector 同一个接口
```

**两段切分**（`detection/pillar_export.py`）：
- 引擎里只有稠密算子：PFN 是 1×1 卷积，RPN 和 head 是 2D 卷积；
- 体素化、特征装饰、scatter、解码、类别映射、合并 NMS 和几何过滤在引擎外面；
- 它们的 numpy 版是 C++ / CUDA 的参考实现。

**`model_meta.json`**（`detection/model_meta.py`）写下引擎以外需要的一切：
- 网格、每柱点数、P_max；
- 每个 task 的 head 布局和类别、解码设置；
- 到狗的类别映射、合并 NMS 半径；
- 模型系变换。

换成微调的模型时，重新导出即可，代码不用改。

**体素化**用 numpy 重写（`detection/voxelize.py`），和 det3d 的 numba 实现逐位一致（`tests/dog/test_deploy.py`），所以板上不需要 det3d。

命令：

```bash
python tools/dog/export_pillar_onnx.py --checkpoint work_dirs/pretrained/nusc_pp.pth \
    --out data/deploy/nusc_pp --check <某一帧.npy>
python tools/deploy/pillar_stats.py --model-dir data/deploy/nusc_pp --frames "data/runs/*/frames/*.npy" \
    --stride 5 --jobs 8                     # 柱数分布和建议的 P_max；和 30000 差得多就按建议值重导
python tools/deploy/build_engines.py --model-dir data/deploy/nusc_pp --precision fp32,fp16 --benchmark 50
python tools/deploy/compare_backends.py --model-dir data/deploy/nusc_pp \
    --frames "data/runs/<场景>/frames/*.npy" --stride 3 --backends torch,onnx,trt:fp32,trt:fp16 --out <报告>.json
python tools/dog/run_sequence.py --replay data/runs/<场景> --detectors '' \
    --exported trt_fp16=data/deploy/nusc_pp:trt:fp16 --save-raw --no-mcap --out <输出>/{scene}
```

**板上**：
- 拷模型目录和 2–3 个场景的帧，在板上建 FP16 引擎；
- 服务器先用 `compare_backends.py --backends torch --dump <dir>` 存下这些帧上 PyTorch 的检测，拷到板上；
- 板上用 `--backends dump:<dir>/torch,trt:fp16` 对比；
- T26 实测（锁频，GPU 时间中位数）：

  | 引擎 | FP32 | FP16 |
  |---|---|---|
  | pfn（24000 柱） | 5.2 ms | 3.4 ms |
  | rpn_head | 9.1 ms | 4.7 ms |

  Python 检测器在板上的 network ms 是 41–44 ms，多出来的是主机拷贝（PFN 输入 19 MB、BEV 画布 42.6 MB）和 numpy 的 scatter；C++ 版都在 GPU 上。
- 延迟用 `trtexec --loadEngine=<引擎> --iterations=200`，看 GPU Compute Time 的中位数和 p99；
- 或 `tools/deploy/bench_engines.py --model-dir <目录> --engines pfn.fp16 rpn_head.fp16 --runs 200`：
  - 输入只拷一次，CUDA event 计每次执行的 GPU 时间，给中位数和 p99，和 trtexec 的 GPU Compute Time 是同一个量；
  - 服务器的 pip TensorRT 没有 trtexec，就用它；
  - `build_engines.py --benchmark N` 建完引擎顺手测的也是这个。

**动态柱数的 PFN**：
- 静态的 PFN 总按 P_max 算，而 0921 的柱数中位数约 10000，P_max 是 24000；
- `export_pillar_onnx.py --dynamic-pillars` 把 PFN 的柱数维度放开，`model_meta.json` 里记 `onnx.pfn.dynamic`；
- `build_engines.py` 给它一个范围（1 到 P_max），按 `--opt-pillars` 调优；
- 检测器只送实际用到的柱，数学上和补零一样；
- 测延迟要给柱数：`trtexec --shapes=pillars:1x10x10000x20`、`bench_engines.py --shapes pillars:1x10x10000x20`；
- T27：上线用。板上 PFN 的 GPU 时间：10000 柱 1.50 ms，16000 柱 2.37 ms，24000 柱 3.52 ms，和静态（总按 24000 算）的 3.52 ms 一样；
  精度和静态的同一个量级。上线模型目录 `data/deploy/nusc_pp_dyn`（板上 `~/dogperc/models/nusc_pp_dyn`）；
- C++ 的两个检测器两种都支持（`setInputShape`）。

## 5. INT8 量化（P2）

- TensorRT 10 的 INT8 要用自带 scale 的图（Q/DQ 节点），老的隐式校准器已不推荐。
- `tools/deploy/quantize_int8.py` 用 NVIDIA ModelOpt 在真实帧上算 scale，写出 `rpn_head.int8.onnx`；`build_engines.py --precision int8` 再建引擎。
- **校准数据**：
  - 只用**训练**场景的帧，验证场景留给打分；
  - 默认 128 帧，一个 BEV 画布 42.6 MB，128 帧共 5.4 GB 内存。
- **混合精度**：`--exclude ".*/hm/.*" --suffix int8_hmfp16` 让 18 个热力图卷积保持 FP16，生成 `rpn_head.int8_hmfp16.onnx` / `.engine`，后端写 `trt:int8_hmfp16`。
- **PFN**：不做 INT8，保持 FP16。它的输入里既有 ±40 m 的绝对坐标，也有 0.1 m 量级的柱内偏移，整个张量共用一个 scale，偏移就量化没了。它的时间（板上 FP16 3.4 ms，占网络的 42%）靠动态柱数来省。INT8 的后端在没有 INT8 PFN 时自动用 `pfn.fp16.engine`。
- **DLA**：`build_engines.py --dla 0` 把 RPN 放到 DLA，DLA 跑不了的层回退到 GPU。只测、只报告。

**T27 结果：不上。**
- 板上 RPN 的 GPU 时间：FP16 5.05 ms，INT8 2.65 ms（低 47%），热力图保持 FP16 的 2.90 ms；
- 精度两版都不行：AP@2 比 PyTorch 掉 0.124 / 0.099（门槛 0.01），逐框匹配率 0.25–0.35，yaw p99 3.1 rad；
- 有一个非热力图的 head 余弦是 −0.58，两版一样。多半是校准的问题：BEV 画布 95% 以上是 0，entropy 给输入取的截断值太小，柱子的特征被截掉了；
- T28 复核：每种 head 都坏了（余弦 hm 0.61、reg 0.71、dim 0.33，height、rot、vel 为负），是整个主干的问题，不是某个 head；
- 网络合计约 6.6 ms（动态 PFN 加 FP16 RPN），离预算还远，现在不追。P7 时间不够再试：
  - `--method max`；
  - 第一层卷积和各 head 的最后一层不量化；
  - 先用 `compare_backends.py` 每种 head 的余弦（`head_<种类>_cos`）找出坏在哪。

## 6. 验收

| 比什么 | 门槛 |
|---|---|
| TRT FP32 对 PyTorch | 检测几乎完全一致 |
| TRT FP16 对 PyTorch | 上报阈值下，过滤前、过滤后都 ≥ 99% 的框配得上（0.5 m 内同类；阈值翻转不算错，见下）；分数差 p99 < 0.02；中心差 p99 < 0.1 m；`compare_backends.py` 打印 pass |
| TRT FP16 对 PyTorch，标签上 | AP@2 / AP@4 掉 ≤ 0.005；`pipeline_eval.py` 的主要指标差值在 90% 区间之内 |
| INT8 对 FP32 | AP 掉 ≤ 0.01；流水线指标不显著变差；板上延迟比 FP16 降 ≥ 30% 才值得用，否则用 FP16（人定） |
| C++ 对 Python（P4） | 点 1e-4 m；体素集合相同；喂相同检测时跟踪 ID 完全一致、状态 1e-6；预测 1e-4 |
| ROS 2 节点（P6） | `dogperc_bag_replay` 和 `dogperc_replay --messages` 逐位相同，对 Python 节点的 bag 回放同上一行（喂相同检测时 ID 每帧一致）；在线回放 bag 不丢帧（`objects_bag.py`，循环播放时每一遍都不丢） |
| 板上（P7） | 网络以外 < 15 ms，端到端 < 50 ms，p99 < 100 ms，10 Hz 不丢帧 |
| 车上部署包（P8） | 板上 `build_deb.sh` 打出 deb；装上后节点起来，离线回放和 `deploy/` 的 `dogperc_bag_replay` 逐位相同，在线回放不丢帧；服务能起停；卸载干净 |

`compare_backends.py` 的检测比较分两层：
- **raw**：几何过滤之前；
- **final**：过滤之后，也就是跟踪器拿到的。

上报阈值用 `DEFAULT_SCORE_THRESHOLDS`。

上报阈值下漏的、多的框各分五类（T26、T27 复核）：
- near：同一个框，分数差 ≤ 0.02，只是跨过了阈值；
- filter（只在 final）：对面过滤前有这个框（0.5 m 内同类，上报了或分数差 ≤ 0.02），被对面的过滤器去掉了；
- score：同一个框，分数差更大；
- class：0.5 m 内只有别的类的框；
- none：0.5 m 内什么都没有。

门槛用不计 near、filter 的匹配率（`reported_match_tol`），过滤前、过滤后都要过。
- near、filter 都是阈值翻转：框只动了一点，分数跨过了上报阈值，或者过滤器的量（框底离地高度、框里的点数）跨过了它的阈值。框动了多少已经由 score p99、center p99 管着；
- T27 里 FP32 的 onnx 对 torch，分数差 p99 只有 8e-4、中心差 p99 0.6 mm，过滤器照样翻了一个框；
- 不算它们，是为了不把"框挪了几毫米"当成"丢了一个物体"。原来的匹配率照样打印（是 1 − 错的个数 / 上报数，错得多时会是负的）。

逐段误差（PFN 输出、BEV、每个 head）只打印、不卡门槛：
- 整张图上的 max |a − b| / max |ref| 多半由空格子里小的回归输出决定，FP16 到 0.23 时框的中心差还在 5 mm 以内；
- 按 head 种类给最差的余弦（`head_<种类>_cos`），用来定位哪一层出问题；
- 要中间张量：参考是 `torch` 或 `dump:` 时，对第一个导出的后端（通常是 onnx）算。

## 7. C++ 核心库（P3）

```text
deploy/cpp/
  include/dogperc/   头文件，一个 Python 模块对应一个
  src/               CPU 部分；src/cuda/：GPU 体素化（pillars.cu）、TensorRT 引擎、GPU 检测器（DOGPERC_WITH_CUDA）
  tests/             GoogleTest，读 tools/deploy/dump_golden.py 写的 golden 数据，逐段和 Python 比
  tools/dogperc_replay.cpp   序列进，逐帧结果 npz 出（格式同 golden 的 pipeline.npz）
deploy/docker/Dockerfile.dev 服务器上编译测试用的镜像（TensorRT 10.3、CUDA 12.6，同板子）
deploy/ros2/       P6：dog_perception_msgs（Object、PredictedPath、ObjectArray）、dogperc_ros（节点、bag 回放，第 8 节）
```

**模块对应**：

| C++ | Python |
|---|---|
| `geometry`、`pose_buffer` | `geometry.py`、`pose_buffer.py`（PoseBuffer、ImuPropagator） |
| `preprocess` | `preprocess/`：去重、自身滤除、去畸变、多帧累积、检测系、地面估计 |
| `localization` | `inspvax.py`：ENU、航向、杠杆臂、航向检查 |
| `online`、`message_log` | `online.py`、`ros_utils.py`：节点的消息处理（PointCloud2 字节、INSPVAX、IMU、里程计），不依赖 ROS（第 8 节） |
| `detections` | `detection/boxes.py`、`frame_adapter.py` |
| `detection/model_meta`、`pillars`、`decode`、`filters`、`detector` | `model_meta.py`、`voxelize.py`、`pillar_host.py`、`decode.py`、`filters.py`、`exported.py` |
| `detection/trt_engine`、`gpu_detector`（CUDA） | `exported.py` 的 TensorRT 后端 |
| `tracking/imm`、`association`、`tracker` | `tracking/` |
| `prediction/imm_predictor` | `prediction/kinematic.py` |
| `pipeline`、`sequence` | `pipeline.py`、`io.py` |

不移植：第二档的学习预测（不在线），评估和可视化（离线，留在 Python）。

**参数**：只有一个来源，Python 的 dataclass 和节点预设。
- `tools/deploy/export_cpp_config.py` 写成 YAML：`--robot kygi` 是在线节点的配置（含 ROS 参数），`--seq` 是一段录制按 meta.json 的配置；
- C++ 读的时候每个键都必须有，缺了就报错；
- C++ 里的默认值只是照抄，测试检查它们和 Python 的一致。

**数值约定**，为了逐位对得上：
- Python 用 float32 的地方 C++ 也用 float32（点云、体素、head 图），位姿和跟踪用 float64；
- 照搬 NumPy 的类型提升：比如 decorate 里 float32 除以 int32 会变成 float64，`np.radians` 是乘预先算好的 pi/180；
- 编译时关掉 FMA 合并（`-ffp-contract=off`，板子是 aarch64，GCC 默认会合并）；CUDA 核函数用 `__fadd_rn` 这类指定舍入的指令；
- GPU 体素化：点按格子稳定排序，格子按各自第一个点的顺序编号，点在格子里的序号就是它在格内的位置。这样和 CPU 的体素逐位相同，也就和 det3d 相同。

**本机结果**（模拟器序列，26 帧，用真值加噪声的模拟检测 OracleDetector）：
- 体素、PFN 输入、BEV 画布和 Python 逐位相同；
- 解码、NMS、过滤原因完全一致；框差在 float32 的 1 ulp 以内（NumPy 的 exp 和 glibc 的 expf 会差 1 ulp）；
- 整条流水线：点逐位相同，跟踪状态 2e-14，预测 1.5e-10，位姿 8e-16；每帧的跟踪 ID 和顺序都完全一样；
- T28：服务器容器和板上，CPU、GPU、TensorRT 的测试全过；真实录制上的结果见 SERVER_TASKS.md T28。

**编译和测试**：

```bash
# 只要 CPU 部分：Eigen、yaml-cpp、GoogleTest、zlib；golden 数据要 python3 加 numpy、scipy
cmake -S deploy/cpp -B deploy/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/cpp/build -j && ctest --test-dir deploy/cpp/build --output-on-failure
# 加上 CUDA 和 TensorRT：服务器在 dogperc-dev 容器里，板上原生编译
cmake -S deploy/cpp -B deploy/cpp/build_cuda -DDOGPERC_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release
# 板上：-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87
```

- CTest 先跑 `dump_golden.py` 生成 golden 数据（约 60 MB，在 build 目录里），再跑 GoogleTest；
- 设了 `DOGPERC_MODEL_DIR`（导出的模型目录，引擎已在本机建好）时，GPU 检测器还要和"主机拷贝"的 TensorRT 检测器逐帧比。

**GPU 检测器**：
- 点云拷到 GPU，体素化、decorate、scatter 都在 GPU 上；
- 两个引擎直接读写 GPU 缓冲区；
- 只有 head 图拷回主机（约 2.7 MB），解码、NMS、过滤在 CPU 上；
- 每段用 CUDA event 计时：h2d（点云上传）、voxelize（含 decorate）、pfn、scatter、rpn、d2h，写进 `dogperc_replay` 的 `timing_det_*`；
- 预处理的计时：`sweep_ms` = `dedup_ms` + `self_filter_ms` + `deskew_ms`，再加 `accumulate_ms`（多帧拼接和地面估计）。T28 的 `deskew_ms` 是现在的 `sweep_ms`。

**Python 和 C++ 在真实录制上比**（`tools/deploy/cpp_parity.py` 加 `dogperc_replay`）：
1. `cpp_parity.py python`：Python 流水线照 `run_sequence.py` 的方式跑一段录制，存逐帧结果、它用的检测、这次的 C++ 配置；
2. `dogperc_replay --detections`：C++ 用同一批检测。跟踪 ID 每帧必须完全一致，状态 1e-6，预测 1e-4（第 6 节）；
3. `dogperc_replay --model-dir ... --gpu`：C++ 用自己的检测器跑全链路，检测按 compare_backends.py 的办法比；
4. `cpp_parity.py to-results`：转成 `results.pkl`，用 `pipeline_eval.py` 和 Python 并排打分；
5. `cpp_parity.py timings`：每段延迟的中位数、p90、p99。

**板上延迟**（T28，锁频，s000 前 150 帧，中位数）：端到端 27.7 ms（p99 30.2），其中
- 预处理 14.8 ms：一帧扫描 11.9 ms（去重占大头）、拼帧和地面 2.8 ms；
- 检测 12.7 ms：GPU 9.3 ms（体素化含上传 2.3、PFN 0.8、scatter 0.3、RPN 4.8、拷回 1.1），CPU 后处理 3.3 ms；
- 跟踪 0.2 ms，预测 0.1 ms；
- 网络以外 21.8 ms，超过 15 ms 的目标。

**提速**（T28 复核，输出逐位不变；本机 x86，样例 bag 的 27 帧真实 XT32 扫描）：
- 去重：逐点 `std::unordered_map` 换成扁平哈希集合（`src/flat_map.hpp`），6.1 → 2.4 ms；
- 去畸变的时间格稠密排名 1.4 → 0.4 ms；一帧扫描合计 8.4 → 3.9 ms；
- 解码先用 logit 筛格子 0.64 → 0.24 ms；过滤的点网格改 CSR；head 图移动、不拷贝；
- thrust 临时显存用缓存分配器，设备缓冲区按 1.5 倍增长（只编译过）；
- `parallel_for` 线程池（`include/dogperc/parallel.hpp`，`dogperc_replay --threads`），逐点的循环都可以分给几个线程，结果和线程数无关。本机没有收益，板上 T28B 测；
- 预处理本机 12.9 → 8.4 ms。

**板上提速后**（T28B，同样 150 帧，中位数 / p99）：

| | T28 | 1 线程 | 4 线程 |
|---|---|---|---|
| 预处理 | 14.75 / 15.12 | 7.64 / 7.98 | 6.21 / 6.67 |
| 其中去重 / 自身滤除 / 去畸变 / 拼帧 | — | 2.58 / 1.68 / 0.55 / 2.82 | 2.57 / 1.29 / 0.38 / 1.96 |
| 检测的 CPU 后处理 | 3.26 | 1.73 | 1.70 |
| 端到端 | 27.71 / 30.16 | 18.70 / 19.65 | 17.26 / 18.19 |
| 网络以外 | 21.79 / 24.25 | 12.87 / 13.75 | 11.41 / 12.28 |

- 板上用 4 个线程（`dogperc_replay --threads 4`，节点的 `threads` 参数，launch 默认 4）；服务器的共享 vCPU 上 4 线程反而慢，用 1 个；
- 输出和线程数无关，和 T28 逐位相同；
- 剩下较长的：去重 2.6 ms（顺序的哈希，没法分线程）、拼帧 2.0 ms、后处理 1.7 ms、体素化 1.6 ms、head 图拷回 1.1 ms；网络 PFN 0.8 ms 加 RPN 4.7 ms。

**还没做 / 下一步**：
- P6 ROS 2 节点已上板（T29，第 8 节）；长时间回放、负载下的延迟 T30 测，上车实时跑要人（校时、和规划对字段）；
- 性能（P7）：延迟已经达标，下面这些等稳定性测完、确实需要时再做：
  - 把各帧的世界坐标留在 GPU 上、在 GPU 上拼帧；
  - 解码搬到 GPU，省掉 head 图的拷贝（1.1 ms）；
  - INT8 的 RPN（再省 2.4 ms，要先修校准，第 5 节）。
- C++ 统一用 C++17：板上是 gcc 11。

## 8. ROS 2 节点（P6）

```text
deploy/ros2/
  dog_perception_msgs/   Object、PredictedPath、ObjectArray：给规划的消息
  dogperc_ros/
    src/ros_convert.*    ROS 消息和 libdogperc 的结构互转（点云不拷贝）；ObjectArray、MarkerArray、调试点云
    src/node.cpp         dogperc_node：在线节点
    src/bag_replay.cpp   dogperc_bag_replay：rosbag2 直接读进节点的同一套处理，离线、逐帧确定
    launch/dogperc.launch.py
```

**分层**，为了在没有 ROS 的机器上也能测：
- 节点的消息处理在 libdogperc 里，不依赖 ROS（`dogperc/online.hpp`）：
  - PointCloud2 的字节按字段表解析，规则同 Python 的 `ros_utils.scan_from_msg`：xyz 转 float32，丢掉原点和非有限的点，Livox 的噪点 tag，四种时间格式（absolute、absolute_ns、relative、relative_ns）；字段可以不对齐（XT32 的 float64 时间戳在偏移 18）；
  - INSPVAX：第一个好解定 ENU 原点，杠杆臂，航向检查；RTK IMU 的单位；LIO 里程计和 twist；
  - 这些在 Python 那边是 `dog_perception/online.py`，Python 节点 `tools/dog/ros2_node.py` 也改成用它；
- `dogperc_ros` 只把 ROS 消息拷进这些结构，再把结果拷回消息。

**参数**：
- `config`：`tools/deploy/export_cpp_config.py --robot kygi` 写的 YAML，流水线各段加 `node:` 一节（话题、定位方式、时间格式、杠杆臂）。和 Python 节点的 `robot:=kygi` 是同一份预设；
- `model_dir`：导出的模型目录，引擎在这台机器上建（上线是 `nusc_pp_dyn`）；`precision` 默认 fp16，`gpu` 默认 true；
- `threads`：预处理逐点循环的线程数（参数默认 1，launch 默认 4，板上 T28B 定的）；`local_frame`：`~/objects` 的 frame_id，默认 `dog_det`；
- 话题照 config 订阅，要换就用 ROS 的 `-r` 重映射。

**发布**：
- `~/objects`（`dog_perception_msgs/ObjectArray`），给规划：
  - 坐标系是这一帧的检测系：重力对齐，yaw 跟车体，原点在车下的地面上（`PerceptionPipeline::to_local`）；
  - `world_pose` 是这个坐标系在定位系里的位姿，要回到世界系就用它；
  - 每个目标：id、类别、运动状态（静止 / 在动 / 停下）、分数、是否这一帧没检测到（coasting）、年龄、框、速度、[x, y, vx, vy] 的协方差，以及预测：时刻表和每种模式的路径、概率、协方差，最可能的在前；
- `~/tracks`、`~/predictions`（MarkerArray，世界系）：和 Python 节点画的一样；
- `~/cloud`：检测器的输入点云（`node.publish_cloud`）；
- `/tf`（参数 `publish_tf`，默认关）：`camera_init → <local_frame>` 和 `camera_init → <点云的 frame_id>`，给查看器把点云、框和轨迹放到一起（第 9 节"Foxglove"）。车上不开：车上的 TF 树里已经有雷达的 frame。

**线程**：回调都在一个线程里，一个接一个执行，和 Python 节点一样。同时有几条在等时，rclcpp 的执行器按订阅创建的顺序取（IMU、INS / 里程计、点云），不严格按到达顺序，这是在线和离线回放的差别之一。离线时 Python 节点对 bag 的回放（`cpp_parity.py python --bag`）按 bag 的顺序处理，拿来当参考。

**时间戳倒退**（T29 复核）：某个话题的时间戳比它的上一条早 1 s 以上（`kTimeJumpBack`，bag 再播一遍、`--loop`、时钟往回跳），节点整个重来，和新起的一样：位姿、IMU、点云累积、地面估计、跟踪（ID 从 1 开始）、ENU 原点、航向检查都清掉，日志一行 warn。
- 不重来的话，位姿缓存把新来的位姿都当乱序丢掉，bag 长于 3 s 时重播后基本没有输出；
- 重来以后，别的话题还属于旧时间线的迟到消息（循环播放时上一遍的最后一帧点云）直接丢掉，不算第二次倒退；
- 规则在 `OnlineInputs` 里（Python `dog_perception/online.py`，C++ `dogperc/online.hpp`），所以节点、`dogperc_bag_replay`、`dogperc_replay --messages`、`cpp_parity.py python --bag` 都一样。

**怎么验**（T29）：
1. 不用 ROS：`cpp_parity.py python --bag` 用 Python 的消息处理按到达顺序回放 bag，存结果，`--messages` 再把这些消息存成普通数组（`msgs.npz`）。`dogperc_replay --messages` 用 C++ 的消息处理跑同一批消息；
   - 本机：样例 bag 27 帧，点逐位相同，位姿差 9e-16；
   - golden 数据里带样例 bag 的前 8 帧，`Online.MessageLogMatchesPythonBagReplay` 每次都测；另有四种点云字节布局逐点对 Python；
2. 有 ROS（板上）：`dogperc_bag_replay` 从 bag 读消息、反序列化，走同一套处理，结果应和 `dogperc_replay --messages` 逐位相同。这一步验的是 ROS 消息到结构体的拷贝；
3. 在线：板上 `ros2 bag play` 样例 bag，节点照常订阅、发布，看帧数（不丢帧）、`~/objects` 的内容和每帧的耗时。

T29 的结果：
- 第 1 步，服务器上样例 bag 和 0921 前 600 云：喂相同检测时 ID 每帧相同，状态差 2e-10；
- 第 2 步，板上逐位相同；
- 第 3 步，在线 27 帧不丢，每帧 16–20 ms。`ros2 bag play` 要加 `--delay 3`：不加的话，DDS 还没配上对，第一帧就丢了。

长时间（T30）：
- `tools/deploy/cut_bag.py` 从原始 bag 截出节点的三个话题。按源 bag 的 rosbag2 版本写，Humble 能读；
- `deploy/scripts/orin_soak.sh` 在板上 1× 循环播放（每遍节点重来一次），后一半时间加 GPU、CPU 负载，录 `~/objects`、内存、tegrastats；
- `objects_bag.py` 按遍查不丢帧，`tools/deploy/soak_report.py` 按 5 分钟汇总延迟、warn、内存、温度。

```bash
# 板上编译（kygi665_msgs 在仓库根目录）
source /opt/ros/humble/setup.bash
colcon build --base-paths deploy/ros2 kygi665_msgs --build-base deploy/ros2/build --install-base deploy/ros2/install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87
source deploy/ros2/install/setup.bash
ros2 launch dogperc_ros dogperc.launch.py config:=$HOME/dogperc/kygi.yaml model_dir:=$HOME/dogperc/models/nusc_pp_dyn
```

## 9. 车上部署（deploy_deb、deb 包，P8）

`deploy_deb` 是车上真实部署的代码树，按车上仓库的样子组织：
- `perception_combo_ros/common_interface/seres_interface/ros/`：接口包 `common_interface`、`nav2_interface`、`perception_interface`；
- `perception_combo_ros/node_src/lidar_3d_detection/`：节点包；
- `slam/interface/kygi665_msgs`：INSPVAX 消息。

怎么用见那里的 [README](../deploy_deb/perception_combo_ros/node_src/lidar_3d_detection/README.md)。

**和 `deploy/` 的关系**：代码是同一套，只换了名字（命名空间 `lidar3d`、包 `lidar_3d_detection`、消息 `perception_interface/LidarObjectArray`，字段和 `ObjectArray` 相同）。
- 镜像文件：库、测试、离线回放、ROS 消息转换、bag 回放，都是 `deploy/` 对应文件按固定规则改名的结果。改代码只在 `deploy/` 改、在那里测，然后 `python tools/deploy/sync_deploy_deb.py --write` 重写镜像；
- 不带 `--write` 是检查，`test_deploy_deb_mirrors_deploy` 每次跑。检查的内容：
  - 镜像文件逐字等于改名后的 `deploy/`；
  - 配置除了末尾 5 个部署键（`model_dir`、四个话题名），等于 `export_cpp_config.py --robot kygi` 的输出；
  - `LidarObject*` 的字段和 `dog_perception_msgs` 相同；
  - `kygi665_msgs` 的消息相同；
- 节点 `lidar_3d_detection` 是车上自己的写法：glog 日志、`Init()`、话题名在配置里。行为和 `deploy/ros2` 的节点对齐：
  - 订阅的 QoS（IMU 400、INSPVAX 200、点云 SensorData 5）；
  - 每秒一行统计，带各段耗时；
  - launch 默认 4 线程；
  - 时间戳倒退重来（在库里）。

**接过来时改的地方**（2026-10-30）：
- 镜像文件只有注释不同，已按规则重写。`bag_replay` 的 `--preprocess-only` 和每帧一行的输出，反过来加进了 `deploy/ros2`；
- `package.xml` 没声明 `kygi665_msgs`：colcon 可能先编节点，节点就没有 INSPVAX，起来报错退出。已声明；CMake 里改成必需（`LIDAR3D_KYGI`）；
- 和 `deploy/ros2` 不一致的几处：
  - INSPVAX 队列 50 → 200；
  - 统计没有耗时、每 10 帧一行 → 每秒一行带耗时；
  - launch 默认 1 线程 → 4；
  - launch 的 `model_dir` 没有默认值（不传就起不来）→ 空，用配置里的；
- 配置的默认位置：`SERES_PERCEPTION_INSTALL_DIR` 没设时，找包自己的安装位置（ament index），不再是当前目录；
- `orin_soak.sh`：`pkill -f lidar_3d_detection` 会连脚本自己一起杀掉（脚本路径里有这个名字）；`ps -C` 永远匹配不到（进程名截断到 15 个字符）。改成按可执行文件的路径找。`sync_to_orin.sh` 改成同步 `deploy_deb` 本身；
- 测试：`LIDAR3D_TESTS=ON` 加 `LIDAR3D_GOLDEN_DIR`（算法仓库 `deploy/cpp` 编译目录下的 golden）可以在这里编、跑同一套对齐测试。

**deb 包**（`scripts/build_deb.sh`，在板上跑，不用 root）：
1. colcon 编接口包、`kygi665_msgs` 和节点（Release、sm_87、CUDA），`-DLIDAR3D_MODEL_DIR` 把模型装进 `<prefix>/model`：`model_meta.json`、两个 ONNX、两个 FP16 引擎、`engines.json`；
2. 安装目录拷一份，装到 `/opt/seres/perception`：
   - 文本文件里的编译路径（colcon、ament 的 setup 脚本，CMake 配置）改成最终路径；
   - 删掉 `__pycache__`；
   - 二进制里还有编译路径就用 patchelf 改 RPATH，改不了就停；
3. 加上 `/lib/systemd/system/lidar-3d-detection.service`（默认不启用，`--enable-service` 装上就启用）和 `/etc/default/lidar-3d-detection`；
4. Depends：节点和各个库加载的每个动态库，查出由哪个系统包提供，写成 `>=` 板上的版本。两个例外：
   - `libnvinfer10` 写成 `=`，因为引擎只能在建它的 TensorRT 上加载；
   - dlopen 加载的 RMW 单独加进去；
5. `dpkg-deb --root-owner-group` 打包，另写一份报告：板子、模型文件的 sha256、Depends、找不到包的库。

本机演练过打包这一步（`--skip-build`，假的安装目录加一个链接 zlib 的小程序）：
- 路径改写、conffile、systemd 单元的替换都对；
- Depends 自动算出 `libc6`、`zlib1g`；
- `dpkg -i` 装上后，启动脚本按安装位置设好环境、传对参数；`dpkg -r` 留下 conffile，`dpkg -P` 删干净。

colcon 那一步和真机运行要在板上做（T31）。

**T31 的结果**（2026-10-30）：
- 板上打出 35 MB 的 deb，Depends 里有 `libnvinfer10 (= 10.3.0.30-1+cuda12.5)`、`cuda-cudart-12-6`、ROS 包、glog、yaml-cpp；
- 不装、解开跑和装好以后跑，bag 回放都和 `deploy/` 的 `dogperc_bag_replay` 逐位相同；
- 在线 27 帧不丢（先等 DDS 对上）；服务起停、`remove` / `purge` 都对；
- 6 分钟循环 59 遍：没有负载时只有起来时的第一帧丢；有负载时 4 遍各丢 1 帧（同 T30）。

复核时改的：
- **RPATH**：接口包的 Python 扩展库（`lib*__rosidl_generator_py.so`）带着指向编译目录的 RPATH。脚本现在：
  - 开编之前先确认有 patchelf；
  - 把安装目录换成最终位置，指向编译目录的项删掉；
- **源码路径**：编译加 `-ffile-prefix-map`，二进制里不再有源码路径（原来 glog 的 `__FILE__` 带着）；
- **版本号**：板上的代码是 rsync 过去的，没有 `.git`，所有包的版本都是 `0.1.0`，apt 分不出新旧。现在同步脚本写 `GIT_COMMIT`（提交数、提交号），版本是 `0.1.0+<提交数>.g<提交号>`，后打的版本更高；
- **ID 比较**：循环播放时 `same_ids_frames` 很低（59 遍里 426 / 1587 帧），但每遍都在差不多同一帧开始不同。原因是在线的到达顺序和 bag 不完全一样，一帧略有不同，多出一条暂定轨迹，后面的 ID 全部顺延一位。`objects_bag.py` 加了 `same_tracks_frames`：不看编号，按类别和世界系位置比报出来的轨迹；
- **读文件**：`sync_deploy_deb.py`、`soak_report.py`、`cut_bag.py` 读文件都指定 UTF-8（T31 碰到 C locale 读中文注释崩溃）。

**Foxglove 实时看**：板上循环播 bag，连着板子的本机（192.168.123.55，直连板子的 192.168.123.100）用 Foxglove 看检测结果。步骤见 [README](../deploy_deb/perception_combo_ros/node_src/lidar_3d_detection/README.md) 的"用 Foxglove 看"：
- 板上 `scripts/orin_foxglove.sh <bag>`：节点开 `publish_tf`，加 foxglove_bridge（端口 8765），循环播放；
- 本机 Foxglove 连 `ws://192.168.123.100:8765`，导入布局 `foxglove/lidar_3d_detection.json`：
  - 3D：原始点云、框、ID、预测路径；
  - Raw Messages：`LidarObjectArray`；
  - 目标数曲线和日志；
- `tools/deploy/foxglove_probe.py` 在板上不开 GUI 也能查 bridge 发了哪些话题、各收到多少条；`objects_bag.py` 录到 `/tf` 时，会核对 TF 和消息里的 `world_pose`。

## 10. 文件一览

| 文件 | 作用 |
|---|---|
| `tools/dog/export_pillar_onnx.py` | 导出两段 ONNX 和 `model_meta.json`，`--check` 与 PyTorch 逐段比 |
| `dog_perception/detection/model_meta.py` | 元数据的读写；网络之后的后处理（PyTorch 和导出的检测器共用） |
| `dog_perception/detection/voxelize.py` | numpy 体素化，和 det3d 一致 |
| `dog_perception/detection/exported.py` | `ExportedPillarDetector`：ONNX Runtime 或 TensorRT 引擎，接口同 `CenterPointDetector` |
| `tools/deploy/build_engines.py` | 建 FP32 / FP16 / INT8（Q/DQ）/ DLA 引擎，写 `engines.json` |
| `tools/deploy/bench_engines.py` | 已建引擎的 GPU 时间（中位数、p99），没有 trtexec 的机器用它 |
| `tools/deploy/quantize_int8.py` | ModelOpt INT8 PTQ，写 `<graph>.int8*.onnx` |
| `tools/deploy/compare_backends.py` | 多个后端逐段、逐框比较，查门槛；`--dump` / `dump:` 跨机器比较 |
| `tools/deploy/pillar_stats.py` | 柱数分布，定 P_max |
| `tools/dog/run_sequence.py --exported` | 导出的检测器跑完整流水线，结果给 `pipeline_eval.py` |
| `deploy/cpp/` | C++ 核心库 libdogperc、GoogleTest 对齐测试、`dogperc_replay`（第 7 节） |
| `tools/deploy/export_cpp_config.py` | 从 Python 的 dataclass 写 C++ 的参数 YAML |
| `tools/deploy/dump_golden.py` | C++ 对齐测试用的 golden 数据（模拟器序列加每一段的 Python 输出） |
| `tools/deploy/cpp_parity.py` | 真实录制上 Python 和 C++ 比：参考运行、比较、转 results.pkl、延迟汇总 |
| `deploy/docker/Dockerfile.dev` | 服务器上编译测试 C++ 的镜像 |
| `deploy/ros2/` | ROS 2 消息包和节点包（第 8 节） |
| `dog_perception/online.py` | 节点的消息处理，不依赖 ROS；Python 节点和 `cpp_parity.py python --bag` 共用 |
| `tools/deploy/message_log.py` | 读 bag（到达顺序），把节点的输入消息存成普通数组，给 `dogperc_replay --messages` |
| `tools/deploy/objects_bag.py` | 读录下的 `~/objects`，按遍和回放比：不丢帧、时间戳、ID |
| `tools/deploy/cut_bag.py` | 原始 bag 截成节点的三个话题、前 N 云，保持 rosbag2 版本（板上 Humble 能播） |
| `tools/deploy/soak_report.py` | 长时间回放的汇总：每 5 分钟的延迟、warn 计数、内存、tegrastats |
| `deploy/scripts/orin_check.sh` | 环境报告 |
| `deploy/scripts/sync_to_orin.sh` | 代码、模型（不带 `engines.json`）、帧、参考检测、原始录制、bag 同步到板上 |
| `deploy/scripts/orin_soak.sh` | 板上长时间回放：节点、循环播放、录包、负载、内存和 tegrastats（T30） |
| `deploy_deb/` | 车上部署的代码树（第 9 节）；`node_src/lidar_3d_detection/scripts/build_deb.sh` 在板上打 deb |
| `tools/deploy/sync_deploy_deb.py` | `deploy_deb` 的镜像文件按 `deploy/` 重写（`--write`）或检查；配置、消息字段一起查 |
| `deploy_deb/.../scripts/orin_foxglove.sh`、`foxglove/lidar_3d_detection.json` | 板上循环播 bag 加 foxglove_bridge；本机 Foxglove 的布局 |
| `tools/deploy/foxglove_probe.py` | 不开 GUI 查 foxglove_bridge：发了哪些话题、订几秒各收到多少条（只用标准库，板上能跑） |
