# Orin 部署：模型转换、量化、C++ 移植和 ROS 2 集成

这份文档写部署到 Jetson AGX Orin 的做法：机器和通道、环境、模型转换和 INT8 量化、每一步怎么验收，以及 C++ / ROS 2 的结构。为什么要上 C++、延迟预算怎么定，见 [DOG_PIPELINE.md](DOG_PIPELINE.md) 第 8 节。服务器上要跑的步骤写在 [SERVER_TASKS.md](SERVER_TASKS.md) 的 T25 起。

## 1. 目标和阶段

Python 流水线是唯一的参考实现。每换一样东西（运行时、精度、语言），都要和它对齐，并在审过的 0921 标签上比最终指标。

| 阶段 | 内容 | 服务器任务 | 状态 |
|---|---|---|---|
| P0 | 环境和通道：服务器经本机的反向隧道连到 Orin，查清两边的版本 | T25 | 完成 |
| P1 | 上线的 Pillar 导出为 ONNX，建 TensorRT FP32 / FP16 引擎，和 PyTorch 比（含原 T8） | T26 | 完成，FP16 可用 |
| P2 | INT8 量化（ModelOpt 显式 Q/DQ），和 FP16 比精度和延迟；动态柱数的 PFN | T27 | 完成：INT8 不上；动态 PFN 上线（`nusc_pp_dyn`，FP16） |
| P3 | C++ 核心库，不依赖 ROS：预处理、检测、跟踪、预测（第 7 节） | — | 已写，本机 CPU 部分全部对上 |
| P4 | Python 和 C++ 逐段对齐（golden 数据加 GoogleTest），GPU 部分和 TensorRT | T28 | 待做 |
| P5 | C++ 全链路离线评估：同一批检测时跟踪对齐，各自检测时 `pipeline_eval.py` 打分 | T28 | 待做 |
| P6 | ROS 2 节点和 `ObjectArray` 消息，bag 回放对比 | T29 | |
| P7 | 板上性能和稳定性 | T30 | |

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
- 测性能前：功耗模式确认是 MAXN（`nvpmodel -q`）；`sudo jetson_clocks` 锁频，重启后失效，由人在板上执行。

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
| 板上（P7） | 网络以外 < 15 ms，端到端 < 50 ms，p99 < 100 ms，10 Hz 不丢帧 |

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
deploy/ros2/       P6：dog_perception_msgs（Object、PredictedPath、ObjectArray）、rclcpp 节点（还没写）
```

**模块对应**：

| C++ | Python |
|---|---|
| `geometry`、`pose_buffer` | `geometry.py`、`pose_buffer.py`（PoseBuffer、ImuPropagator） |
| `preprocess` | `preprocess/`：去重、自身滤除、去畸变、多帧累积、检测系、地面估计 |
| `localization` | `inspvax.py`：ENU、航向、杠杆臂、航向检查（ROS 消息解析留给 P6 节点） |
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
- 整条流水线：点最大差 9.3e-10 m，跟踪状态 2e-14，预测 1.5e-10，位姿 8e-16；每帧的跟踪 ID 和顺序都完全一样；
- CUDA 部分在本机只编译过（CUDA 12.0，`sm_87` / `sm_89`），没有 GPU、也没有 TensorRT 头文件，GPU 测试和 TensorRT 都留给 T28。

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
- 每段用 CUDA event 计时：voxelize、pfn、scatter、rpn、d2h，写进 `dogperc_replay` 的 `timing_*`。

**Python 和 C++ 在真实录制上比**（`tools/deploy/cpp_parity.py` 加 `dogperc_replay`）：
1. `cpp_parity.py python`：Python 流水线照 `run_sequence.py` 的方式跑一段录制，存逐帧结果、它用的检测、这次的 C++ 配置；
2. `dogperc_replay --detections`：C++ 用同一批检测。跟踪 ID 每帧必须完全一致，状态 1e-6，预测 1e-4（第 6 节）；
3. `dogperc_replay --model-dir ... --gpu`：C++ 用自己的检测器跑全链路，检测按 compare_backends.py 的办法比；
4. `cpp_parity.py to-results`：转成 `results.pkl`，用 `pipeline_eval.py` 和 Python 并排打分；
5. `cpp_parity.py timings`：每段延迟的中位数、p90、p99。

**还没做 / 下一步**：
- P6 ROS 2 节点：消息、rclcpp 组件、INSPVAX 和点云消息解析，bag 回放对比；
- 性能（P7）：
  - CPU 预处理在本机 x86 约 12 ms 一帧（去畸变 7 ms，累积和地面 4 ms，约 20 万点），板上 A78 会更慢；
  - 要优化，或把去畸变和累积搬到 GPU；
  - 解码搬到 GPU，可以省掉 head 图的拷贝；
  - 用 cub 并预分配临时内存，代替 thrust 每帧的临时分配；用锁页内存。
- C++ 统一用 C++17：板上是 gcc 11。

## 8. 文件一览

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
| `deploy/scripts/orin_check.sh` | 环境报告 |
| `deploy/scripts/sync_to_orin.sh` | 代码、模型、帧、参考检测同步到板上 |
