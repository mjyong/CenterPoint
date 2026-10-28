# Orin 部署：模型转换、量化、C++ 移植和 ROS 2 集成

这份文档写部署到 Jetson AGX Orin 的做法：机器和通道、环境、模型转换和 INT8 量化、每一步怎么验收，以及 C++ / ROS 2 的结构。为什么要上 C++、延迟预算怎么定，见 [DOG_PIPELINE.md](DOG_PIPELINE.md) 第 8 节。服务器上要跑的步骤写在 [SERVER_TASKS.md](SERVER_TASKS.md) 的 T25 起。

## 1. 目标和阶段

Python 流水线是唯一的参考实现。每换一样东西（运行时、精度、语言），都要和它对齐，并在审过的 0921 标签上比最终指标。

| 阶段 | 内容 | 服务器任务 | 状态 |
|---|---|---|---|
| P0 | 环境和通道：服务器经本机的反向隧道连到 Orin，查清两边的版本 | T25 | 待做 |
| P1 | 上线的 Pillar 导出为 ONNX，建 TensorRT FP32 / FP16 引擎，和 PyTorch 比（含原 T8） | T26 | 待做 |
| P2 | INT8 量化（ModelOpt 显式 Q/DQ），和 FP16 比精度和延迟 | T27 | 待做 |
| P3 | C++ 核心库，不依赖 ROS：预处理、检测、跟踪、预测 | — | 写代码中 |
| P4 | Python 和 C++ 逐段对齐（golden 数据加 GoogleTest） | T28 | |
| P5 | C++ 全链路离线评估（`pipeline_eval.py`） | T29 | |
| P6 | ROS 2 节点和 `ObjectArray` 消息，bag 回放对比 | T30 | |
| P7 | 板上性能和稳定性 | T31 | |

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
- 根分区剩余约 5.5 GB；
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

**服务器上的 C++ 开发环境**：服务器没有 CUDA 12 工具链，也没有 sudo。T26 第 0 步检查 Docker 能不能免 sudo 用 GPU，再定：
- **能用**：用 `nvcr.io/nvidia/tensorrt:24.08-py3`（TensorRT 10.3.0、CUDA 12.6，Ubuntu 22.04，和板子一致，约 9 GB），
  在里面装 Eigen、yaml-cpp、GoogleTest 和 ROS 2 Humble。C++ 全部（含 CUDA、TensorRT）先在服务器的 4090 上开发、测试，再到板上原生编译、终验。
- **不能用**：服务器只编 C++ 的 CPU 部分（Eigen、yaml-cpp、GoogleTest 用 CMake FetchContent 拉，不要 sudo）；
  CUDA 和 TensorRT 部分直接在板上编、测，板上依赖都齐。
- 两种情况下，Python 参考、golden 数据和全量评估都在服务器上做。

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
- 延迟用 `trtexec --loadEngine=<引擎> --iterations=200`，看 GPU Compute Time 的中位数和 p99；
- 或 `tools/deploy/bench_engines.py --model-dir <目录> --engines pfn.fp16 rpn_head.fp16 --runs 200`：
  - 输入只拷一次，CUDA event 计每次执行的 GPU 时间，给中位数和 p99，和 trtexec 的 GPU Compute Time 是同一个量；
  - 服务器的 pip TensorRT 没有 trtexec，就用它；
  - `build_engines.py --benchmark N` 建完引擎顺手测的也是这个。

## 5. INT8 量化（P2）

- TensorRT 10 的 INT8 要用自带 scale 的图（Q/DQ 节点），老的隐式校准器已不推荐。
- `tools/deploy/quantize_int8.py` 用 NVIDIA ModelOpt 在真实帧上算 scale，写出 `rpn_head.int8.onnx`；`build_engines.py --precision int8` 再建引擎。
- **校准数据**：
  - 只用**训练**场景的帧，验证场景留给打分；
  - 默认 128 帧，一个 BEV 画布 42.6 MB，128 帧共 5.4 GB 内存。
- **混合精度**：`--exclude ".*/hm/.*" --suffix int8_hmfp16` 让 18 个热力图卷积保持 FP16，生成 `rpn_head.int8_hmfp16.onnx` / `.engine`，后端写 `trt:int8_hmfp16`。
- **PFN**：计算量小，默认保持 FP16。INT8 的后端在没有 INT8 PFN 时自动用 `pfn.fp16.engine`。
- **DLA**：`build_engines.py --dla 0` 把 RPN 放到 DLA，DLA 跑不了的层回退到 GPU。只测、只报告。

## 6. 验收

| 比什么 | 门槛 |
|---|---|
| TRT FP32 对 PyTorch | head 图相对误差 < 1e-3；检测几乎完全一致 |
| TRT FP16 对 PyTorch | head 图相对误差 < 1e-2；上报阈值下 ≥ 99% 的框配得上（0.5 m 内同类）；分数差 p99 < 0.02；中心差 p99 < 0.1 m；`compare_backends.py` 打印 pass |
| TRT FP16 对 PyTorch，标签上 | AP@2 / AP@4 掉 ≤ 0.005；`pipeline_eval.py` 的主要指标差值在 90% 区间之内 |
| INT8 对 FP32 | AP 掉 ≤ 0.01；流水线指标不显著变差；板上延迟比 FP16 降 ≥ 30% 才值得用，否则用 FP16（人定） |
| C++ 对 Python（P4） | 点 1e-4 m；体素集合相同；喂相同检测时跟踪 ID 完全一致、状态 1e-6；预测 1e-4 |
| 板上（P7） | 网络以外 < 15 ms，端到端 < 50 ms，p99 < 100 ms，10 Hz 不丢帧 |

`compare_backends.py` 的检测比较分两层：
- **raw**：几何过滤之前；
- **final**：过滤之后，也就是跟踪器拿到的。

上报阈值用 `DEFAULT_SCORE_THRESHOLDS`。

## 7. C++ 和 ROS 2（P3–P6，写代码中）

```text
deploy/
  cpp/            不依赖 ROS 的核心库 libdogperc：loc、preprocess、detect（CUDA + TensorRT）、track、predict、io；
                  tests/（GoogleTest，读 golden npz）；tools/dogperc_replay（序列进，逐帧结果 npz 出）
  ros2/           dog_perception_msgs（Object、PredictedPath、ObjectArray）、dog_perception_ros（rclcpp component）
  config/         tools/deploy/export_cpp_config.py 从 Python 的 dataclass 生成的 YAML，C++ 不硬编码参数
  scripts/        orin_check.sh、sync_to_orin.sh
```

- 每个 C++ 模块对应一个 Python 模块；第二档预测不在线，不移植。
- 参数只有一个来源：Python 的 dataclass 和节点预设。
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
| `deploy/scripts/orin_check.sh` | 环境报告 |
| `deploy/scripts/sync_to_orin.sh` | 代码、模型、帧、参考检测同步到板上 |
