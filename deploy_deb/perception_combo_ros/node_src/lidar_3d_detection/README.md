# lidar_3d_detection

车上部署的激光感知节点：点云预处理 → 3D 检测（CenterPoint Pillar，TensorRT FP16）→ 跟踪 → 预测，发布给规划。在 Jetson AGX Orin 上编译、打成 deb 包，装上就能跑。

C++ 代码只有这一份：算法仓库（CenterPoint 仓库）的 Python 是参考实现，本包是它的 C++ 版。不带 ROS 也能编，在算法仓库里跑逐段对 Python 的测试（见"和算法仓库对齐"）；bag 回放的逐位比较、板上的延迟和长时间测试见算法仓库的 `docs/DEPLOY_ORIN.md`。

## 目录

```text
perception_combo_ros/
  common_interface/seres_interface/ros/   common_interface、nav2_interface、perception_interface（LidarObjectArray）
  node_src/lidar_3d_detection/
    include/lidar3d/, src/                库 lidar3d：预处理、检测、跟踪、预测，不依赖 ROS
    tests/                                对 Python 的逐段测试（GoogleTest，见"和算法仓库对齐"）
    tools/lidar_3d_replay.cpp             离线回放（录好的序列或消息日志，不要 ROS）
    src/ros_convert.cpp, src/bag_replay.cpp, include/ros_convert.hpp     ROS 消息转换、bag 回放
    src/lidar_3d_detection.cpp, src/main.cpp, include/lidar_3d_detection.h, include/logging.h   节点
    config/lidar_3d_detection.yaml        参数：算法仓库导出，末尾 5 个部署键（model_dir、四个话题名）是本包自己的
    docker/Dockerfile.dev                 服务器上编译、测试用的镜像（TensorRT 10.3、CUDA 12.6，和板上一样）
    launch/                               ros2 launch
    packaging/                            deb 里的启动脚本、systemd 单元、默认设置、安装脚本
    scripts/build_deb.sh                  在板上打 deb
    scripts/orin_soak.sh                  长时间循环回放（装好的节点）
    scripts/orin_foxglove.sh              板上循环播 bag，用 Foxglove 实时看（见"用 Foxglove 看"）
    foxglove/lidar_3d_detection.json      Foxglove 的布局
    scripts/sync_to_orin.sh, scripts/orin_check.sh, scripts/orin_clocks.sh（锁频了没有）
slam/interface/kygi665_msgs/              RTK 组合导航的 INSPVAX（定位）
```

## 输入、输出

| | 话题 | 类型 |
|---|---|---|
| 输入 | `/robot/topic/rdap/top_point_cloud2` | `sensor_msgs/PointCloud2`（XT32） |
| 输入 | `/rte/rdap/rtk/imus/data_raw` | `sensor_msgs/Imu`（deg/s，kygi 约定） |
| 输入 | `/rte/rdap/rtk/inspvax` | `kygi665_msgs/INSPVAX`（第一个好解定 ENU 原点） |
| 输出 | `/perception/lidar_3d/objects` | `perception_interface/LidarObjectArray`，给规划 |
| 输出 | `/perception/lidar_3d/tracks`、`/perception/lidar_3d/predictions` | `visualization_msgs/MarkerArray`（世界系，调试用） |

- 话题名都在配置的 `node:` 一节里。
- `LidarObjectArray` 的坐标系是这一帧的检测系（frame_id `lidar_det`）：重力对齐，航向跟着车体，原点在车体正下方的地面上。`world_pose` 是它在定位系（`camera_init`）里的位姿。字段说明见消息文件。
- 某个话题的时间戳倒退 1 s 以上（bag 再播一遍、时钟回跳），节点会整个重来，日志里有一行 warn。
- 参数 `publish_cloud`：发 `/perception/lidar_3d/cloud`（检测器的输入点云，检测系），默认取配置里的 `node.publish_cloud`（false）；
- 参数 `publish_tf`（默认关）：发 TF `camera_init → lidar_det` 和 `camera_init → hesai_lidar`（点云自己的 frame_id，取点云时间戳那一刻的位姿）。查看器靠它把原始点云、检测框和轨迹放进同一个 3D 视图。车上不要开：车上自己的 TF 树里已经有雷达的 frame，再发就有两个父节点。

## 在板上打 deb

前置：JetPack 6（L4T R36）、ROS 2 Humble（`/opt/ros/humble`）、colcon、CUDA、TensorRT、patchelf（`sudo apt install patchelf`：接口包的 Python 扩展库带着指向编译目录的 RPATH，要改掉），以及**在这块板上建好的引擎**。引擎只能在建它的 TensorRT 上加载，所以 deb 依赖的 `libnvinfer10` 版本和板上完全一致。

```bash
# 模型目录：model_meta.json、pfn.onnx、rpn_head.onnx、pfn.fp16.engine、rpn_head.fp16.engine
# （引擎用算法仓库的 tools/deploy/build_engines.py --precision fp16 --opt-pillars 10000 在板上建）
cd perception_combo_ros
node_src/lidar_3d_detection/scripts/build_deb.sh --model ~/dogperc/models/nusc_pp_dyn --out ~/deb
```

- **编译**：colcon 编 `--packages-up-to lidar_3d_detection`，即接口包、`kygi665_msgs` 和本包。Release、sm_87、CUDA 打开。第一次约 10 分钟，以后增量编译；
- **改路径**：安装目录拷一份，装到最终位置 `/opt/seres/perception`。文本文件里的编译路径改成最终路径，二进制里还有编译路径就停下；
- **依赖**：每个库由哪个系统包提供（`dpkg -S`），写进 Depends；
- **输出**：
  - `lidar-3d-detection_<版本>_arm64.deb`，版本是 `package.xml` 的版本加提交数和提交号，例如 `0.1.0+412.g1a2b3c4`。提交数在前，后打的包版本更高，apt 能直接升级。板上的代码是 rsync 过去的，没有 `.git`，这时从同步脚本写的 `GIT_COMMIT` 文件读；
  - 同名 `.txt` 报告：板子、模型文件的 sha256、Depends，以及找不到包的库；
- **选项**：
  - `--prefix`：安装位置；
  - `--service-user`：服务用哪个用户跑，默认是打包的人；
  - `--no-enable-service`：只装不启用。默认装上就开机自启并启动；
  - `--skip-build`：直接打包已有的安装目录；
- 不需要 root。

## 安装、运行

```bash
sudo apt install ./lidar-3d-detection_*_arm64.deb      # 或 sudo dpkg -i
# 装上就启用并启动了服务（开机自启；打包时 --no-enable-service 就只装不启用）：
systemctl status lidar-3d-detection
journalctl -u lidar-3d-detection -f            # 输出；glog 文件在 /tmp/lidar_3d_detection_logs
# 手动跑（前台），先停服务，不然两个节点抢 GPU：
sudo systemctl stop lidar-3d-detection
/opt/seres/perception/lidar_3d_detection/bin/start_lidar_3d_detection.sh
```

- 升级（装新的 deb）时先停服务，装完只要服务是启用的就再起来，不管是装的时候启用的还是人手动启用的；
- `scripts/orin_foxglove.sh`、`scripts/orin_soak.sh` 会自己停掉正在跑的服务（`sudo -n systemctl stop`），结束时再起来；没有免密 sudo 时会提示先手动停；

- 装到 `/opt/seres/perception`：四个接口包、本包、模型（`lidar_3d_detection/model`）、配置（`lidar_3d_detection/config`）；
- 另外两个文件：
  - `/lib/systemd/system/lidar-3d-detection.service`；
  - `/etc/default/lidar-3d-detection`：`ROS_DOMAIN_ID`、`RMW_IMPLEMENTATION`、`LIDAR3D_THREADS`（默认 4）、`LIDAR3D_CONFIG`、`LIDAR3D_MODEL_DIR`、`LIDAR3D_LOG_DIR`、`LIDAR3D_ARGS`（更多节点参数，比如 `-p publish_tf:=true -p publish_cloud:=true` 让服务本身也能给查看器看）；
- 配置和 `/etc/default` 那个文件是 conffile：升级时保留你改过的；`dpkg -r` 卸载时也留着，`dpkg -P` 才删掉；
- 起来以后日志里有一行 "lidar_3d_detection up: ..."，之后每秒一行：框数、轨迹数、各段耗时（ms），最后是这一秒里最长的回调（`callback_max_ms`，含发布）和 `clouds N frames N skipped N`（收到几帧点云、发出几帧结果、按点云时间戳算中间缺了几帧）；
- 延迟要在锁频后看：`sudo jetson_clocks`，重启就没了。`scripts/orin_clocks.sh` 查锁没锁。没锁时节点慢 2–3 倍（T32：每帧 55 ms，锁频时 15–21 ms）；
- 不经过 deb，直接用 colcon 的安装目录也行：`source install/setup.bash && ros2 launch lidar_3d_detection lidar_3d_detection.launch.py`。

## 用 Foxglove 看（板上循环播 bag）

板上跑节点和 foxglove_bridge（WebSocket），连着板子的本机用 Foxglove 连上去看。本机和板子直连：本机 192.168.123.55，板子 192.168.123.100。

1. 板上装一次 bridge：`sudo apt install ros-humble-foxglove-bridge`；
2. 板上（从本机 `ssh seres@192.168.123.100` 进去）：
   ```bash
   ~/dogperc/repo/deploy_deb/perception_combo_ros/node_src/lidar_3d_detection/scripts/orin_foxglove.sh bags/<bag>
   ```
   这个脚本：
   - 服务在跑就先停掉，结束时再起来；
   - 起装好的节点，打开 `publish_tf` 和 `publish_cloud`（检测器的输入点云 `/perception/lidar_3d/cloud`，布局里画的是它）；
   - 起 bridge，端口 8765；
   - 1× 循环播 bag，第二个参数可以改倍速；
   - `ROS_DOMAIN_ID` 默认 77；
   - Ctrl-C 全部停掉。

3. 本机打开 Foxglove 桌面版：
   - Open connection → Foxglove WebSocket → `ws://192.168.123.100:8765`；
   - Layout → Import from file → `foxglove/lidar_3d_detection.json`。

   布局里有：
   - **3D**：显示系是 `camera_init`（世界系，机器人在里面走）；检测器的输入点云（5 帧拼起来的，按强度上色）、轨迹的框、ID、速度、预测路径；
   - **Raw Messages**：`/perception/lidar_3d/objects` 的内容，即给规划的 `LidarObjectArray`。

   想看原始点云（`/robot/topic/rdap/top_point_cloud2`，单帧，靠 `camera_init → hesai_lidar` 那条 TF），在 3D 面板的话题列表里勾上就行；

- bridge 3.x（板上是 3.6）基于 Foxglove SDK，握手只认 `foxglove.sdk.v1`，新版 Foxglove 桌面版支持。开源的 Lichtblick 能不能连 3.x 没验证过，连不上就用 Foxglove 桌面版；
- 连不上 8765（防火墙），就在本机开端口转发 `ssh -N -L 8765:localhost:8765 seres@192.168.123.100`，再连 `ws://localhost:8765`；
- 点云流量：原始点云每帧约 3.3 MB，检测器的输入点云是 5 帧拼起来的，每帧几 MB，10 Hz 下都要几十 MB/s。直连的网口一般够用，卡的话在 3D 面板里关掉点云，只看框；
- 检测话题明显不到 10 Hz（Foxglove 里框一卡一卡的）：看 `/tmp/lidar3d_foxglove/node.log` 每秒那一行。`callback_max_ms` 接近 100、`skipped` 跟着涨，是节点忙不过来（先锁频，再看要不要关掉 `publish_cloud`）；`callback_max_ms` 不大而 `skipped` 多，是点云在 DDS 里就丢了（几 MB 的点云走 UDP，T32 加大 Fast DDS 的 socket 缓冲后收齐）；
- 每遍循环开头，节点会重来（日志 warn "jumped back"），时间戳往回跳，Foxglove 里可能闪一下；
- 不起 GUI 也能查 bridge：`python3 ~/dogperc/repo/tools/deploy/foxglove_probe.py`，列出 bridge 发布的话题，订几秒、数消息条数（Foxglove 会看的那几个话题都要有）。

## 和算法仓库对齐

- 改 C++ 就在这里改（算法仓库原来的 `deploy/` 已经删掉，不再有镜像）。库 `lidar3d` 不依赖 ROS，在算法仓库里不带 ROS 编，跑对 Python 的测试：
  ```bash
  # CenterPoint 仓库根目录；L=deploy_deb/perception_combo_ros/node_src/lidar_3d_detection
  cmake -S $L -B build/lidar3d -G Ninja -DLIDAR3D_ROS=OFF -DLIDAR3D_TESTS=ON     # GPU 部分再加 -DLIDAR3D_WITH_CUDA=ON
  cmake --build build/lidar3d -j && ctest --test-dir build/lidar3d --output-on-failure
  ```
  - `golden`：先用算法仓库的 `tools/deploy/dump_golden.py` 生成黄金数据（同样的输入，Python 每一段的输出，写在 `build/lidar3d/golden`），要 python3、numpy、scipy；
  - `parity`：GoogleTest 逐段和黄金数据比（预处理、体素化、解码、跟踪、预测……），再用 `lidar_3d_replay` 回放黄金序列，和 Python 的整条流水线比；
  - 设了环境变量 `LIDAR3D_MODEL_DIR`（导出的模型目录，引擎已在本机建好）时，GPU 检测器还要和主机拷贝的 TensorRT 检测器逐帧比；
  - 不在算法仓库里（只有 deploy_deb）时用 `-DLIDAR3D_GOLDEN_DIR=<目录>` 指向生成好的黄金数据；
  - 服务器上用 `docker/Dockerfile.dev` 的镜像编，命令在文件开头；
- 参数和消息：`python tools/deploy/check_deploy_deb.py`，算法仓库的测试 `test_deploy_deb_config_follows_python` 每次都跑。检查两项：
  - 配置：除末尾 5 个部署键外，等于 `export_cpp_config.py --robot kygi` 的输出（参数的唯一来源是 Python 的 dataclass）。改了 Python 的参数后用 `--write` 重写；
  - `kygi665_msgs` 的消息和算法仓库的 `kygi665_msgs` 一致；
- 算法仓库的离线工具直接读本包的输出：`cpp_parity.py` 读回放的 npz，`objects_bag.py` 按本包的消息定义读 `LidarObjectArray`，`soak_report.py` 读 glog。

## 验证

- **离线**：`lib/lidar_3d_detection/lidar_3d_bag_replay --bag <bag> --config <配置> --model-dir <模型> --gpu --threads 4 --out x.npz`，同一块板、同一份引擎下，两次结果逐位相同，也应和升级前的版本逐位相同（只改了部署、没改算法时）。和 Python 比用算法仓库的 `cpp_parity.py`。`--preprocess-only` 只做预处理；
- **长时间**：`scripts/orin_soak.sh <bag> - - 30 data/soak`，对装好的节点循环播放 30 分钟，后一半时间加负载。算法仓库的 `objects_bag.py`（能读 `LidarObjectArray`）按遍查不丢帧，`soak_report.py`（能读 glog）汇总延迟、内存、温度。
