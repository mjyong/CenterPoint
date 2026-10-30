# lidar_3d_detection

车上部署的激光感知节点：点云预处理 → 3D 检测（CenterPoint Pillar，TensorRT FP16）→ 跟踪 → 预测，发布给规划。在 Jetson AGX Orin 上编译、打成 deb 包，装上就能跑。

代码和算法仓库（CenterPoint 仓库的 `deploy/`）是同一套，只是换了名字。算法仓库里有逐段对 Python 的测试、bag 回放的逐位比较、板上的延迟和长时间测试（`docs/DEPLOY_ORIN.md`）。

## 目录

```text
perception_combo_ros/
  common_interface/seres_interface/ros/   common_interface、nav2_interface、perception_interface（LidarObjectArray）
  node_src/lidar_3d_detection/
    include/lidar3d/, src/, tests/        库 lidar3d：镜像，不要在这里改（见"和算法仓库对齐"）
    tools/lidar_3d_replay.cpp             离线回放（镜像）
    src/ros_convert.cpp, src/bag_replay.cpp, include/ros_convert.hpp     ROS 消息转换、bag 回放（镜像）
    src/lidar_3d_detection.cpp, src/main.cpp, include/lidar_3d_detection.h, include/logging.h   节点（本包自己的）
    config/lidar_3d_detection.yaml        参数：算法仓库导出，末尾 5 个部署键（model_dir、四个话题名）是本包自己的
    launch/                               ros2 launch
    packaging/                            deb 里的启动脚本、systemd 单元、默认设置、安装脚本
    scripts/build_deb.sh                  在板上打 deb
    scripts/orin_soak.sh                  长时间循环回放（装好的节点）
    scripts/sync_to_orin.sh, scripts/orin_check.sh
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

## 在板上打 deb

前置：JetPack 6（L4T R36）、ROS 2 Humble（`/opt/ros/humble`）、colcon、CUDA、TensorRT，以及**在这块板上建好的引擎**。引擎只能在建它的 TensorRT 上加载，所以 deb 依赖的 `libnvinfer10` 版本和板上完全一致。

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
  - `lidar-3d-detection_<版本>_arm64.deb`，版本是 `package.xml` 的版本加 git 提交号；
  - 同名 `.txt` 报告：板子、模型文件的 sha256、Depends，以及找不到包的库；
- **选项**：
  - `--prefix`：安装位置；
  - `--service-user`：服务用哪个用户跑，默认是打包的人；
  - `--enable-service`：装上就开机自启并启动，默认只装不启用；
  - `--skip-build`：直接打包已有的安装目录；
- 不需要 root。

## 安装、运行

```bash
sudo apt install ./lidar-3d-detection_*_arm64.deb      # 或 sudo dpkg -i
# 手动跑（前台）：
/opt/seres/perception/lidar_3d_detection/bin/start_lidar_3d_detection.sh
# 当服务跑：
sudo systemctl start lidar-3d-detection        # enable：开机自启
journalctl -u lidar-3d-detection -f            # 输出；glog 文件在 /tmp/lidar_3d_detection_logs
```

- 装到 `/opt/seres/perception`：四个接口包、本包、模型（`lidar_3d_detection/model`）、配置（`lidar_3d_detection/config`）；
- 另外两个文件：
  - `/lib/systemd/system/lidar-3d-detection.service`；
  - `/etc/default/lidar-3d-detection`：`ROS_DOMAIN_ID`、`RMW_IMPLEMENTATION`、`LIDAR3D_THREADS`（默认 4）、`LIDAR3D_CONFIG`、`LIDAR3D_MODEL_DIR`、`LIDAR3D_LOG_DIR`；
- 配置和 `/etc/default` 那个文件是 conffile：升级时保留你改过的；`dpkg -r` 卸载时也留着，`dpkg -P` 才删掉；
- 起来以后日志里有一行 "lidar_3d_detection up: ..."，之后每秒一行：框数、轨迹数、各段耗时（ms）；
- 不经过 deb，直接用 colcon 的安装目录也行：`source install/setup.bash && ros2 launch lidar_3d_detection lidar_3d_detection.launch.py`。

## 和算法仓库对齐

- 改代码在算法仓库的 `deploy/` 里改，在那里测（对 Python 的逐段测试、bag 回放）。改完运行 `python tools/deploy/sync_deploy_deb.py --write`，按固定的改名规则重写这里的镜像文件；
- 不带 `--write` 就是检查，算法仓库的测试 `test_deploy_deb_mirrors_deploy` 每次都跑。检查四项：
  - 镜像文件；
  - 配置：除末尾 5 个部署键外，等于 `export_cpp_config.py --robot kygi` 的输出；
  - `LidarObject*` 的字段和 `dog_perception_msgs` 一致；
  - `kygi665_msgs` 的消息一致；
- 本包自己的文件（节点、CMakeLists、package.xml、launch、packaging、scripts）不镜像，改的时候和 `deploy/ros2` 的节点保持同样的行为：
  - 订阅的 QoS；
  - 每秒一行的统计；
  - 线程数。

## 验证

- **离线**：`lib/lidar_3d_detection/lidar_3d_bag_replay --bag <bag> --config <配置> --model-dir <模型> --gpu --threads 4 --out x.npz`，结果应和算法仓库的 `dogperc_bag_replay` 逐位相同（同一块板、同一份引擎）。`--preprocess-only` 只做预处理；
- **长时间**：`scripts/orin_soak.sh <bag> - - 30 data/soak`，对装好的节点循环播放 30 分钟，后一半时间加负载。算法仓库的 `objects_bag.py`（能读 `LidarObjectArray`）按遍查不丢帧，`soak_report.py`（能读 glog）汇总延迟、内存、温度。
