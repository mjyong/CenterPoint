# ptz_data_detection 上板编译与运行

本文对应 Jetson Orin、L4T R36.4.4、Ubuntu 22.04 aarch64。二进制只能在同架构、同 JetPack 的板上运行。

本地已经打好的发布目录：

`perception_combo_ros/orin_release/ptz_data_detection_orin`

| 内容 | 路径 |
| --- | --- |
| colcon 安装结果（程序、接口包、配置） | `install/` |
| 启动脚本 | `run_ptz_data_detection.sh` |
| TensorRT 10.3.0.30 arm64 安装包 | `deps/tensorrt/` |

编译来源板：`seres@192.168.123.100`，系统 Ubuntu 22.04.5，内核 `5.15.148-tegra`。发布包约 280 MB，其中 TensorRT 安装包约占 216 MB。

## 1. 目标板环境

目标板需要和编译板同一套用户态库：

- Ubuntu 22.04 arm64，L4T R36.4.4（`/etc/nv_tegra_release` 为 R36 REVISION 4.4）
- ROS 2 Humble：`/opt/ros/humble`
- CUDA 12.6：`/usr/local/cuda`
- cuDNN 9（`libcudnn.so.9`）
- OpenCV **4.10**：`/usr/local/lib/libopencv_core.so.410`。apt 里的 OpenCV 4.5 不能替代
- 系统库：`libgoogle-glog-dev`、`libgflags-dev`、`nlohmann-json3-dev`
- TensorRT **10.3.0.30+cuda12.5**（与 JetPack 6.2 配套，头文件和库在 `/usr/include/aarch64-linux-gnu`、`/usr/lib/aarch64-linux-gnu`）

上板后先核对：

```bash
uname -m
cat /etc/nv_tegra_release
ls /opt/ros/humble/setup.bash /usr/local/cuda/bin/nvcc
ls /usr/local/lib/libopencv_core.so.410
ls /usr/include/aarch64-linux-gnu/NvInfer.h /usr/lib/aarch64-linux-gnu/libnvinfer.so.10
date
```

`uname -m` 应为 `aarch64`。`date` 必须是真实日期。这台编译板曾经停在 1970-01-01，`make` 会报 clock skew。不对时在能访问这台电脑的前提下执行：

```bash
sudo date -s "2026-10-20 11:00:00"
sudo hwclock --systohc
```

板上如果没有 DNS，不要在板上 `apt update`。TensorRT 用随包的 deb 安装。

## 2. 把编译结果拷到其它板并运行

在开发机上，发布目录可以直接拷，也可以先打成一个包：

```bash
cd perception_combo_ros/orin_release
tar -C ptz_data_detection_orin -czf ptz_data_detection_orin.tar.gz .
scp ptz_data_detection_orin.tar.gz seres@<目标板IP>:~/
```

在目标板上：

```bash
mkdir -p ~/ptz_data_detection_orin
tar -xzf ~/ptz_data_detection_orin.tar.gz -C ~/ptz_data_detection_orin
cd ~/ptz_data_detection_orin

# 目标板还没有 TensorRT 时安装。只运行可以只装三个运行库；要在这块板上重新编译则全部安装。
sudo dpkg -i deps/tensorrt/libnvinfer10_10.3.0.30-1+cuda12.5_arm64.deb \
             deps/tensorrt/libnvinfer-plugin10_10.3.0.30-1+cuda12.5_arm64.deb \
             deps/tensorrt/libnvonnxparsers10_10.3.0.30-1+cuda12.5_arm64.deb

chmod +x run_ptz_data_detection.sh
./run_ptz_data_detection.sh
```

启动脚本会：

1. `source /opt/ros/humble/setup.bash` 和 `install/setup.bash`
2. 设置 `SERES_PERCEPTION_INSTALL_DIR` 为该目录下的 `install`
3. 把 `libptz_data_detection.so`、CUDA、`/usr/local/lib` 放进 `LD_LIBRARY_PATH`
4. 执行 `install/ptz_data_detection/lib/ptz_data_detection/ptz_data_detection_2d`

`install/setup.bash` 用脚本自身位置定位，目录可以改名或换路径，不必固定在 `/home/seres`。

运行前改配置 `install/ptz_data_detection/config/ptz_data_detection.json`：

- `det_model_path` 指向本板的 TensorRT engine。当前值是编译板配置里的绝对路径，别的板子上这个文件通常不存在。engine 跟 TensorRT 版本和 GPU 绑定，换 TensorRT 大版本或换一种 GPU 后需要在目标板上重新生成。
- `save_task_img_path` 改成目标板上存在、且进程用户可写的目录。
- `use_detection` 为 `true` 时，`Init()` 会加载 `det_model_path` 并调用 `make_pipe`。模型文件缺失时进程在启动阶段退出。

程序读取的配置路径是：

`$SERES_PERCEPTION_INSTALL_DIR/ptz_data_detection/config/ptz_data_detection.json`

话题与服务（默认值）：

| 配置项 | 默认名 |
| --- | --- |
| `perception_detect_srv` | `/perception/perception_detect` |
| `take_photo_ctrl_topic` | `/robot/topic/ptz/nav_take_photo_ctrl` |
| `take_photo_data_topic` | `/robot/topic/ptz/nav_take_photo_data` |
| `cam_tem_matrix_topic` | `/robot/topic/ptz/cam_tem_matrix` |

启动后可用下面命令确认进程和依赖都在：

```bash
pgrep -a ptz_data_detection_2d
ldd install/ptz_data_detection/lib/ptz_data_detection/ptz_data_detection_2d | grep "not found" || echo "依赖完整"
```

第二条要在 `source install/setup.bash` 并且设置好 `LD_LIBRARY_PATH` 之后执行。出现 `not found` 时，缺的是 ROS、OpenCV 4.10 或 TensorRT，不是这个发布包本身。

## 3. 在板上从源码编译

在开发机 Cursor 里改代码，再把源码同步到板子上编译。不要在 x86 笔记本上编这个包拿去板上跑：`ON_LAPTOP` 链的是 x86 TensorRT。

板上原生编译必须加 `-DON_ORIN_DEBUG=ON`。检测到 `aarch64` 时，CMake 默认走 `/l4t/jetson_arm64_sysroot`，那是交叉编译 sysroot，Orin 本机没有这个目录。

### 3.1 安装编译依赖

ROS、CUDA、OpenCV、glog、json、colcon 按第 1 节确认。TensorRT 开发包：

```bash
sudo dpkg -i deps/tensorrt/*.deb
```

包名与版本：

- `libnvinfer10`、`libnvinfer-dev`、`libnvinfer-headers-dev`
- `libnvinfer-plugin10`、`libnvinfer-plugin-dev`、`libnvinfer-headers-plugin-dev`
- `libnvonnxparsers10`、`libnvonnxparsers-dev`

版本均为 `10.3.0.30-1+cuda12.5`。

### 3.2 同步源码

在开发机仓库根目录执行。只同步这个节点和它依赖的接口包，不要带 `build/`、`install/`、`log/`。

```bash
BOARD=seres@<板子IP>
ssh "$BOARD" 'mkdir -p ~/Nav_D20/perception_combo_ros/common_interface/seres_interface/ros ~/Nav_D20/perception_combo_ros/node_src'
rsync -az \
  perception_combo_ros/common_interface/seres_interface/ros/common_interface \
  perception_combo_ros/common_interface/seres_interface/ros/nav_interface \
  perception_combo_ros/common_interface/seres_interface/ros/nav2_interface \
  perception_combo_ros/common_interface/seres_interface/ros/perception_interface \
  perception_combo_ros/common_interface/seres_interface/ros/rte_interface \
  "$BOARD:~/Nav_D20/perception_combo_ros/common_interface/seres_interface/ros/"
rsync -az --exclude build --exclude install --exclude log \
  perception_combo_ros/node_src/ptz_data_detection \
  "$BOARD:~/Nav_D20/perception_combo_ros/node_src/"
```

### 3.3 编译

在板子上：

```bash
source /opt/ros/humble/setup.bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

cd ~/Nav_D20/perception_combo_ros
colcon build \
  --base-paths common_interface/seres_interface/ros \
  --packages-select common_interface nav2_interface nav_interface perception_interface rte_interface \
  --cmake-args -DBUILD_TESTING=OFF \
  --install-base install

source install/setup.bash
cd node_src
colcon build \
  --packages-select ptz_data_detection \
  --cmake-args -DON_ORIN_DEBUG=ON -DBUILD_TESTING=ON \
  --install-base ~/Nav_D20/perception_combo_ros/install
```

成功时最后一行是 `Summary: 1 package finished`。stderr 里的 unused variable 警告不影响产物。

产物位置：

```text
~/Nav_D20/perception_combo_ros/install/ptz_data_detection/lib/ptz_data_detection/ptz_data_detection_2d
~/Nav_D20/perception_combo_ros/install/ptz_data_detection/lib/ptz_data_detection/libptz_data_detection.so
~/Nav_D20/perception_combo_ros/install/ptz_data_detection/config/ptz_data_detection.json
```

把新的 `install/` 覆盖回开发机发布目录后，按第 2 节重新打包即可：

```bash
rsync -az seres@<板子IP>:~/Nav_D20/perception_combo_ros/install/ \
  perception_combo_ros/orin_release/ptz_data_detection_orin/install/
```

### 3.4 板上测试

单元测试不加载 engine，`use_detection` 在测试里被关掉。

```bash
source /opt/ros/humble/setup.bash
source ~/Nav_D20/perception_combo_ros/install/setup.bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export ROS_LOCALHOST_ONLY=1
cd ~/Nav_D20/perception_combo_ros/node_src
colcon test --packages-select ptz_data_detection \
  --event-handlers console_direct+ \
  --return-code-on-test-failure
```

本次编译板上的结果：`ptz_take_photo_test` 10 项全部通过。

## 4. 用仓库里的脚本运行

`perception_combo_ros/run_ptz_data_detection.sh` 假定自己和 `install/` 在同一级，并设置 `SERES_PERCEPTION_INSTALL_DIR` 为该目录下的 `install`。把发布包的 `install/` 放到 `perception_combo_ros/install` 后也可以用这个脚本，但要同时保证 `LD_LIBRARY_PATH` 包含 `install/ptz_data_detection/lib/ptz_data_detection`。发布包里的 `run_ptz_data_detection.sh` 已经包含这一项，拷到其它板时优先用它。
