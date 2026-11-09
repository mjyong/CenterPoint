在 Orin（aarch64，L4T R36.4.4，ROS 2 Humble）上原生编译，再在同一台机器的 `install/` 旁打包。板上编译必须加 `-DON_ORIN_DEBUG=ON`。不传这个参数时，CMake 会去找交叉编译用的 `/l4t/jetson_arm64_sysroot`，Orin 本机没有这个目录。

打进 deb 的节点：`lidar_seg`、`visual_perception`、`pedestrian_code_conduct`、`grid_map_memory_node`、`ptz_data_detection`，以及 `install/` 里已有的接口包。

### 1. 编译

依赖：ROS 2 Humble、CUDA 12.6、cuDNN 9、`/usr/local/lib` 下的 OpenCV 4.10、PCL 1.12、octomap、TensorRT 10.3.0.30。系统时间必须是真实日期，停在 1970 年时 `make` 会报 clock skew。打包时会把 `libopencv_{core,imgproc,imgcodecs,dnn}.so.410` 以及它们的非 CUDA 依赖（`libcblas`、`libatlas`、`liblapack`、`libopenblas` 和 jpeg/png/tiff/webp 等）放进 deb 的 `/opt/seres/nav_perception/lib`。目标板不必再单独安装 OpenCV 4.10。CUDA、cuDNN、TensorRT 仍用板子上已有的库。

```bash
source /opt/ros/humble/setup.bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}

# 接口包。common_interface、nav2_interface、nav_interface、perception_interface、rte_interface
cd perception_combo_ros
colcon build \
  --base-paths common_interface/seres_interface/ros \
  --packages-select common_interface nav2_interface nav_interface perception_interface rte_interface \
  --cmake-args -DBUILD_TESTING=OFF \
  --install-base install

source install/setup.bash
cd node_src
colcon build \
  --packages-select visual_perception pedestrian_code_conduct lidar_seg grid_map_memory_node ptz_data_detection \
  --cmake-args -DON_ORIN_DEBUG=ON -DBUILD_TESTING=OFF -DCOM_IPC=OFF \
    -DSERES_PERCEPTION_ROOT_DIR="$(cd .. && pwd)" \
  --install-base ../install
```

`pedestrian_code_conduct` 的目录名和包名一致，需有 `package.xml`，否则 `colcon --packages-select` 找不到它。`lidar_seg` 的源码目录是 `node_src/lidar_seg_module`，`grid_map_memory_node` 的源码目录是 `node_src/GridMapTask`。

### 2. 打包

在 `perception_combo_ros/` 下执行，脚本默认读取旁边的 `install/`。版本号必须是 `X.XX.XX`，默认 `0.00.01`。`-v 0.1` 只有两段，脚本会直接退出。

```bash
./deploy_perception.sh -v 0.00.01
```

构建号是 `年月日 + 两位序号`，例如 `26102002`。输出文件：

```text
deb_output/nav_perception-D20-02-v0.00.01.26102002-arm64.deb
```

包名是 `nav-perception-d20-02`。`install/` 里缺哪个节点，脚本会跳过并打印警告，不会失败。`deb_output/` 不纳入 git。

已打出来的包装在 `deb_output/`，按文件名查找：

```bash
find deb_output -name 'nav_perception-D20*.deb'
```

### 3. 安装与运行

安装到 `/opt/seres/nav_perception`。postinst 把 `cfg/nav_perception.service` 复制到 `/etc/systemd/system/nav_perception.service`（以 root 运行），并 `systemctl enable`。

```bash
sudo dpkg -i nav_perception-D20-02-v0.00.01.26102002-arm64.deb
sudo systemctl start nav_perception.service
sudo systemctl status nav_perception.service
journalctl -u nav_perception.service -f
```

调试时先停掉 service，再用 tmux 脚本，避免同一节点起两份：

```bash
sudo systemctl stop nav_perception.service
bash /opt/seres/nav_perception/start/run_perception_tmux_opt.sh
```

开机自启：

```bash
sudo systemctl is-enabled nav_perception.service   # enabled
```

`visual_perception_2d`、`ptz_data_detection_2d`、`pedestrian_code_conduct_2d` 的动态库在各自的 `lib/<包名>/` 下。`cfg/start_nav_perception_bg.sh` 启动前会把该目录加进 `LD_LIBRARY_PATH`。节点配置从 `SERES_PERCEPTION_INSTALL_DIR`（`/opt/seres/nav_perception`）读取。`ptz_data_detection` 的 `det_model_path`、`save_task_img_path` 要改成目标板上的路径；TensorRT engine 与 TensorRT 版本和 GPU 绑定，换板或换 TensorRT 大版本后需要在目标板上重新生成。

手动触发 `ptz_data_detection` 抓图：服务 `/perception/perception_detect`，脚本 `grab_ptz_image.py`。jpeg 写到当前目录下的 `./tmp/`，文件名带事件号和时间。拍照指令都发到 `/robot/topic/ptz/nav_take_photo_ctrl`，消息类型是 `NavPtzTakePhotoCtrlMulti`。单张事件的 `num` 为 1，事件 19（货车停放）的张数和间隔由 `ptz_data_detection.json` 的 `multi_photo_num`（默认 5）、`multi_photo_interval_ms`（默认 20）决定。云台要把图发回 `/robot/topic/ptz/nav_take_photo_data`，否则服务会空转到超时。脚本会把返回的每一张都存下来。

```bash
source /opt/ros/humble/setup.bash
source /opt/seres/nav_perception/setup.bash
cd perception_combo_ros
python3 grab_ptz_image.py        # 默认 event_id 14，彩色
python3 grab_ptz_image.py 13     # 红外
python3 grab_ptz_image.py 15     # 路灯，彩色
python3 grab_ptz_image.py 19     # 货车停放，彩色连拍
```

### 4. 还原上一版安装包

`dpkg` 不会自己留下上一版 deb，要留着 `deb_output/` 里的旧文件。先找出本机有哪些包，再对上一版执行 `dpkg -i`。当前装的是 `26102002`，上一版是 `26102001`（只有 `ptz_data_detection`）。先停服务，避免正在跑的进程占着文件。

```bash
find deb_output -name 'nav_perception-D20*.deb'
```

```bash
sudo systemctl stop nav_perception.service
sudo dpkg -i nav_perception-D20-02-v0.00.01.26102001-arm64.deb
sudo systemctl start nav_perception.service
dpkg -l nav-perception-d20-02
```

`dpkg -l` 的版本应变回 `0.00.01.26102001`。新版本多出来、旧包里没有的文件会被卸掉。

### 5. 清除安装内容

`dpkg -r` 会停掉并 disable `nav_perception.service`，删除 unit 文件。它不删除 `/opt/seres/nav_perception`。

```bash
sudo dpkg -r nav-perception-d20-02
sudo dpkg --purge nav-perception-d20-02
sudo rm -rf /opt/seres/nav_perception
```

确认：

```bash
dpkg -l | grep nav-perception-d20-02          # 无输出
systemctl status nav_perception.service       # Unit not found
```
