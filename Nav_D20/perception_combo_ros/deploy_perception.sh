#!/bin/bash
# 将感知模块交叉编译产物打包成 deb 包，可在 Jetson Orin (arm64) 上直接 dpkg -i 安装
#
# 版本号规则（与 script/make_deb.sh 保持一致）:
#   nav_perception-D20-02-v0.00.01.26042401-arm64.deb
#   <team>-<product>-<ros2_code>-v<ver>.<build>-<arch>.deb
#
# usage:
#   ./deploy_perception.sh                         # 默认参数打包
#   ./deploy_perception.sh -v 0.01.00              # 指定版本号
#   ./deploy_perception.sh -s ./install            # 指定编译产物目录（默认 ./install）
#   ./deploy_perception.sh -o ./deb_output         # 指定 deb 输出目录
#   ./deploy_perception.sh -h                      # 帮助

# ============================================================
# 日志函数
# ============================================================
function log_error() { echo -e "\E[1;31m$(date '+%Y-%m-%d %H:%M:%S') [Error] $*\033[0m"; }
function log_ok()    { echo -e "\E[1;32m$(date '+%Y-%m-%d %H:%M:%S') [ OK ] $*\033[0m"; }
function log_warn()  { echo -e "\E[33m$(date '+%Y-%m-%d %H:%M:%S') [Warn] $*\033[0m"; }
function log_info()  { echo "$(date '+%Y-%m-%d %H:%M:%S') [Info] $*"; }

# ============================================================
# 基础路径
# ============================================================
script_abs_file_path=$(realpath "$0")
PERCEPTION_ROOT_DIR=$(dirname ${script_abs_file_path})
PROJECT_DIR=${PERCEPTION_ROOT_DIR}

# ============================================================
# 版本号默认值（与 make_deb.sh 规则一致）
# ============================================================
team_name="nav_perception"       # 团队/板块名称
prodct_name="D20"            # 产品代码名称
ros2_name="Humble"           # ROS2 版本
declare -A ros2_code_dict=( ["Humble"]="02" ["Foxy"]="01" )
ros2_code=${ros2_code_dict[$ros2_name]}

version="0.00.01"            # 语义版本号，可由外部 -v 传入
deb_src_dir="${PERCEPTION_ROOT_DIR}/install"   # 编译产物目录
output_dir="${PERCEPTION_ROOT_DIR}/deb_output" # deb 输出目录
arch_code=1                  # 1=arm64(默认), 0=x64
arch_names=("x64" "arm64")
arch_types=("amd64" "arm64")

print_help() {
    echo "usage: $0 [options]"
    echo "  -v <version>     版本号, 格式: X.XX.XX, 默认: ${version}"
    echo "  -s <src_dir>     编译产物目录（install/），默认: ${deb_src_dir}"
    echo "  -o <output_dir>  deb 输出目录, 默认: ${output_dir}"
    echo "  -a <arch_code>   架构: 0=x64, 1=arm64, 默认: 1"
    echo "  -r <ros2_name>   ROS2 版本: Humble/Foxy, 默认: Humble"
    echo "  -h               打印帮助"
}

# ============================================================
# 解析命令行参数
# ============================================================
while getopts "v:s:o:a:r:h" opt; do
    case $opt in
        v) version="${OPTARG}"
           IFS='.' read -r -a ver_arr <<< "$version"
           if [ ${#ver_arr[@]} -ne 3 ]; then
               log_error "版本号格式错误: ${version}，应为 X.XX.XX"
               exit 1
           fi
           [ ${#ver_arr[1]} -ne 2 ] && ver_arr[1]="0${ver_arr[1]}"
           [ ${#ver_arr[2]} -ne 2 ] && ver_arr[2]="0${ver_arr[2]}"
           version="${ver_arr[0]}.${ver_arr[1]}.${ver_arr[2]}"
           ;;
        s) deb_src_dir="${OPTARG}" ;;
        o) output_dir="${OPTARG}" ;;
        a) arch_code="${OPTARG}" ;;
        r) ros2_name="${OPTARG}"
           ros2_code=${ros2_code_dict[$ros2_name]}
           if [ -z "${ros2_code}" ]; then
               log_error "不支持的 ros2_name: ${ros2_name}"
               exit 1
           fi
           ;;
        h) print_help; exit 0 ;;
        *) print_help; exit 1 ;;
    esac
done

# ============================================================
# 校验编译产物目录
# ============================================================
if [ ! -d "${deb_src_dir}" ]; then
    log_error "编译产物目录不存在: ${deb_src_dir}"
    log_error "请先运行 build_perception_crosscompile.sh 完成交叉编译"
    exit 1
fi
if [ ! -f "${deb_src_dir}/setup.bash" ]; then
    log_error "${deb_src_dir}/setup.bash 不存在，编译产物可能不完整"
    exit 1
fi

# ============================================================
# 构建版本号（自增逻辑与 make_deb.sh 一致）
# ============================================================
ymd_str=$(date +"%y%m%d")

if [ -d "$CI_BUILDS_DIR" ]; then
    ENV_GLR_VER="${CI_BUILDS_DIR}/../glrws/version"
else
    ENV_GLR_VER=/tmp/glrws/version
fi
mkdir -p ${ENV_GLR_VER}
ENV_VER_FILE="${ENV_GLR_VER}/${team_name}_${prodct_name}.txt"

if [ ! -f "${ENV_VER_FILE}" ]; then
    log_warn "版本记录文件不存在: ${ENV_VER_FILE}，从 01 开始"
    rec_ver="${version}.${ymd_str}00"
    echo "${rec_ver}" > ${ENV_VER_FILE}
else
    rec_ver=$(cat ${ENV_VER_FILE})
fi

IFS='.' read -r -a ver_arr <<< "${rec_ver}"
if [ ${#ver_arr[@]} -ne 4 ]; then
    log_warn "版本记录格式异常 (${rec_ver})，重置"
    rec_ver="${version}.${ymd_str}00"
    echo "${rec_ver}" > ${ENV_VER_FILE}
    IFS='.' read -r -a ver_arr <<< "${rec_ver}"
fi

rec_build_ver=${ver_arr[3]}
rec_ymd=${rec_build_ver:0:6}
rec_index_id=${rec_build_ver:6:2}

if [ "${rec_ymd}" != "${ymd_str}" ]; then
    index_id=1
else
    index_id=$((10#${rec_index_id} + 1))
    [ $index_id -gt 99 ] && index_id=1
fi

build_ver="${ymd_str}$(printf "%02d" ${index_id})"
pkg_ver="${version}.${build_ver}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCH=${arch_types[$arch_code]}
deb_pkg_name="${team_name}-${prodct_name}-${ros2_code}-v${pkg_ver}-${arch_names[$arch_code]}"

log_info "team_name  : ${team_name}"
log_info "prodct_name: ${prodct_name}"
log_info "ros2_name  : ${ros2_name} (code: ${ros2_code})"
log_info "ARCH       : ${ARCH}"
log_info "version    : ${pkg_ver}"
log_info "deb_pkg    : ${deb_pkg_name}.deb"
log_info "src_dir    : ${deb_src_dir}"
log_info "output_dir : ${output_dir}"

# ============================================================
# 构建 deb 目录结构
# ============================================================
DEB_DIR="${PROJECT_DIR}/tmp/${deb_pkg_name}"

# 目标机上的安装路径
PERCEPTION_INSTALL_TARGET="/opt/seres/nav_perception"
PERCEPTION_LIB_TARGET="/opt/seres/nav_perception/lib"
PERCEPTION_START_TARGET="/opt/seres/nav_perception/start"
PERCEPTION_CFG_TARGET="/opt/seres/nav_perception/cfg"

log_info "创建 deb 目录结构..."
rm -rf "${DEB_DIR}"
mkdir -p "${DEB_DIR}${PERCEPTION_INSTALL_TARGET}"
mkdir -p "${DEB_DIR}${PERCEPTION_LIB_TARGET}"
mkdir -p "${DEB_DIR}${PERCEPTION_START_TARGET}"
mkdir -p "${DEB_DIR}${PERCEPTION_CFG_TARGET}"
mkdir -p "${DEB_DIR}/DEBIAN"

# ============================================================
# 拷贝编译产物到 deb 目录
# ============================================================
log_info "拷贝感知节点编译产物..."

# 感知节点包列表（对应 colcon install 的子目录）
perception_nodes=(
    lidar_seg
    visual_perception
    pedestrian_code_conduct
    grid_map_memory_node
    ptz_data_detection
    lidar_3d_detection
)
# ROS2 接口消息包（运行时依赖）
interface_pkgs=(
    common_interface
    interfaces
    mc_interface
    nav_interface
    rte_interface
    robot_interface
    pcl_msgs
    octomap_msgs
    seres_proto
    perception_interface
    nav2_interface
    kygi665_msgs
)

# 拷贝感知节点
for node in "${perception_nodes[@]}"; do
    if [ -d "${deb_src_dir}/${node}" ]; then
        cp -ar "${deb_src_dir}/${node}" "${DEB_DIR}${PERCEPTION_INSTALL_TARGET}/"
        log_info "  已拷贝节点: ${node}"
    else
        log_warn "  节点目录不存在，跳过: ${deb_src_dir}/${node}"
    fi
done

# 拷贝接口消息包
for pkg in "${interface_pkgs[@]}"; do
    if [ -d "${deb_src_dir}/${pkg}" ]; then
        cp -ar "${deb_src_dir}/${pkg}" "${DEB_DIR}${PERCEPTION_INSTALL_TARGET}/"
    fi
done

# 拷贝 colcon 环境文件（setup.bash 等）
for f in setup.bash setup.sh setup.zsh setup.ps1 \
          local_setup.bash local_setup.sh \
          _local_setup_util_sh.py _local_setup_util_ps1.py; do
    [ -f "${deb_src_dir}/${f}" ] && cp "${deb_src_dir}/${f}" "${DEB_DIR}${PERCEPTION_INSTALL_TARGET}/"
done

# ============================================================
# 拷贝三方库（gflags / glog / octomap）的 .so 到 PERCEPTION_LIB_TARGET
# ============================================================
log_info "拷贝三方依赖库..."
for lib_dir in gflags glog-0.6.0 octomap; do
    src_lib="${deb_src_dir}/${lib_dir}/lib"
    if [ -d "${src_lib}" ]; then
        find "${src_lib}" -maxdepth 1 \( -name "*.so" -o -name "*.so.*" \) \
            -exec cp -P {} "${DEB_DIR}${PERCEPTION_LIB_TARGET}/" \;
        log_info "  已拷贝: ${lib_dir}/lib -> ${PERCEPTION_LIB_TARGET}"
    fi
done

# OpenCV 4.10 运行库。节点按 libopencv_*.so.410 链接，目标板的 /usr/local 里不一定有这份库。
# 同时把这些库的 NEEDED 闭包打进包（cblas/atlas、lapack、图像编解码等）。
# CUDA、cuDNN、TensorRT 和 glibc 留在目标板，不打进 deb。
opencv_src="/usr/local/lib"
for name in core imgproc imgcodecs dnn; do
    soname="${opencv_src}/libopencv_${name}.so.410"
    if [ ! -e "${soname}" ]; then
        log_warn "  未找到 ${soname}，跳过"
        continue
    fi
    cp -a "${soname}" "${DEB_DIR}${PERCEPTION_LIB_TARGET}/"
    if [ -L "${soname}" ]; then
        cp -a "$(readlink -f "${soname}")" "${DEB_DIR}${PERCEPTION_LIB_TARGET}/"
    fi
    log_info "  已拷贝 OpenCV: libopencv_${name}.so.410"
done

# 把一个 soname 及其非系统依赖拷进安装库目录。已拷过的库直接跳过。
copy_needed_lib() {
    local name="$1"
    local dest="${DEB_DIR}${PERCEPTION_LIB_TARGET}"
    if [ -e "${dest}/${name}" ]; then
        return 0
    fi
    case "${name}" in
        libc.so.*|libm.so.*|libdl.so.*|libpthread.so.*|librt.so.*|libstdc++.so.*|libgcc_s.so.*|ld-linux*)
            return 0 ;;
        libcublas.so.*|libcublasLt.so.*|libcudnn.so.*|libcudart.so.*|libnvinfer.so.*|libnvonnxparser.so.*|libnvparsers.so.*)
            return 0 ;;
    esac
    local path
    path=$(ldconfig -p | awk -v n="${name}" '$1==n {print $NF; exit}')
    if [ -z "${path}" ] || [ ! -e "${path}" ]; then
        log_warn "  未找到依赖库 ${name}"
        return 0
    fi
    # 只存真实文件和相对 soname 链接。alternatives 的绝对链接在目标板上会断。
    local real real_name
    real=$(readlink -f "${path}")
    real_name=$(basename "${real}")
    if [ ! -e "${dest}/${real_name}" ]; then
        cp -a "${real}" "${dest}/${real_name}"
    fi
    if [ "${name}" != "${real_name}" ]; then
        ln -sfn "${real_name}" "${dest}/${name}"
    fi
    log_info "  已拷贝依赖: ${name} -> ${real_name}"
    local dep
    local needed
    needed=$(readelf -d "${real}" | awk '/NEEDED/ {gsub(/[\[\]]/, "", $5); print $5}')
    for dep in ${needed}; do
        copy_needed_lib "${dep}"
    done
}

for name in core imgproc imgcodecs dnn; do
    real=$(readlink -f "${opencv_src}/libopencv_${name}.so.410" 2>/dev/null || true)
    [ -n "${real}" ] && [ -e "${real}" ] || continue
    needed=$(readelf -d "${real}" | awk '/NEEDED/ {gsub(/[\[\]]/, "", $5); print $5}')
    for dep in ${needed}; do
        copy_needed_lib "${dep}"
    done
done


# ============================================================
# 拷贝启动脚本
# ============================================================
log_info "拷贝启动脚本..."
# service 用后台启动脚本（无 tmux，供 systemd ExecStart 使用）
if [ -f "${PROJECT_DIR}/cfg/start_nav_perception_bg.sh" ]; then
    cp "${PROJECT_DIR}/cfg/start_nav_perception_bg.sh" "${DEB_DIR}${PERCEPTION_START_TARGET}/"
    chmod +x "${DEB_DIR}${PERCEPTION_START_TARGET}/start_nav_perception_bg.sh"
fi
# tmux 手动调试启动脚本（使用 /opt/seres/nav_perception 路径）
if [ -f "${PROJECT_DIR}/run_perception_tmux_opt.sh" ]; then
    cp "${PROJECT_DIR}/run_perception_tmux_opt.sh" "${DEB_DIR}${PERCEPTION_START_TARGET}/"
    chmod +x "${DEB_DIR}${PERCEPTION_START_TARGET}/run_perception_tmux_opt.sh"
fi
# 单节点运行脚本
if [ -f "${PROJECT_DIR}/run_lidar_seg.sh" ]; then
    cp "${PROJECT_DIR}/run_lidar_seg.sh" "${DEB_DIR}${PERCEPTION_START_TARGET}/"
    chmod +x "${DEB_DIR}${PERCEPTION_START_TARGET}/run_lidar_seg.sh"
fi
if [ -f "${PROJECT_DIR}/run_visualperception_2d.sh" ]; then
    cp "${PROJECT_DIR}/run_visualperception_2d.sh" "${DEB_DIR}${PERCEPTION_START_TARGET}/"
    chmod +x "${DEB_DIR}${PERCEPTION_START_TARGET}/run_visualperception_2d.sh"
fi
if [ -f "${PROJECT_DIR}/run_pedestrian_code_conduct.sh" ]; then
    cp "${PROJECT_DIR}/run_pedestrian_code_conduct.sh" "${DEB_DIR}${PERCEPTION_START_TARGET}/"
    chmod +x "${DEB_DIR}${PERCEPTION_START_TARGET}/run_pedestrian_code_conduct.sh"
fi
if [ -f "${PROJECT_DIR}/run_ptz_data_detection.sh" ]; then
    cp "${PROJECT_DIR}/run_ptz_data_detection.sh" "${DEB_DIR}${PERCEPTION_START_TARGET}/"
    chmod +x "${DEB_DIR}${PERCEPTION_START_TARGET}/run_ptz_data_detection.sh"
fi

# ============================================================
# 拷贝 service 和 cfg 文件
# ============================================================
log_info "拷贝 service / 配置文件..."
if [ -f "${PROJECT_DIR}/cfg/nav_perception.service" ]; then
    cp "${PROJECT_DIR}/cfg/nav_perception.service" "${DEB_DIR}${PERCEPTION_CFG_TARGET}/"
fi
# 如有其他配置文件（yaml 等），也一并拷贝
if [ -d "${PROJECT_DIR}/cfg" ]; then
    find "${PROJECT_DIR}/cfg" -maxdepth 1 \
        \( -name "*.yaml" -o -name "*.json" -o -name "*.conf" -o -name "*.env" \) \
        -exec cp {} "${DEB_DIR}${PERCEPTION_CFG_TARGET}/" \;
fi

# ============================================================
# 生成 DEBIAN/control
# ============================================================
cat > "${DEB_DIR}/DEBIAN/control" << EOF
Package: ${team_name//_/-}-${prodct_name,,}-${ros2_code}
Version: ${pkg_ver}
Architecture: ${ARCH}
Maintainer: seres-nav_perception <nav_perception@seres.cn>
Description: Seres ${prodct_name} Perception Package (${ros2_name})
 Installs to ${PERCEPTION_INSTALL_TARGET}, includes lidar_seg,
 visual_perception, pedestrian_code_conduct, grid_map_memory_node,
 ptz_data_detection and supporting libraries.
EOF

# ============================================================
# 生成 DEBIAN/preinst（安装前校验：必须 sudo）
# ============================================================
cat > "${DEB_DIR}/DEBIAN/preinst" << 'PREINST_EOF'
#!/bin/bash
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: This package must be installed with administrator privileges!" >&2
    echo "Please use: sudo dpkg -i <package>.deb" >&2
    exit 1
fi
# 如果 nav_perception tmux session 正在运行，先停止
if command -v tmux &>/dev/null && tmux has-session -t nav_perception 2>/dev/null; then
    echo "[preinst] 停止已有的 nav_perception tmux session..."
    tmux kill-session -t nav_perception
fi
# 如果服务正在运行，先停止
if systemctl is-active --quiet nav_perception.service 2>/dev/null; then
    echo "[preinst] 停止 nav_perception.service..."
    systemctl stop nav_perception.service
fi
PREINST_EOF
chmod 755 "${DEB_DIR}/DEBIAN/preinst"

# ============================================================
# 生成 DEBIAN/postinst（安装后：ldconfig + 替换 service 用户 + 注册服务）
# ============================================================
cat > "${DEB_DIR}/DEBIAN/postinst" << 'POSTINST_EOF'
#!/bin/bash
set -e

# 1. 注册 systemd 服务（固定以 root 运行，避免 sudo 安装时误用 SUDO_USER）
# 注：依赖库安装在 /opt/seres/nav_perception/lib，通过启动脚本中的 LD_LIBRARY_PATH 加载，无需 ldconfig
SERVICE_SRC="/opt/seres/nav_perception/cfg/nav_perception.service"
SERVICE_DST="/etc/systemd/system/nav_perception.service"
if [ -f "${SERVICE_SRC}" ]; then
    cp "${SERVICE_SRC}" "${SERVICE_DST}"
    chmod 644 "${SERVICE_DST}"
    echo "[postinst] 已安装 ${SERVICE_DST} (User=root)"
else
    echo "[postinst] Warning: ${SERVICE_SRC} 不存在，跳过 service 注册" >&2
fi

# 3. 确保后台启动脚本有可执行权限
chmod +x /opt/seres/nav_perception/start/start_nav_perception_bg.sh 2>/dev/null || true
chmod +x /opt/seres/nav_perception/start/run_perception_tmux_opt.sh 2>/dev/null || true

# 4. 注册并启用服务
if [ -f "${SERVICE_DST}" ]; then
    systemctl daemon-reload
    systemctl enable nav_perception.service
    echo "[postinst] nav_perception.service 已注册并设置为开机自启"
    echo "[postinst] 如需立即启动请执行: sudo systemctl start nav_perception.service"
fi

echo "[postinst] 安装完成，感知模块路径: /opt/seres/nav_perception"
POSTINST_EOF
chmod 755 "${DEB_DIR}/DEBIAN/postinst"

# ============================================================
# 生成 DEBIAN/prerm（卸载前：停止服务）
# ============================================================
cat > "${DEB_DIR}/DEBIAN/prerm" << 'PRERM_EOF'
#!/bin/bash
# 卸载前停止感知服务和 tmux session
if systemctl is-active --quiet nav_perception.service 2>/dev/null; then
    echo "[prerm] 停止 nav_perception.service..."
    systemctl stop nav_perception.service
fi
if systemctl is-enabled --quiet nav_perception.service 2>/dev/null; then
    echo "[prerm] 禁用 nav_perception.service..."
    systemctl disable nav_perception.service
fi
if command -v tmux &>/dev/null && tmux has-session -t nav_perception 2>/dev/null; then
    echo "[prerm] 停止 nav_perception tmux session..."
    tmux kill-session -t nav_perception
fi
PRERM_EOF
chmod 755 "${DEB_DIR}/DEBIAN/prerm"

# ============================================================
# 生成 DEBIAN/postrm（卸载后：清理 service 文件和 ldconfig）
# ============================================================
cat > "${DEB_DIR}/DEBIAN/postrm" << 'POSTRM_EOF'
#!/bin/bash
if [ "$1" = "remove" ] || [ "$1" = "purge" ]; then
    SERVICE_DST="/etc/systemd/system/nav_perception.service"
    [ -f "${SERVICE_DST}" ] && rm -f "${SERVICE_DST}"
    systemctl daemon-reload 2>/dev/null || true
    ldconfig
    echo "[postrm] 感知模块已卸载"
fi
POSTRM_EOF
chmod 755 "${DEB_DIR}/DEBIAN/postrm"

# ============================================================
# 构建 deb 包
# ============================================================
mkdir -p "${output_dir}"
log_info "开始构建 deb 包..."
dpkg-deb --build "${DEB_DIR}" "${output_dir}/${deb_pkg_name}.deb"
if [ $? -ne 0 ]; then
    log_error "deb 包构建失败"
    exit 1
fi

# 清理临时目录
rm -rf "${PROJECT_DIR}/tmp"

# 写回版本记录（非 CI 环境）
if [ "${X86_BUILD_JOB}" != "true" ]; then
    echo "${pkg_ver}" > "${ENV_VER_FILE}"
fi

log_ok "deb 包构建成功!"
log_ok "输出路径: ${output_dir}/${deb_pkg_name}.deb"
log_ok "Orin 上安装: sudo dpkg -i ${deb_pkg_name}.deb"
