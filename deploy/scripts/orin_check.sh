#!/usr/bin/env bash
# Environment report of the Jetson (also runs on the x86 server, where the Jetson parts print nothing):
# versions, power mode, memory and disk, the build tools the C++ pipeline needs. docs/DEPLOY_ORIN.md.
#
#   deploy/scripts/orin_check.sh > orin_env.txt
#   PYTHON=/path/to/venv/bin/python deploy/scripts/orin_check.sh     # the server: check the project env
# Lines starting with MISSING name what to install (apt on the Orin, docs/DEPLOY_ORIN.md 3).
set -u
PY=${PYTHON:-python3}

section() { printf '\n== %s\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
pkg() { dpkg-query -W -f='${Package} ${Version}\n' "$@" 2>/dev/null || true; }

section host
hostname; uname -srm
[ -r /proc/device-tree/model ] && tr -d '\0' < /proc/device-tree/model && echo
[ -r /etc/nv_tegra_release ] && head -1 /etc/nv_tegra_release
pkg nvidia-l4t-core
. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"

section cuda
if [ -x /usr/local/cuda/bin/nvcc ]; then /usr/local/cuda/bin/nvcc --version | tail -2; else echo "MISSING nvcc (/usr/local/cuda)"; fi
ls -d /usr/local/cuda* 2>/dev/null

section "cudnn / tensorrt"
pkg 'libcudnn9*' 'libnvinfer10' 'libnvinfer-dev' 'libnvonnxparsers10' 'libnvonnxparsers-dev' 'python3-libnvinfer'
"$PY" -c "import tensorrt as t; print('python tensorrt', t.__version__)" 2>/dev/null || echo "MISSING python tensorrt (python3-libnvinfer / pip tensorrt)"
# trtexec is optional: tools/deploy/bench_engines.py measures the engines without it
if have trtexec; then command -v trtexec; elif [ -x /usr/src/tensorrt/bin/trtexec ]; then echo /usr/src/tensorrt/bin/trtexec
else echo "(no trtexec: tools/deploy/bench_engines.py instead)"; fi

section opencv
ls /usr/local/lib/libopencv_core.so* /usr/lib/aarch64-linux-gnu/libopencv_core.so* 2>/dev/null | head -3

section "ros 2"
ls /opt/ros 2>/dev/null || echo "MISSING /opt/ros"
pkg ros-humble-rclcpp ros-humble-rosbag2-cpp ros-humble-sensor-msgs ros-humble-visualization-msgs ros-humble-rclcpp-components
# not `colcon version-check`: it queries PyPI and hangs without network
if have colcon; then command -v colcon; pkg python3-colcon-common-extensions; else echo "MISSING colcon (python3-colcon-common-extensions)"; fi

section "build tools"
gcc --version | head -1; g++ --version | head -1; cmake --version | head -1
for p in libeigen3-dev libyaml-cpp-dev libgtest-dev zlib1g-dev; do
  v=$(dpkg-query -W -f='${Version}' "$p" 2>/dev/null); [ -n "$v" ] && echo "$p $v" || echo "MISSING $p"
done

section python
echo "$PY: $("$PY" --version 2>&1)"
"$PY" -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)" 2>/dev/null || echo "MISSING numpy / scipy"
"$PY" -c "import onnxruntime as o; print('onnxruntime', o.__version__)" 2>/dev/null || echo "(no onnxruntime: not needed on the Orin)"
"$PY" -c "import torch; print('torch', torch.__version__)" 2>/dev/null || echo "(no torch: not needed on the Orin)"

section power
if have nvpmodel; then (sudo -n nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null) | head -3; fi
if have jetson_clocks; then sudo -n jetson_clocks --show 2>/dev/null | grep -E "SOC family|Online CPUs|GPU|EMC" | head -6 \
  || echo "(jetson_clocks --show needs sudo)"; fi

section "memory / disk"
free -h | head -2
df -h / "$HOME" 2>/dev/null | awk 'NR == 1 || !seen[$0]++'

section gpu
if have nvidia-smi; then nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; fi
if have tegrastats; then timeout 3 tegrastats --interval 1000 2>/dev/null | head -1; fi
