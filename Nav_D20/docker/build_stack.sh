#!/bin/bash
# Build the D20 stack, one prefix per module, each against its own interface copies, the way each module's own build
# does it (planning/build.sh, slam/build_l4t.sh -arm, perception_combo_ros/build_perception.sh; our node:
# docs/DEPLOY_ORIN.md section 8):
#   planning        seres_interface (6 packages), planning/third_party/nav_interface, planning
#   slam            seres_interface (kygi665_msgs ...), slam's novatel_oem7_msgs, basic, livox_ros_driver2, lightning
#   nav_perception  perception_combo_ros/common_interface, lidar_seg, grid_map_memory_node (the robot team's)
#   perception      deploy_deb: common_interface, slam/interface, lidar_3d_detection (ours, CUDA + TensorRT)
# The robot team keeps a copy of the interface packages in each module and the copies differ (13 names twice or three
# times; kygi665_msgs: "RTK fixed" is 58 in seres_interface, 55 in slam/interface): one shared workspace would quietly
# change what a module reads. On the dog each module is installed apart the same way (/opt/seres/nav/planning,
# /opt/seres/nav/slam, /opt/seres/nav_perception, /opt/seres/perception).
#
#   bash Nav_D20/docker/build_stack.sh [<root>] [planning|slam|nav_perception|perception ...]
#     <root> (default <repo>/build/nav_dev): prefixes <root>/install/<module>, build trees <root>/build/<module>
#   In the nav-dev image (Dockerfile.nav-dev, run with Nav_D20/docker/run.sh) and on dog 8 (T56: /opt/seres/nav_cw,
#   with its deps prefix already in CMAKE_PREFIX_PATH / AMENT_PREFIX_PATH).
#   NAV_CUDA_ARCH (default 89 on x86, 87 on aarch64), NAV_JOBS (make -j, default nproc), NAV_WORKERS (colcon, 4),
#   NAV_ACADOS_DIR (default /opt/acados/cmake when it exists), NAV_DRY_RUN=1 prints the colcon commands only.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nav="$(cd "${here}/.." && pwd)"
repo="$(cd "${nav}/.." && pwd)"
root="${repo}/build/nav_dev"
if [[ $# -gt 0 && "$1" != planning && "$1" != slam && "$1" != nav_perception && "$1" != perception ]]; then
  root="$1"
  shift
fi
modules=("$@")
[[ ${#modules[@]} -gt 0 ]] || modules=(planning slam nav_perception perception)
arch="${NAV_CUDA_ARCH:-$([[ "$(uname -m)" == aarch64 ]] && echo 87 || echo 89)}"
jobs="${NAV_JOBS:-$(nproc)}"
dd="${repo}/deploy_deb"
common=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
cuda=(-DCMAKE_CUDA_ARCHITECTURES="${arch}" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc)
acados=()
acados_dir="${NAV_ACADOS_DIR:-/opt/acados/cmake}"
[[ -d "${acados_dir}" ]] && acados=(-Dacados_DIR="${acados_dir}")

# a package found in any prefix already on AMENT_PREFIX_PATH (the system's ROS, a deps prefix)
have_pkg() {
  local p
  IFS=: read -r -a ps <<< "${AMENT_PREFIX_PATH:-/opt/ros/humble}:/opt/ros/humble"
  for p in "${ps[@]}"; do [[ -n "${p}" && -d "${p}/share/$1" ]] && return 0; done
  return 1
}

# stage <module> <base paths, ':'-separated> <all | up-to:PKG | PKG,PKG,...> [cmake args...]
stage() {
  local m=$1 bases=$2 sel=$3
  shift 3
  local base_args=() pick=() b
  IFS=: read -r -a bs <<< "${bases}"
  for b in "${bs[@]}"; do base_args+=("${b}"); done
  case "${sel}" in
    all) ;;
    up-to:*) pick=(--packages-up-to "${sel#up-to:}") ;;
    *) IFS=, read -r -a ps <<< "${sel}"; pick=(--packages-select "${ps[@]}") ;;
  esac
  local cmd=(colcon --log-base "${root}/log/${m}" build --base-paths "${base_args[@]}" "${pick[@]}"
             --build-base "${root}/build/${m}" --install-base "${root}/install/${m}"
             --parallel-workers "${NAV_WORKERS:-4}" --cmake-args "${common[@]}" "$@")
  echo "== ${m}: ${sel} (from ${bases//${repo}\//})"
  if [[ -n "${NAV_DRY_RUN:-}" ]]; then
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi
  (
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    # shellcheck disable=SC1090
    [[ -f "${root}/install/${m}/setup.bash" ]] && source "${root}/install/${m}/setup.bash"
    set -u
    export MAKEFLAGS="-j${jobs}"
    "${cmd[@]}"
  )
}

for m in "${modules[@]}"; do
  case "${m}" in
    planning)
      stage planning "${nav}/seres_interface" \
        nav2_interface,common_interface,mc_interface,perception_interface,rte_interface,rhr_interface
      stage planning "${nav}/planning/third_party" nav_interface
      stage planning "${nav}/planning" planning "${cuda[@]}" "${acados[@]}" ;;
    slam)
      stage slam "${nav}/seres_interface" kygi665_msgs,rhr_interface,nav2_interface,common_interface
      stage slam "${nav}/slam" novatel_oem7_msgs,basic,livox_ros_driver2
      stage slam "${nav}/slam" lightning ;;
    nav_perception)
      # their build compiles pcl_msgs / octomap_msgs from third_party; only when no prefix has them
      if ! have_pkg octomap_msgs || ! have_pkg pcl_msgs; then
        stage nav_perception "${nav}/perception_combo_ros/third_party/octomap_related" pcl_msgs,octomap_msgs
      fi
      stage nav_perception "${nav}/perception_combo_ros/common_interface" all
      stage nav_perception "${nav}/perception_combo_ros/node_src" lidar_seg,grid_map_memory_node ;;
    perception)
      stage perception \
        "${dd}/perception_combo_ros/common_interface:${dd}/slam/interface:${dd}/perception_combo_ros/node_src" \
        up-to:lidar_3d_detection "${cuda[@]}" ;;
    *) echo "unknown module ${m} (planning, slam, nav_perception, perception)"; exit 2 ;;
  esac
done
echo "done: ${root}/install/{$(IFS=,; echo "${modules[*]}")}"
