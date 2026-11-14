#!/bin/bash
# Run a command in dog 8's root filesystem (the image from dog_rootfs.sh) on the server, under QEMU (T60): builds
# there use exactly the dog's compilers, libraries, ROS, CUDA and TensorRT. The work directory is mounted at the
# dog's paths, /opt/seres/nav_cw/{src,build,install,log}, so what it installs can go to the dog as it is (setup.bash
# files and CMake configs carry absolute paths); deps (/opt/seres/nav_cw/deps) are the dog's, from the image.
# No network: a build that wants to download something fails here instead of fetching what the dog does not have.
# The environment is the one of T56's build on the dog (docs/SERVER_TASKS.md, T56 step 3).
#
#   bash Nav_D20/docker/run_arm64.sh <work dir> <command...>
#   e.g. bash Nav_D20/docker/run_arm64.sh work_dirs/nav_cw_arm64 \
#            bash /opt/seres/nav_cw/src/Nav_D20/docker/build_stack.sh /opt/seres/nav_cw nav_perception slam planning
#   NAV_ARM64_IMAGE (default dog8-rootfs:latest), NAV_JOBS (make -j, default nproc), NAV_WORKERS (colcon, 4),
#   NAV_DRY_RUN=1 prints the docker command.
#   QEMU has to be registered once per boot of the server:
#     docker run --privileged --rm tonistiigi/binfmt --install arm64
set -euo pipefail
work="${1:?usage: bash Nav_D20/docker/run_arm64.sh <work dir> <command...>}"
shift
[[ $# -gt 0 ]] || set -- bash
mkdir -p "${work}"/{src,build,install,log}
work="$(cd "${work}" && pwd)"
D=/opt/seres/nav_cw/deps
envs=(
  HOME=/tmp
  "CMAKE_PREFIX_PATH=${D}/opt/ros/humble:${D}/usr:${D}"
  "AMENT_PREFIX_PATH=${D}/opt/ros/humble"
  "PKG_CONFIG_PATH=${D}/usr/lib/aarch64-linux-gnu/pkgconfig:${D}/usr/share/pkgconfig:${D}/lib/pkgconfig"
  "LD_LIBRARY_PATH=${D}/usr/lib/aarch64-linux-gnu:${D}/usr/lib:${D}/lib:${D}/opt/ros/humble/lib:${D}/opt/acados/lib:/usr/local/cuda/lib64"
  "PATH=${D}/usr/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  NAV_CUDA_ARCH=87
  "NAV_ACADOS_DIR=${D}/opt/acados/cmake"
  "NAV_JOBS=${NAV_JOBS:-$(nproc)}"
  "NAV_WORKERS=${NAV_WORKERS:-4}"
)
env_args=()
for e in "${envs[@]}"; do env_args+=(-e "${e}"); done
tty=()
[[ -t 0 && -t 1 ]] && tty=(-it)
cmd=(docker run --rm "${tty[@]}" --platform linux/arm64 --network none
     -u "$(id -u):$(id -g)" --tmpfs /tmp:exec,mode=1777 --tmpfs /var/tmp:exec,mode=1777 --tmpfs /run
     -v "${work}/src:/opt/seres/nav_cw/src" -v "${work}/build:/opt/seres/nav_cw/build"
     -v "${work}/install:/opt/seres/nav_cw/install" -v "${work}/log:/opt/seres/nav_cw/log"
     -w /opt/seres/nav_cw "${env_args[@]}" "${NAV_ARM64_IMAGE:-dog8-rootfs:latest}" "$@")
if [[ -n "${NAV_DRY_RUN:-}" ]]; then
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi
exec "${cmd[@]}"
