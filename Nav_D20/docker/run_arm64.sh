#!/bin/bash
# Run a command in dog 8's root filesystem (the image from dog_rootfs.sh) on the server, under QEMU (T60): builds
# there use exactly the dog's compilers, libraries, ROS, CUDA and TensorRT. The work directory is mounted at the
# dog's paths, /opt/seres/nav_cw/{src,build,install,log}, so what it installs can go to the dog as it is (setup.bash
# files and CMake configs carry absolute paths); deps (/opt/seres/nav_cw/deps) are the dog's, from the image.
# No network: a build that wants to download something fails here instead of fetching what the dog does not have.
# The environment is the one of T56's build on the dog (docs/SERVER_TASKS.md, T56 step 3).
#
# One container is kept and reused (docker exec): on this server's vfs storage driver every `docker run` copies the
# 22 GB image first, about 50 minutes (T60); an exec into the kept container starts at once. It is created on first
# use, as root opens the dog's osqp / qdldl CMake files (mode 0700, owned by the dog's nvidia: find_package could
# not read them, T60), and every command then runs as the caller.
#
#   bash Nav_D20/docker/run_arm64.sh <work dir> <command...>
#   e.g. bash Nav_D20/docker/run_arm64.sh work_dirs/nav_cw_arm64 \
#            bash /opt/seres/nav_cw/src/Nav_D20/docker/build_stack.sh /opt/seres/nav_cw nav_perception slam planning
#   bash Nav_D20/docker/run_arm64.sh <work dir> --stop     remove the kept container (a new image needs a new one)
#   NAV_ARM64_IMAGE (default dog8-rootfs:latest), NAV_ARM64_NAME (the container, default nav-arm64-<user>),
#   NAV_JOBS (make -j, default nproc), NAV_WORKERS (colcon, 4), NAV_DOCKER (default docker),
#   NAV_DRY_RUN=1 prints the docker commands.
#   QEMU has to be registered once per boot of the server:
#     docker run --privileged --rm tonistiigi/binfmt --install arm64
set -euo pipefail
work="${1:?usage: bash Nav_D20/docker/run_arm64.sh <work dir> <command...> | --stop}"
shift
[[ $# -gt 0 ]] || set -- bash
DOCKER="${NAV_DOCKER:-docker}"
image="${NAV_ARM64_IMAGE:-dog8-rootfs:latest}"
name="${NAV_ARM64_NAME:-nav-arm64-$(id -un)}"
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
mounts=()
for d in src build install log; do mounts+=(-v "${work}/${d}:/opt/seres/nav_cw/${d}"); done
create=("${DOCKER}" run -d --name "${name}" --platform linux/arm64 --network none
        --tmpfs /tmp:exec,mode=1777 --tmpfs /var/tmp:exec,mode=1777 --tmpfs /run
        "${mounts[@]}" -w /opt/seres/nav_cw "${image}" sleep infinity)
open_cmake=("${DOCKER}" exec -u 0 "${name}" chmod -R a+rX "${D}/lib/cmake/osqp" "${D}/lib/cmake/qdldl")
tty=()
[[ -t 0 && -t 1 ]] && tty=(-it)
run=("${DOCKER}" exec "${tty[@]}" -u "$(id -u):$(id -g)" -w /opt/seres/nav_cw "${env_args[@]}" "${name}" "$@")

if [[ "$1" == --stop ]]; then
  [[ -n "${NAV_DRY_RUN:-}" ]] && { echo "${DOCKER} rm -f ${name}"; exit 0; }
  exec "${DOCKER}" rm -f "${name}"
fi
if [[ -n "${NAV_DRY_RUN:-}" ]]; then
  echo "# when ${name} does not exist yet:"
  printf '%q ' "${create[@]}"; echo
  printf '%q ' "${open_cmake[@]}"; echo
  printf '%q ' "${run[@]}"; echo
  exit 0
fi

# a missing container makes docker inspect exit 1 after a blank line on stdout; that line is not "missing"
raw="$("${DOCKER}" inspect -f '{{.State.Running}}' "${name}" 2> /dev/null || true)"
state="${raw//[[:space:]]/}"
[[ -n "${state}" ]] || state=missing
if [[ "${state}" == missing ]]; then
  echo "creating ${name} from ${image} (on vfs the first one takes long, T60)..."
  "${create[@]}" > /dev/null
  "${open_cmake[@]}" || echo "warning: could not open ${D}/lib/cmake/osqp, qdldl (planning's find_package(osqp) may fail)"
elif [[ "${state}" != true ]]; then
  "${DOCKER}" start "${name}" > /dev/null
fi
# the kept container has to be the one for this work directory
have="$("${DOCKER}" inspect -f '{{range .Mounts}}{{.Source}}={{.Destination}};{{end}}' "${name}")"
if [[ "${have}" != *"${work}/src=/opt/seres/nav_cw/src;"* ]]; then
  echo "${name} mounts another work directory (${have}); NAV_ARM64_NAME=<another name>, or --stop it first"
  exit 2
fi
exec "${run[@]}"
