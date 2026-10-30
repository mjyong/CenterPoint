#!/usr/bin/env bash
# Build the vehicle .deb on the Orin (README.md). colcon builds the interface packages and lidar_3d_detection
# (CUDA, TensorRT, this board's engines) into <work>/install; a copy is moved to its final prefix
# (/opt/seres/perception) and dpkg-deb packs it with the start script, the systemd unit and its settings.
#
#   node_src/lidar_3d_detection/scripts/build_deb.sh --model ~/dogperc/models/nusc_pp_dyn [--out DIR]
#       [--prefix /opt/seres/perception] [--version V] [--service-user USER]
#       [--precision fp16] [--cuda-arch 87] [--jobs N] [--work DIR] [--skip-build]
#       [--no-enable-service]
#
#   --model           model_meta.json, pfn.onnx, rpn_head.onnx and the <precision> engines built on this board
#                     (tools/deploy/build_engines.py). Engines only load on the TensorRT that built them, so the
#                     .deb depends on exactly this board's libnvinfer version
#   --service-user    User= of lidar-3d-detection.service (default: who runs this script)
#   --enable-service  postinst enables and starts the service (the default)
#   --no-enable-service  install the unit, do not enable or start it
#   --skip-build      package <work>/install as it is
#
# No root needed (dpkg-deb --root-owner-group); installing is `sudo apt install ./lidar-3d-detection_*.deb`.
# Needs patchelf (the interface packages' Python extensions carry an RPATH into the build tree; T31): on PATH, in
# ~/bin, or PATCHELF=. The version is package.xml's plus the commit count and hash, from git or, in a copy without
# .git, from the GIT_COMMIT file the sync scripts write next to perception_combo_ros ("count hash [dirty]").
set -euo pipefail

NODE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)        # node_src/lidar_3d_detection
WS=$(dirname "$(dirname "$NODE")")                               # perception_combo_ros
SLAM=$(dirname "$WS")/slam/interface                             # kygi665_msgs
MODEL="" OUT=$PWD PREFIX=/opt/seres/perception VERSION="" SERVICE_USER=$(id -un) ENABLE=yes PRECISION=fp16
CUDA_ARCH=87 JOBS=$(nproc) WORK=$WS/build_deb SKIP_BUILD=0
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$(readlink -f "$2"); shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --prefix) PREFIX=$2; shift 2 ;;
    --version) VERSION=$2; shift 2 ;;
    --service-user) SERVICE_USER=$2; shift 2 ;;
    --enable-service) ENABLE=yes; shift ;;
    --no-enable-service) ENABLE=no; shift ;;
    --precision) PRECISION=$2; shift 2 ;;
    --cuda-arch) CUDA_ARCH=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    *) sed -n '2,19p' "$0"; exit 2 ;;
  esac
done
die() { echo "build_deb.sh: $*" >&2; exit 1; }
case $PREFIX in /*) ;; *) die "--prefix must be absolute" ;; esac
mkdir -p "$OUT" "$WORK"
OUT=$(cd "$OUT" && pwd)
WORK=$(cd "$WORK" && pwd)
INSTALL=$WORK/install
command -v dpkg-deb > /dev/null || die "dpkg-deb not found"
PATCHELF=${PATCHELF:-$(command -v patchelf || echo "$HOME/bin/patchelf")}
[ -x "$PATCHELF" ] || die "patchelf not found (sudo apt-get install patchelf, or PATCHELF=...)"

# ---------------------------------------------------------------- build
if [ $SKIP_BUILD = 0 ]; then
  [ -n "$MODEL" ] || die "--model DIR: the model with the engines built on this board"
  for f in model_meta.json pfn.onnx rpn_head.onnx pfn.$PRECISION.engine rpn_head.$PRECISION.engine; do
    [ -e "$MODEL/$f" ] || die "$MODEL has no $f (build the engines on this board: tools/deploy/build_engines.py)"
  done
  [ -d "$SLAM/kygi665_msgs" ] || die "no kygi665_msgs in $SLAM"
  [ -e /opt/ros/humble/setup.bash ] || die "ROS 2 Humble not found in /opt/ros/humble"
  NVCC=${CUDACXX:-/usr/local/cuda/bin/nvcc}
  [ -x "$NVCC" ] || die "nvcc not found ($NVCC; set CUDACXX)"
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  command -v colcon > /dev/null || die "colcon not found"
  MAKEFLAGS=-j$JOBS colcon --log-base "$WORK/log" build \
    --base-paths "$WS/common_interface" "$SLAM" "$WS/node_src" \
    --build-base "$WORK/build" --install-base "$INSTALL" --packages-up-to lidar_3d_detection \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_C_FLAGS="-ffile-prefix-map=$WS/=" -DCMAKE_CXX_FLAGS="-ffile-prefix-map=$WS/=" \
    -DCMAKE_CUDA_FLAGS="-Xcompiler=-ffile-prefix-map=$WS/=" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DLIDAR3D_WITH_CUDA=ON -DLIDAR3D_KYGI=ON \
    -DLIDAR3D_MODEL_DIR="$MODEL" -DLIDAR3D_PRECISION="$PRECISION"
fi
EXE=$INSTALL/lidar_3d_detection/lib/lidar_3d_detection/lidar_3d_detection
[ -x "$EXE" ] || die "$EXE not built"
[ -e "$INSTALL/lidar_3d_detection/model/pfn.$PRECISION.engine" ] || die "no engines in $INSTALL/lidar_3d_detection/model"

# ---------------------------------------------------------------- stage: the install at its final prefix
PKG=lidar-3d-detection
if [ -z "$VERSION" ]; then
  VERSION=$(sed -n 's:.*<version>\(.*\)</version>.*:\1:p' "$NODE/package.xml" | head -1)
  # the commit count first, so that a later build is a higher version for apt
  if sha=$(git -C "$NODE" rev-parse --short HEAD 2> /dev/null); then
    count=$(git -C "$NODE" rev-list --count HEAD)
    dirty=$(git -C "$NODE" diff --quiet HEAD -- "$WS" "$SLAM" 2> /dev/null || echo dirty)
  elif [ -e "$(dirname "$WS")/GIT_COMMIT" ]; then
    read -r count sha dirty < "$(dirname "$WS")/GIT_COMMIT"
  fi
  if [ -n "${sha:-}" ]; then
    VERSION=$VERSION+$count.g$sha${dirty:+.dirty}
  fi
fi
ARCH=$(dpkg --print-architecture)
STAGE=$WORK/stage
rm -rf "$STAGE"
mkdir -p "$STAGE$PREFIX"
cp -a "$INSTALL/." "$STAGE$PREFIX/"
# bytecode keeps the build path in it; Python recompiles in memory
find "$STAGE$PREFIX" -name __pycache__ -type d -prune -exec rm -rf {} +
# text files (colcon and ament setup scripts, CMake configs) name the install path: rewrite it
esc=$(printf '%s' "$INSTALL" | sed 's/[][\.*^$]/\\&/g')
{ grep -rlIZF "$INSTALL" "$STAGE$PREFIX" || true; } | xargs -0 -r sed -i "s|$esc|$PREFIX|g"
# a binary that still has it or the build tree in its RPATH (rosidl's Python extensions): the install path becomes
# the prefix, build-tree entries go (the setup scripts' LD_LIBRARY_PATH finds the libraries)
mapfile -t left < <({ grep -rlF "$INSTALL" "$STAGE$PREFIX"; grep -rlF "$WS/" "$STAGE$PREFIX"; } | sort -u)
for f in "${left[@]}"; do
  rp=$("$PATCHELF" --print-rpath "$f" 2> /dev/null) || continue      # not an ELF file
  keep=()
  IFS=: read -ra parts <<< "$rp"
  for e in "${parts[@]}"; do
    e=${e//$INSTALL/$PREFIX}
    case $e in "$WS"/*) ;; *) keep+=("$e") ;; esac
  done
  new=$(IFS=:; echo "${keep[*]}")
  if [ "$new" != "$rp" ]; then "$PATCHELF" --set-rpath "$new" "$f"; fi
done
if grep -rlF "$INSTALL" "$STAGE$PREFIX"; then die "files above still have the build path $INSTALL"; fi
# the sources are not on the target either; a file that names them is worth a look (not an error)
mapfile -t src_refs < <(grep -rlF "$WS/" "$STAGE$PREFIX" || true)

sub() { sed -e "s|@PREFIX@|$PREFIX|g" -e "s|@USER@|$SERVICE_USER|g" -e "s|@ENABLE@|$ENABLE|g" "$1"; }
mkdir -p "$STAGE/lib/systemd/system" "$STAGE/etc/default" "$STAGE/DEBIAN"
sub "$NODE/packaging/lidar-3d-detection.service" > "$STAGE/lib/systemd/system/lidar-3d-detection.service"
sub "$NODE/packaging/lidar-3d-detection.default" > "$STAGE/etc/default/lidar-3d-detection"
for s in postinst prerm postrm; do
  sub "$NODE/packaging/debian/$s" > "$STAGE/DEBIAN/$s"
  chmod 755 "$STAGE/DEBIAN/$s"
done
cat > "$STAGE/DEBIAN/conffiles" << EOF
/etc/default/lidar-3d-detection
$PREFIX/lidar_3d_detection/config/lidar_3d_detection.yaml
EOF

# ---------------------------------------------------------------- Depends: who provides each library loaded
owner() {     # the package of a file (merged /usr: try both spellings, and the resolved path)
  local p q o
  for p in "$1" "$(readlink -f "$1")"; do
    for q in "$p" "${p#/usr}" "/usr$p"; do
      o=$(dpkg -S "$q" 2> /dev/null | grep -v '^diversion' | head -1 | cut -d: -f1) || true
      if [ -n "$o" ]; then echo "$o"; return; fi
    done
  done
  echo "?$1"
}
if [ -e /opt/ros/humble/setup.bash ]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi
libdirs=$(find "$STAGE$PREFIX" -name '*.so*' -printf '%h\n' | sort -u | tr '\n' ':')
mapfile -t owners < <(
  find "$STAGE$PREFIX" -type f \( -name '*.so*' -o -perm -u+x \) -print0 |
    while IFS= read -r -d '' f; do
      [ "$(od -An -tx1 -N4 "$f" | tr -d ' \n')" = 7f454c46 ] || continue     # ELF
      LD_LIBRARY_PATH="$libdirs${LD_LIBRARY_PATH:-}" ldd "$f" 2> /dev/null | awk '$2 == "=>" && $3 ~ /^\// {print $3}'
    done | sort -u | { grep -vF "$STAGE" || true; } | while read -r lib; do owner "$lib"; done | sort -u)
deps=(python3)    # the colcon setup scripts
# loaded with dlopen, so ldd does not list them: the DDS middleware of this board's ROS 2
for o in ros-humble-rmw-fastrtps-cpp ros-humble-rmw-cyclonedds-cpp; do
  if v=$(dpkg-query -W -f='${Version}' "$o" 2> /dev/null); then deps+=("$o (>= $v)"); break; fi
done
unowned=()
for o in "${owners[@]}"; do
  case $o in
    \?*) unowned+=("${o#?}"); continue ;;
  esac
  v=$(dpkg-query -W -f='${Version}' "$o" 2> /dev/null) || { deps+=("$o"); continue; }
  case $o in
    libnvinfer*) deps+=("$o (= $v)") ;;       # the engines' TensorRT
    *) deps+=("$o (>= $v)") ;;
  esac
done
recommends=()
dpkg-query -W ros-humble-ros2launch > /dev/null 2>&1 && recommends+=(ros-humble-ros2launch)
for o in ros-humble-ros2bag ros-humble-rosbag2-storage-default-plugins; do    # lidar_3d_bag_replay, ros2 bag
  dpkg-query -W "$o" > /dev/null 2>&1 && recommends+=("$o")
done
join() { local IFS=,; echo "$*" | sed 's/,/, /g'; }

MAINTAINER=$(sed -n 's:.*<maintainer email="\([^"]*\)">\([^<]*\)</maintainer>.*:\2 <\1>:p' "$NODE/package.xml" | head -1)
L4T=$(head -1 /etc/nv_tegra_release 2> /dev/null | sed 's/^# //; s/, *$//' || true)
SIZE=$(du -sk --exclude=DEBIAN "$STAGE" | cut -f1)
{
  echo "Package: $PKG"
  echo "Version: $VERSION"
  echo "Architecture: $ARCH"
  echo "Maintainer: ${MAINTAINER:-unknown}"
  echo "Section: misc"
  echo "Priority: optional"
  echo "Installed-Size: $SIZE"
  echo "Depends: $(join "${deps[@]}")"
  if [ ${#recommends[@]} -gt 0 ]; then echo "Recommends: $(join "${recommends[@]}")"; fi
  echo "Description: lidar 3D detection, tracking and prediction (ROS 2 node lidar_3d_detection)"
  echo " lidar_3d_detection and its interface packages (common_interface, nav2_interface,"
  echo " perception_interface, kygi665_msgs) installed in $PREFIX, with the model and the"
  echo " TensorRT $PRECISION engines built on $(hostname)${L4T:+ ($L4T)}, and the systemd unit"
  echo " lidar-3d-detection.service (settings in /etc/default/lidar-3d-detection)."
} > "$STAGE/DEBIAN/control"
chmod -R u+rwX,go+rX,go-w "$STAGE"
chmod 755 "$STAGE/DEBIAN"

DEB=$OUT/${PKG}_${VERSION}_${ARCH}.deb
dpkg-deb --root-owner-group --build "$STAGE" "$DEB" > /dev/null
REPORT=${DEB%.deb}.txt
{
  echo "package   $DEB"
  echo "built on  $(hostname) ${L4T:-}"
  echo "prefix    $PREFIX"
  echo "git       $(git -C "$NODE" rev-parse HEAD 2> /dev/null || echo none)"
  echo "model     $MODEL"
  (cd "$STAGE$PREFIX/lidar_3d_detection/model" && sha256sum -- *)
  echo
  dpkg-deb --info "$DEB" | sed -n '/Package:/,$p'
  if [ ${#unowned[@]} -gt 0 ]; then
    echo
    echo "libraries no installed package owns (not in Depends; the target needs them too):"
    printf '  %s\n' "${unowned[@]}"
  fi
  if [ ${#src_refs[@]} -gt 0 ]; then
    echo
    echo "installed files that name the source tree $WS (not on the target):"
    printf '  %s\n' "${src_refs[@]#$STAGE}"
  fi
} > "$REPORT"
cat "$REPORT"
echo
echo "wrote $DEB ($(du -h "$DEB" | cut -f1)), report $REPORT"
