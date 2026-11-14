#!/bin/bash
# A copy of dog 8's root filesystem as an arm64 docker image (T60): on the server, under QEMU (run_arm64.sh), our
# stack builds with exactly the dog's compilers, libraries, ROS, CUDA and TensorRT, without loading the dog.
# Read-only on the dog: one tar of / streamed over ssh (sudo -n, lowest CPU and IO priority).
# Left out: what is not the system (/proc, /sys, /dev, /run, /tmp, other filesystems such as /userdata), the robot
# team's installs under /opt/seres (of ours only /opt/seres/nav_cw/deps, the libraries T56 built), homes, logs,
# caches, kernel modules and sources, and secrets (password hashes, ssh host keys, Wi-Fi and network connections).
#
#   bash Nav_D20/docker/dog_rootfs.sh <dir> [tag]          on the server; `ssh dog` must work
#     <dir>/dog8_rootfs_<tag>.tar (tag default: today, e.g. 20261115), its listing and the dog's versions,
#     then docker import --platform linux/arm64 -> image dog8-rootfs:<tag>, also tagged dog8-rootfs:latest
#   NAV_DOG (ssh host, default dog), NAV_SSH (default ssh), NAV_DOCKER (default docker), NAV_DRY_RUN=1 prints what it
#   would run.
set -euo pipefail
dir="${1:?usage: bash Nav_D20/docker/dog_rootfs.sh <dir> [tag]}"
tag="${2:-$(date +%Y%m%d)}"
dog="${NAV_DOG:-dog}"
SSH="${NAV_SSH:-ssh}"
DOCKER="${NAV_DOCKER:-docker}"
image="dog8-rootfs:${tag}"
tarball="${dir%/}/dog8_rootfs_${tag}.tar"

EXCLUDE=(
  # not the system, or other filesystems (tar --one-file-system too)
  ./proc ./sys ./dev ./run ./tmp ./var/tmp ./mnt ./media ./lost+found ./boot ./swapfile ./snap ./userdata ./data
  # people's files, logs, caches, kernel modules and sources: nothing a build needs
  ./home ./root ./var/log ./var/cache ./var/crash ./var/backups ./var/lib/apt/lists ./var/lib/docker
  ./var/lib/containerd ./var/lib/snapd ./usr/lib/modules ./usr/src
  # secrets
  ./etc/shadow ./etc/shadow- ./etc/gshadow ./etc/gshadow- './etc/ssh/ssh_host_*' ./etc/NetworkManager/system-connections
  ./etc/netplan ./etc/wpa_supplicant ./var/lib/NetworkManager ./var/lib/iwd
)
# what must not be in the tar (checked on the listing afterwards)
SECRETS='^\./etc/(shadow|gshadow)-?$|^\./etc/ssh/ssh_host_|^\./etc/NetworkManager/system-connections/.|^\./etc/netplan/.|^\./etc/wpa_supplicant/.'

# /opt/seres: only our deps (the robot team's installs, and our own sources, build trees, installs and maps, stay)
listing="$("${SSH}" "${dog}" 'ls -A /opt/seres; echo --; ls -A /opt/seres/nav_cw')"
in_nav=0
while read -r e; do
  [[ -z "${e}" ]] && continue
  if [[ "${e}" == -- ]]; then in_nav=1; continue; fi
  if (( in_nav )); then
    [[ "${e}" == deps ]] || EXCLUDE+=("./opt/seres/nav_cw/${e}")
  else
    [[ "${e}" == nav_cw ]] || EXCLUDE+=("./opt/seres/${e}")
  fi
done <<< "${listing}"
(( in_nav )) || { echo "no /opt/seres/nav_cw on ${dog} (T56)"; exit 1; }

args=()
for e in "${EXCLUDE[@]}"; do args+=("--exclude=${e}"); done
remote="sudo -n nice -n 19 ionice -c3 tar -C / --one-file-system --numeric-owner -cpf - \
--warning=no-file-changed --warning=no-file-removed --warning=no-file-ignored $(printf '%q ' "${args[@]}") ."
versions='cat /etc/nv_tegra_release; lsb_release -ds; uname -r; gcc --version | head -1; \
/usr/local/cuda/bin/nvcc --version | tail -1; dpkg-query -W libnvinfer10 ros-humble-rclcpp 2>/dev/null; \
echo "dpkg packages: $(dpkg-query -W | wc -l), sha256 of the list: $(dpkg-query -W | sha256sum | cut -c1-16)"'
import=("${DOCKER}" import --platform linux/arm64
        --change 'ENV PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
        --change 'ENV LANG=C.UTF-8' --change 'CMD ["/bin/bash"]' "${tarball}" "${image}")

if [[ -n "${NAV_DRY_RUN:-}" ]]; then
  echo "${SSH} -C ${dog} ${remote} > ${tarball}"
  printf '%q ' "${import[@]}"
  echo
  exit 0
fi

mkdir -p "${dir}"
"${SSH}" "${dog}" "${versions}" | tee "${dir%/}/dog8_rootfs_${tag}.versions.txt"
echo "sizes on the dog, before the excludes:"
"${SSH}" "${dog}" 'sudo -n du -xsh /etc /opt /usr /var 2> /dev/null; sudo -n du -xsh /opt/* 2> /dev/null' || true
echo "tar of the dog's / -> ${tarball}"
set +e
"${SSH}" -C "${dog}" "${remote}" > "${tarball}.part"
rc=$?
set -e
# tar exits 1 when a file changed while it was read (a live system: logs, the robot's state files)
(( rc <= 1 )) || { echo "tar on the dog failed (${rc}); ${tarball}.part kept"; exit 1; }
mv "${tarball}.part" "${tarball}"
tar -tvf "${tarball}" > "${tarball%.tar}.list.txt"
if tar -tf "${tarball}" | grep -Eq "${SECRETS}"; then
  tar -tf "${tarball}" | grep -E "${SECRETS}" | head
  rm -f "${tarball}"
  echo "secrets in the tar (above): deleted it"
  exit 1
fi
du -h "${tarball}"
sha256sum "${tarball}" | tee "${tarball}.sha256"
"${import[@]}"
"${DOCKER}" tag "${image}" dog8-rootfs:latest
echo "image ${image} (dog8-rootfs:latest)"
