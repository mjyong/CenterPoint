#!/bin/bash
# One command to record a bag on dog 8: the field guide's topics (DOG_FIELD_GUIDE.md section 9), the planner's goal
# and /mc/state, sqlite3, as root (the robot's shared memory is root's, T40). It stops by itself 5 s after the dog
# (walking or remote-controlled) has come within 1.5 m of the planner's goal; otherwise Ctrl-C. Either way the
# recorder gets one SIGINT and writes metadata.yaml. The logic is nav_cw_record.py, next to this file.
#
#   sudo bash /opt/seres/nav_cw/nav_cw_record.sh                 -> /home/nvidia/bags/<YYYYMMDD>_cw_<n>
#   sudo bash /opt/seres/nav_cw/nav_cw_record.sh -o /home/nvidia/bags/<name> [--radius 1.5] [--hold 5] [--no-auto]
#                                                [more topics ...]   (--dry-run prints the record command)
#   NAV_CW_PREFIX (/opt/seres/nav_cw: our installs give the message types), NAV_CW_BAG_DIR (/home/nvidia/bags).
# Domain: ROS_DOMAIN_ID if set, else nav_cw.env's (42 during a rehearsal), else 0. It only subscribes.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
prefix="${NAV_CW_PREFIX:-/opt/seres/nav_cw}"
if [[ "$(id -u)" != 0 && -z "${NAV_CW_RECORD_ANY_USER:-}" ]]; then
  echo "run as root (the robot's shared memory is root's): sudo bash $0 $*"
  exit 2
fi
# the ROS setup files are not written for set -u (T57: AMENT_TRACE_SETUP_FILES unbound); planning last, or its
# mc_interface type support is shadowed and the recording fails (T57)
set +u
# shellcheck disable=SC1090
source "${NAV_CW_ROS_SETUP:-/opt/ros/humble/setup.bash}"
for p in slam nav_perception planning; do
  # shellcheck disable=SC1090
  [[ -f "${prefix}/install/${p}/setup.bash" ]] && source "${prefix}/install/${p}/setup.bash"
done
set -u
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///opt/ros/cyclonedds.xml}"
if [[ -z "${ROS_DOMAIN_ID:-}" && -f "${prefix}/nav_cw.env" ]]; then
  d="$(sed -n 's/^ROS_DOMAIN_ID="\(.*\)"$/\1/p' "${prefix}/nav_cw.env" | tail -1)"
  [[ -n "${d}" ]] && export ROS_DOMAIN_ID="${d}"
fi
exec python3 "${here}/nav_cw_record.py" "$@"
