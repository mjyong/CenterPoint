#!/usr/bin/env bash
# ExecStartPre of lidar-3d-detection.service (run as root, the unit's "+"): with LIDAR3D_LOCK_CLOCKS=1 in
# /etc/default/lidar-3d-detection, lock the Orin's clocks (jetson_clocks) before the node starts. Unlocked, the
# node is 2-3x slower (T32, T34), and a lock by hand is lost at every reboot.
#
# jetson_clocks sets the CPU, GPU and memory clocks of the whole machine to their maximum (and the fan): more
# power and heat for every process on the robot, so whether it is on is the robot team's call. Default 0: nothing
# is changed. Setting it back to 0 takes effect at the next reboot (this script does not unlock).
#
# Never stops the service: a missing or failing jetson_clocks is reported and the node starts anyway.
#   LIDAR3D_LOCK_CLOCKS   1 / yes / true: lock; anything else: leave the clocks as they are
#   JETSON_CLOCKS         the program (default jetson_clocks on PATH, else /usr/bin/jetson_clocks); for the tests
case "${LIDAR3D_LOCK_CLOCKS:-0}" in
  1 | yes | true) ;;
  *) exit 0 ;;
esac
jc=${JETSON_CLOCKS:-$(command -v jetson_clocks || echo /usr/bin/jetson_clocks)}
if [ ! -x "$jc" ]; then
  echo "lock_clocks.sh: LIDAR3D_LOCK_CLOCKS=$LIDAR3D_LOCK_CLOCKS but $jc not found; clocks left as they are" >&2
  exit 0
fi
if "$jc"; then
  echo "lock_clocks.sh: clocks locked ($jc)"
else
  echo "lock_clocks.sh: $jc failed (exit $?); clocks left as they are, the node starts anyway" >&2
fi
exit 0
