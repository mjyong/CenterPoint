#!/bin/bash
# Dog 8: navigation by the robot team's services (planning.service, slam.service, nav_perception.service, their
# files untouched; with our earlier lidar-3d-detection.service, the .deb) or by ours (/opt/seres/nav_cw: this
# repository's planning, localization and perception, our lidar_3d_detection inside the perception;
# cw-planning.service, cw-slam.service, cw-nav-perception.service). One command each way. Ours runs the node in
# cw-nav-perception, so the .deb's service stops with theirs (two nodes would publish the same objects).
#
#   sudo bash nav_cw.sh status          which set is running
#   sudo bash nav_cw.sh use-new         stop theirs, start ours; if one of ours is not up and steady after
#                                       NAV_CW_WAIT s (default 15), put theirs back and exit 1
#   sudo bash nav_cw.sh use-original    stop ours, start theirs
#   bash nav_cw.sh configure <prefix> <map dir>
#                                       at install time (T56): point our localization (map_path) and planning
#                                       (startup_map_dir) at the site map, vehicle type d20/001; the planner's debug
#                                       topics on (pub_flag, enable_global_map_pub: Foxglove, nav_cw_foxglove.sh;
#                                       NAV_CW_DEBUG=0 keeps the repository's off)
#
# Switches for debugging on the dog (docs/DOG_FIELD_GUIDE.md). Kept in <root>/nav_cw.env (NAV_CW_ROOT, default
# /opt/seres/nav_cw), which our units read (EnvironmentFile); each restarts only the unit it concerns, if it runs:
#   sudo bash nav_cw.sh shadow on|off       the planner runs but its commands go to /nav_cw/shadow/cmd_vel (and
#                                           /nav_cw/shadow/mc/motion_control): the dog does not move by them; walk
#                                           it with the remote and compare (restarts cw-planning)
#   sudo bash nav_cw.sh debug planning on|off     the planner's debug topics, at once (/planning/debug/set_pub_flag)
#   sudo bash nav_cw.sh debug perception on|off   lidar_3d_detection's debug cloud and its TF camera_init ->
#                                           lidar_det (restarts cw-nav-perception: the local grid stops a few s)
#   sudo bash nav_cw.sh speed <m/s>|off|show  cap the planner: MPPI max_vx and its cruise reference_velocity in the
#                                           installed mission_conf (the first cap keeps the installed file as
#                                           mission_conf.pb.txt.repo; off puts it back; restarts cw-planning)
#   bash nav_cw.sh switches                 what is set
#
# Nothing is enabled or disabled: after a reboot the robot team's services start (they are enabled, ours are not).
# Their units are not edited; ours carry Conflicts= on theirs, so systemd never runs both.
set -euo pipefail

ORIG=(lidar-3d-detection.service nav_perception.service slam.service planning.service)   # start order; stop in reverse
NEW=(cw-nav-perception.service cw-slam.service cw-planning.service)
WAIT="${NAV_CW_WAIT:-15}"
ROOT="${NAV_CW_ROOT:-/opt/seres/nav_cw}"
ENVFILE="${ROOT}/nav_cw.env"
ROS2="${ROS2:-ros2}"                  # the tests put a fake one here
SHADOW_ARGS="--ros-args -r /cmd_vel:=/nav_cw/shadow/cmd_vel -r /mc/motion_control:=/nav_cw/shadow/mc/motion_control"
PERCEPTION_DEBUG_ARGS="-p publish_cloud:=true -p publish_tf:=true"
SYSTEMCTL="${SYSTEMCTL:-systemctl}"   # the tests put a fake one here
JOURNALCTL="${JOURNALCTL:-journalctl}"

is_active() { "${SYSTEMCTL}" is-active --quiet "$1"; }

show() {
  local u
  for u in "$@"; do
    printf '  %-28s %s\n' "${u}" "$("${SYSTEMCTL}" is-active "${u}" 2>/dev/null || true)"
  done
}

running() {   # -> theirs / ours / mixed
  local o=0 n=0 u
  for u in "${ORIG[@]}"; do if is_active "${u}"; then o=$((o + 1)); fi; done
  for u in "${NEW[@]}"; do if is_active "${u}"; then n=$((n + 1)); fi; done
  if (( n == ${#NEW[@]} && o == 0 )); then echo ours
  elif (( o == ${#ORIG[@]} && n == 0 )); then echo theirs
  else echo "mixed (${o} of theirs, ${n} of ours active)"
  fi
}

status() {
  echo "robot team's:"; show "${ORIG[@]}"
  echo "ours (/opt/seres/nav_cw):"; show "${NEW[@]}"
  local r; r="$(running)"
  echo "running: ${r}"
  [[ "${r}" == ours || "${r}" == theirs ]]
}

restarts() { "${SYSTEMCTL}" show -p NRestarts --value "$1" 2>/dev/null || echo 0; }

# switch <from units...> -- <to units...>: stop `from` (reverse order), start `to` (in order), then the `to` units
# have to stay active for WAIT s without a restart
switch() {
  local from=() to=() u i seen=0
  for u in "$@"; do
    if [[ "${u}" == -- ]]; then seen=1; continue; fi
    if (( seen )); then to+=("${u}"); else from+=("${u}"); fi
  done
  for (( i = ${#from[@]} - 1; i >= 0; i-- )); do "${SYSTEMCTL}" stop "${from[i]}" || true; done
  declare -A before
  for u in "${to[@]}"; do
    before[${u}]="$(restarts "${u}")"
    "${SYSTEMCTL}" start "${u}" || { echo "start ${u} failed"; return 1; }
  done
  sleep "${WAIT}"
  for u in "${to[@]}"; do
    if ! is_active "${u}"; then echo "${u} is not active after ${WAIT} s"; return 1; fi
    if [[ "$(restarts "${u}")" != "${before[${u}]}" ]]; then echo "${u} restarted within ${WAIT} s"; return 1; fi
  done
  for u in "${from[@]}"; do
    if is_active "${u}"; then echo "${u} still active"; return 1; fi
  done
}

need_units() {
  local u
  for u in "${NEW[@]}"; do
    "${SYSTEMCTL}" cat "${u}" > /dev/null 2>&1 || { echo "no ${u}: install /opt/seres/nav_cw first (T55)"; exit 1; }
  done
}

configure() {   # configure <prefix> <map dir>
  local prefix="$1" map="${2%/}" loc plan vin
  [[ -f "${map}/global_map.yaml" ]] || { echo "${map}: no global_map.yaml (a site map directory)"; exit 1; }
  # the installed copies (build_stack.sh: <prefix>/install/<module>), never the sources under <prefix>/src
  local base="${prefix}"
  [[ -d "${prefix}/install" ]] && base="${prefix}/install"
  loc="$(find "${base}" -path '*lightning/config/default_d20.yaml' -not -path '*/src/*' | head -1)"
  plan="$(find "${base}" -path '*robot_factory/d20/001/planning_conf/preprocess_conf.pb.txt' -not -path '*/src/*' | head -1)"
  vin="$(find "${base}" -path '*robot_factory/default_robot_vin' -not -path '*/src/*' | head -1)"
  comm="$(find "${base}" -path '*robot_factory/d20/001/planning_conf/communication_conf.pb.txt' -not -path '*/src/*' | head -1)"
  [[ -n "${loc}" && -n "${plan}" && -n "${vin}" ]] || { echo "${prefix}: lightning config, d20/001 planning_conf or default_robot_vin not found"; exit 1; }
  sed -i -E "s|^([[:space:]]*map_path:[[:space:]]*).*|\1${map}|" "${loc}"
  awk -v map="${map}" '
    /^[[:space:]]*startup_map_dir:/ { next }
    { print }
    /global_map_manager_conf[[:space:]]*\{/ { print "        startup_map_dir: \"" map "\"" }
  ' "${plan}" > "${plan}.tmp" && mv "${plan}.tmp" "${plan}"
  echo d20/001 > "${vin}"
  if [[ -n "${comm}" && "${NAV_CW_DEBUG:-1}" != 0 ]]; then
    sed -i -E 's/^([[:space:]]*(pub_flag|enable_global_map_pub):[[:space:]]*)false/\1true/' "${comm}"
    echo "debug topics: $(grep -E '^[[:space:]]*(pub_flag|enable_global_map_pub):' "${comm}" | tr -s ' ' | tr '\n' ' ')"
  fi
  echo "localization: $(grep -m1 -E '^[[:space:]]*map_path:' "${loc}" | sed 's/^ *//')  (${loc})"
  echo "planning:     $(grep -m1 startup_map_dir "${plan}" | sed 's/^ *//')  (${plan})"
  echo "vehicle:      $(cat "${vin}")"
}

# env_set KEY VALUE (empty VALUE: remove the key)
env_set() {
  local key=$1 value=$2
  mkdir -p "${ROOT}"
  touch "${ENVFILE}"
  grep -v "^${key}=" "${ENVFILE}" > "${ENVFILE}.tmp" || true
  [[ -n "${value}" ]] && printf '%s="%s"\n' "${key}" "${value}" >> "${ENVFILE}.tmp"
  mv "${ENVFILE}.tmp" "${ENVFILE}"
}
env_get() { [[ -f "${ENVFILE}" ]] && sed -n "s/^$1=\"\(.*\)\"$/\1/p" "${ENVFILE}" | tail -1; true; }
restart_if_active() {
  if is_active "$1"; then
    "${SYSTEMCTL}" restart "$1"
    echo "restarted $1"
  else
    echo "$1 not running: takes effect when it starts"
  fi
}

ros_env() {
  set +u
  # shellcheck disable=SC1091
  [[ -f /opt/ros/humble/setup.bash ]] && source /opt/ros/humble/setup.bash
  set -u
  export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
  export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///opt/ros/cyclonedds.xml}"
}

mission_conf() {
  find "${ROOT}/install" -path '*robot_factory/d20/001/planning_conf/mission_conf.pb.txt' -not -path '*/src/*' \
    2> /dev/null | head -1
}

speed() {   # speed <m/s> | off | show
  local conf
  conf="$(mission_conf)"
  [[ -n "${conf}" ]] || { echo "no installed mission_conf under ${ROOT}/install"; exit 1; }
  if [[ "${1:-show}" == off ]]; then
    [[ -f "${conf}.repo" ]] && mv "${conf}.repo" "${conf}"
    env_set NAV_CW_SPEED ""
    restart_if_active cw-planning.service
  elif [[ "${1:-show}" != show ]]; then
    [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "speed: a number in m/s (e.g. 0.8), off or show"; exit 2; }
    [[ -f "${conf}.repo" ]] || cp -p "${conf}" "${conf}.repo"      # the installed values, for speed off
    # in the MPPI block: the first max_vx (dynamic_constraints) and reference_velocity_critic's reference_velocity
    awk -v v="$1" '
      /mppi_planner_conf/ { mppi = 1 }
      mppi && !dmax && /^[[:space:]]*max_vx:/ { sub(/max_vx:.*/, "max_vx: " v); dmax = 1 }
      mppi && /reference_velocity_critic/ { rvc = 1 }
      rvc && !dref && /^[[:space:]]*reference_velocity:/ { sub(/reference_velocity:.*/, "reference_velocity: " v); dref = 1 }
      { print }' "${conf}" > "${conf}.tmp" && mv "${conf}.tmp" "${conf}"
    env_set NAV_CW_SPEED "$1"
    restart_if_active cw-planning.service
  fi
  awk '/mppi_planner_conf/ { m = 1 } m && /^[[:space:]]*max_vx:/ && !a { print "  MPPI " $0; a = 1 }
       m && /reference_velocity_critic/ { r = 1 } r && /^[[:space:]]*reference_velocity:/ && !b { print "  MPPI " $0; b = 1 }' \
    "${conf}" | sed 's/  */ /g'
}

switches() {
  echo "switches (${ENVFILE}):"
  echo "  shadow:           $([[ -n "$(env_get NAV_CW_PLANNING_ARGS)" ]] && echo "on (commands to /nav_cw/shadow/cmd_vel)" || echo off)"
  echo "  perception debug: $([[ -n "$(env_get LIDAR3D_ARGS)" ]] && echo "on ($(env_get LIDAR3D_ARGS))" || echo off)"
  local v; v="$(env_get NAV_CW_SPEED)"
  echo "  speed cap:        ${v:+${v} m/s}${v:-off (repository value)}"
}

case "${1:-status}" in
  shadow)
    case "${2:-}" in
      on) env_set NAV_CW_PLANNING_ARGS "${SHADOW_ARGS}" ;;
      off) env_set NAV_CW_PLANNING_ARGS "" ;;
      *) echo "shadow on|off"; exit 2 ;;
    esac
    restart_if_active cw-planning.service
    switches ;;
  debug)
    case "${2:-}/${3:-}" in
      planning/on | planning/off)
        ros_env
        "${ROS2}" service call /planning/debug/set_pub_flag std_srvs/srv/SetBool "{data: $([[ $3 == on ]] && echo true || echo false)}" ;;
      perception/on) env_set LIDAR3D_ARGS "${PERCEPTION_DEBUG_ARGS}"; restart_if_active cw-nav-perception.service; switches ;;
      perception/off) env_set LIDAR3D_ARGS ""; restart_if_active cw-nav-perception.service; switches ;;
      *) echo "debug planning|perception on|off"; exit 2 ;;
    esac ;;
  speed) shift; speed "$@" ;;
  switches) switches ;;
  status) status ;;
  use-new)
    need_units
    echo "switching to ours..."
    if switch "${ORIG[@]}" -- "${NEW[@]}"; then
      status
    else
      echo "ours did not come up: last log lines"
      for u in "${NEW[@]}"; do echo "== ${u}"; "${JOURNALCTL}" -u "${u}" -n 15 --no-pager 2>/dev/null || true; done
      echo "putting the robot team's back..."
      switch "${NEW[@]}" -- "${ORIG[@]}" || true
      status || true
      exit 1
    fi ;;
  use-original)
    echo "switching to the robot team's..."
    switch "${NEW[@]}" -- "${ORIG[@]}" || { status || true; exit 1; }
    status ;;
  configure) shift; configure "$@" ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
