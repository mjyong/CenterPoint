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
# /opt/seres/nav_cw), which our units read (EnvironmentFile), and written into the installed configs where the
# programs read them there (apply_switches); each restarts only the unit it concerns, if it runs:
#   sudo bash nav_cw.sh shadow on|off       the planner runs but its commands go to /nav_cw/shadow/cmd_vel and
#                                           /nav_cw/shadow/mc/motion_control: the dog does not move by them; walk
#                                           it with the remote and compare. The installed communication_conf's
#                                           pub_cmd_vel / pub_motion names (the planner ignores --ros-args: it
#                                           calls rclcpp::init(0, nullptr)); restarts cw-planning and checks in the
#                                           ROS graph who publishes what (if the planner still publishes
#                                           /mc/motion_control with shadow on, it stops cw-planning, exit 3)
#   sudo bash nav_cw.sh appmap follow|ignore   follow: our planning and localization switch maps when the App
#                                           sends /map/switch (latched: its last one arrives at start; the App's
#                                           current map has to be the repository's). ignore: they listen on
#                                           /nav_cw/map/switch instead and stay on the site map configure set
#                                           (planning: communication_conf sub_map_switch; localization: the launch
#                                           argument map_switch_topic, NAV_CW_SLAM_ARGS); restarts both, checks
#   sudo bash nav_cw.sh check               who publishes the dog's commands and who listens to /map/switch, against
#                                           the switches (use-new runs it too)
#   sudo bash nav_cw.sh apply               write the switches kept in nav_cw.env into the installed configs again
#                                           (after a reinstall; use-new does it before it starts ours)
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
SHADOW_CMD=/nav_cw/shadow/cmd_vel
SHADOW_MC=/nav_cw/shadow/mc/motion_control
APP_SWITCH=/map/switch
OWN_SWITCH=/nav_cw/map/switch
VERIFY_WAIT="${NAV_CW_VERIFY_WAIT:-20}"   # [s] for the restarted node to show up in the ROS graph
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
  # lightning's launch reads the share copy (get_package_share_directory). The install also has one under lib/.
  loc="$(find "${base}" -path '*lightning/share/*/config/default_d20.yaml' -not -path '*/src/*' | head -1)"
  [[ -n "${loc}" ]] || loc="$(find "${base}" -path '*lightning/config/default_d20.yaml' -not -path '*/src/*' | head -1)"
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

# cap_speed <conf> <m/s | off>: MPPI's max_vx and cruise reference_velocity; the first cap keeps the installed file as
# <conf>.repo, off puts it back
cap_speed() {
  local conf=$1 v=$2
  if [[ "${v}" == off ]]; then
    if [[ -f "${conf}.repo" ]]; then mv "${conf}.repo" "${conf}"; fi
    return 0
  fi
  [[ -f "${conf}.repo" ]] || cp -p "${conf}" "${conf}.repo"
  # in the MPPI block: the first max_vx (dynamic_constraints) and reference_velocity_critic's reference_velocity
  awk -v v="${v}" '
      /mppi_planner_conf/ { mppi = 1 }
      mppi && !dmax && /^[[:space:]]*max_vx:/ { sub(/max_vx:.*/, "max_vx: " v); dmax = 1 }
      mppi && /reference_velocity_critic/ { rvc = 1 }
      rvc && !dref && /^[[:space:]]*reference_velocity:/ { sub(/reference_velocity:.*/, "reference_velocity: " v); dref = 1 }
      { print }' "${conf}" > "${conf}.tmp" && mv "${conf}.tmp" "${conf}"
}

speed() {   # speed <m/s> | off | show
  local conf
  conf="$(mission_conf)"
  [[ -n "${conf}" ]] || { echo "no installed mission_conf under ${ROOT}/install"; exit 1; }
  if [[ "${1:-show}" == off ]]; then
    env_set NAV_CW_SPEED ""
    cap_speed "${conf}" off
    restart_if_active cw-planning.service
  elif [[ "${1:-show}" != show ]]; then
    [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "speed: a number in m/s (e.g. 0.8), off or show"; exit 2; }
    env_set NAV_CW_SPEED "$1"
    cap_speed "${conf}" "$1"
    restart_if_active cw-planning.service
  fi
  awk '/mppi_planner_conf/ { m = 1 } m && /^[[:space:]]*max_vx:/ && !a { print "  MPPI " $0; a = 1 }
       m && /reference_velocity_critic/ { r = 1 } r && /^[[:space:]]*reference_velocity:/ && !b { print "  MPPI " $0; b = 1 }' \
    "${conf}" | sed 's/  */ /g'
}

comm_conf() {
  find "${ROOT}/install" -path '*robot_factory/d20/001/planning_conf/communication_conf.pb.txt' -not -path '*/src/*' \
    2> /dev/null | head -1
}
# topic_name <conf> <field>: the name: "..." of `field { name: "..." ... }`
topic_name() { sed -n -E "s/^[[:space:]]*$2[[:space:]]*\\{[[:space:]]*name:[[:space:]]*\"([^\"]*)\".*/\\1/p" "$1" | head -1; }
set_topic_name() {   # set_topic_name <conf> <field> <name>
  sed -i -E "s|^([[:space:]]*$2[[:space:]]*\\{[[:space:]]*name:[[:space:]]*)\"[^\"]*\"|\\1\"$3\"|" "$1"
  [[ "$(topic_name "$1" "$2")" == "$3" ]] || { echo "could not set $2 to $3 in $1"; return 1; }
}

# The switches are kept in nav_cw.env (NAV_CW_SHADOW=on, NAV_CW_APPMAP=ignore, NAV_CW_SPEED=<m/s>; not set: off,
# follow, no cap) and written into the installed configs here: shadow and appmap in communication_conf, the speed cap
# in mission_conf, appmap's launch argument for the localization in NAV_CW_SLAM_ARGS. A reinstall puts the
# repository's configs back; use-new (and apply) write the switches in again, so they hold across installs.
apply_switches() {
  local conf mission
  conf="$(comm_conf)"
  mission="$(mission_conf)"
  [[ -n "${conf}" && -n "${mission}" ]] || { echo "no installed communication_conf / mission_conf under ${ROOT}/install"; return 1; }
  if [[ "$(env_get NAV_CW_SHADOW)" == on ]]; then
    set_topic_name "${conf}" pub_cmd_vel "${SHADOW_CMD}" && set_topic_name "${conf}" pub_motion "${SHADOW_MC}" || return 1
  else
    set_topic_name "${conf}" pub_cmd_vel /cmd_vel && set_topic_name "${conf}" pub_motion /mc/motion_control || return 1
  fi
  if [[ "$(env_get NAV_CW_APPMAP)" == ignore ]]; then
    set_topic_name "${conf}" sub_map_switch "${OWN_SWITCH}" || return 1
    env_set NAV_CW_SLAM_ARGS "map_switch_topic:=${OWN_SWITCH}"
  else
    set_topic_name "${conf}" sub_map_switch "${APP_SWITCH}" || return 1
    env_set NAV_CW_SLAM_ARGS ""
  fi
  cap_speed "${mission}" "$(env_get NAV_CW_SPEED | grep . || echo off)"
}
shadow_state() {
  local conf; conf="$(comm_conf)"
  [[ -n "${conf}" ]] || { echo "no installed communication_conf"; return; }
  case "$(topic_name "${conf}" pub_motion)" in
    "${SHADOW_MC}") echo on ;;
    /mc/motion_control) echo off ;;
    *) echo "unknown (pub_motion $(topic_name "${conf}" pub_motion))" ;;
  esac
}
appmap_state() {
  local conf; conf="$(comm_conf)"
  [[ -n "${conf}" ]] || { echo "no installed communication_conf"; return; }
  case "$(topic_name "${conf}" sub_map_switch)" in
    "${OWN_SWITCH}") echo ignore ;;
    "${APP_SWITCH}") echo follow ;;
    *) echo "unknown (sub_map_switch $(topic_name "${conf}" sub_map_switch))" ;;
  esac
}

# endpoints <topic>: "PUBLISHER <node>" / "SUBSCRIPTION <node>" lines, from the ROS graph
endpoints() {
  "${ROS2}" topic info -v "$1" 2> /dev/null |
    awk '/^Node name:/ { n = $3 } /^Endpoint type:/ { print $3, n }'
}
# expect <yes|no> <PUBLISHER|SUBSCRIPTION> <node> <topic>
expect() {
  local found=no
  if endpoints "$4" | grep -qx "$2 $3"; then found=yes; fi
  if [[ "${found}" == "$1" ]]; then
    printf '  ok    %-12s %-17s %s %s\n' "$3" "$2" "$([[ $1 == yes ]] && echo on || echo "not on")" "$4"
    return 0
  fi
  printf '  WRONG %-12s %-17s %s %s\n' "$3" "$2" "$([[ $1 == yes ]] && echo "missing on" || echo "still on")" "$4"
  return 1
}
# the ROS graph against the switches; retried for VERIFY_WAIT s (a restarted node takes a moment to be discovered)
check() {
  ros_env
  local shadow appmap ok t0=${SECONDS}
  shadow="$(shadow_state)"
  appmap="$(appmap_state)"
  while :; do
    ok=0
    echo "graph (shadow ${shadow}, appmap ${appmap}):"
    if is_active cw-planning.service; then
      if [[ "${shadow}" == on ]]; then
        expect no PUBLISHER planning /mc/motion_control || ok=1
        expect no PUBLISHER planning /cmd_vel || ok=1
        expect yes PUBLISHER planning "${SHADOW_MC}" || ok=1
      else
        expect yes PUBLISHER planning /mc/motion_control || ok=1
      fi
      if [[ "${appmap}" == ignore ]]; then
        expect no SUBSCRIPTION planning "${APP_SWITCH}" || ok=1
      else
        expect yes SUBSCRIPTION planning "${APP_SWITCH}" || ok=1
      fi
    else
      echo "  cw-planning not running"
    fi
    if is_active cw-slam.service; then
      if [[ "${appmap}" == ignore ]]; then
        expect no SUBSCRIPTION mode_switch_node "${APP_SWITCH}" || ok=1
      else
        expect yes SUBSCRIPTION mode_switch_node "${APP_SWITCH}" || ok=1
      fi
    fi
    if (( ok == 0 )); then return 0; fi
    if (( SECONDS - t0 >= VERIFY_WAIT )); then return 1; fi
    sleep 2
  done
}
# check, and when shadow is on but the planner still publishes the dog's commands: stop it
check_or_stop() {
  if check; then return 0; fi
  if [[ "$(shadow_state)" == on ]] && endpoints /mc/motion_control | grep -qx "PUBLISHER planning"; then
    echo "SHADOW NOT IN EFFECT: the planner publishes /mc/motion_control. Stopping cw-planning."
    "${SYSTEMCTL}" stop cw-planning.service || true
    return 3
  fi
  echo "the graph does not match the switches (above)"
  return 1
}

switches() {
  local want_shadow want_appmap
  want_shadow="$(env_get NAV_CW_SHADOW | grep . || echo off)"
  want_appmap="$(env_get NAV_CW_APPMAP | grep . || echo follow)"
  echo "switches (${ENVFILE}; installed: $(comm_conf)):"
  echo "  shadow:           ${want_shadow}$([[ "${want_shadow}" == on ]] && echo " (commands to ${SHADOW_CMD}, ${SHADOW_MC})")"
  echo "  appmap:           ${want_appmap}$([[ "${want_appmap}" == ignore ]] && echo " (listening on ${OWN_SWITCH}; localization: $(env_get NAV_CW_SLAM_ARGS))")"
  if [[ "$(shadow_state)" != "${want_shadow}" || "$(appmap_state)" != "${want_appmap}" ]]; then
    echo "  NOT APPLIED: the installed config has shadow $(shadow_state), appmap $(appmap_state) (a reinstall?): nav_cw.sh apply"
  fi
  echo "  perception debug: $([[ -n "$(env_get LIDAR3D_ARGS)" ]] && echo "on ($(env_get LIDAR3D_ARGS))" || echo off)"
  local v; v="$(env_get NAV_CW_SPEED)"
  echo "  speed cap:        ${v:+${v} m/s}${v:-off (repository value)}"
}

case "${1:-status}" in
  shadow)
    case "${2:-}" in
      on) env_set NAV_CW_SHADOW on ;;
      off) env_set NAV_CW_SHADOW "" ;;
      *) echo "shadow on|off"; exit 2 ;;
    esac
    apply_switches || exit 1
    switches
    restart_if_active cw-planning.service
    if is_active cw-planning.service; then check_or_stop || exit $?; fi ;;
  appmap)
    case "${2:-}" in
      follow) env_set NAV_CW_APPMAP "" ;;
      ignore) env_set NAV_CW_APPMAP ignore ;;
      *) echo "appmap follow|ignore"; exit 2 ;;
    esac
    apply_switches || exit 1
    switches
    restart_if_active cw-slam.service
    restart_if_active cw-planning.service
    if is_active cw-planning.service || is_active cw-slam.service; then check_or_stop || exit $?; fi ;;
  check) check_or_stop || exit $? ;;
  apply)
    apply_switches || exit 1
    switches
    restart_if_active cw-slam.service
    restart_if_active cw-planning.service
    if is_active cw-planning.service || is_active cw-slam.service; then check_or_stop || exit $?; fi ;;
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
    apply_switches || exit 1      # the switches into the installed configs before ours start
    switches
    echo "switching to ours..."
    if switch "${ORIG[@]}" -- "${NEW[@]}"; then
      status
      check_or_stop || { echo "putting the robot team's back..."; switch "${NEW[@]}" -- "${ORIG[@]}" || true; status || true; exit 1; }
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
  *) sed -n '2,47p' "$0"; exit 2 ;;
esac
