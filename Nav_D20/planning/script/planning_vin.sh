#!/bin/bash
# 规划按 $PLANNING_ROOT_DIR/robot_factory/default_robot_vin 选车型。机器人团队新版的默认是 d30/001（T48 发现：
# 用 d30 的 788 x 497 地图，0921 的终点在图外，任务进 fault）。我们的狗是 D20。
#   planning_root_for_vin <装好的 PLANNING_ROOT_DIR> <车型，如 d20/001> <临时目录>
# 车型和装好的一样就原样返回；不一样就在临时目录搭一个只换掉 default_robot_vin 的 PLANNING_ROOT_DIR（其余都是软链接），
# 打印它的路径。不改安装目录，也不改源码里的 default_robot_vin。
planning_root_for_vin() {
  local root="$1" vin="$2" out="$3" e
  if [[ ! -d "${root}/robot_factory/${vin}" ]]; then
    echo "没有这个车型：${root}/robot_factory/${vin}" >&2
    return 1
  fi
  if [[ "$(tr -d '[:space:]' < "${root}/robot_factory/default_robot_vin")" == "${vin}" ]]; then
    echo "${root}"
    return 0
  fi
  rm -rf "${out}"
  mkdir -p "${out}/robot_factory"
  for e in "${root}"/*; do
    [[ "$(basename "${e}")" == robot_factory ]] || ln -s "${e}" "${out}/"
  done
  for e in "${root}"/robot_factory/*; do
    [[ "$(basename "${e}")" == default_robot_vin ]] || ln -s "${e}" "${out}/robot_factory/"
  done
  echo "${vin}" > "${out}/robot_factory/default_robot_vin"
  echo "${out}"
}

# planning_root_with_startup_map <PLANNING_ROOT_DIR> <车型> <站点地图目录> <临时目录>
# 搭一个 PLANNING_ROOT_DIR：车型目录拷一份，它的 preprocess_conf 里 global_map_manager_conf 加上 startup_map_dir
# （我们的补丁，T53 复核：规划起来就读这张图和它的 semantic.osm，不等 /map/switch），其余都是软链接；打印路径。
# 狗上的安装用 tools/deploy/nav_cw.sh configure，改的是同一项。
planning_root_with_startup_map() {
  local root="$1" vin="$2" map="${3%/}" out="$4" e type="${2%%/*}" id="${2##*/}"
  if [[ ! -f "${map}/global_map.yaml" ]]; then
    echo "不是站点地图目录（没有 global_map.yaml）：${map}" >&2
    return 1
  fi
  rm -rf "${out}"
  mkdir -p "${out}/robot_factory/${type}"
  for e in "${root}"/*; do
    [[ "$(basename "${e}")" == robot_factory ]] || ln -s "$(readlink -f "${e}")" "${out}/"
  done
  for e in "${root}"/robot_factory/*; do
    case "$(basename "${e}")" in default_robot_vin|"${type}") ;; *) ln -s "$(readlink -f "${e}")" "${out}/robot_factory/" ;; esac
  done
  for e in "${root}/robot_factory/${type}"/*; do
    [[ "$(basename "${e}")" == "${id}" ]] || ln -s "$(readlink -f "${e}")" "${out}/robot_factory/${type}/"
  done
  cp -rL "${root}/robot_factory/${vin}" "${out}/robot_factory/${vin}"
  local conf="${out}/robot_factory/${vin}/planning_conf/preprocess_conf.pb.txt"
  awk -v map="${map}" '
    /^[[:space:]]*startup_map_dir:/ { next }
    { print }
    /global_map_manager_conf[[:space:]]*\{/ { print "        startup_map_dir: \"" map "\"" }
  ' "${conf}" > "${conf}.tmp" && mv "${conf}.tmp" "${conf}"
  echo "${vin}" > "${out}/robot_factory/default_robot_vin"
  echo "${out}"
}
