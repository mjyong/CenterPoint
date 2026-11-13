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
