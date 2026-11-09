#!/bin/bash
# Sourced by run_loc_online_debug.sh and Nav_D20/planning/script/run_planning_stack_debug.sh (2026-11-09, T42 复核).
# 本机回放时不改仓库里的定位配置，用环境变量换掉两项：
#   LOC_MAP_PATH=<地图目录>  用这张定位地图（目录里要有 3d/index.txt 或 index.txt），不用配置里狗上的 /opt/seres 路径。
#                           仓库里的内置地图 0c3f8a2d-... 倾斜约 51°，不能用来定位（scripts/check_map_tilt.py）。
#   LOC_INS_ONLY=1           INS 直出（ins_only_mode）：不跑 LIO 和激光定位，直接发 INS 位姿；只在全程 RTK 固定解的包上用。
# loc_debug_config <配置>：两个都没设时打印原配置；设了就写一份临时配置（别的都不变）并打印它的路径。出错时返回 1。
loc_debug_config() {
  local config=$1 tmp
  if [[ -z "${LOC_MAP_PATH:-}" && "${LOC_INS_ONLY:-0}" != "1" ]]; then
    echo "${config}"
    return 0
  fi
  if [[ -n "${LOC_MAP_PATH:-}" && ! -f "${LOC_MAP_PATH}/3d/index.txt" && ! -f "${LOC_MAP_PATH}/index.txt" ]]; then
    echo "LOC_MAP_PATH=${LOC_MAP_PATH}: 没有 3d/index.txt 或 index.txt" >&2
    return 1
  fi
  tmp="$(mktemp --tmpdir loc_debug_XXXXXX.yaml)"
  cp "${config}" "${tmp}"
  local map="${LOC_MAP_PATH:-}"
  if [[ -z "${map}" ]]; then
    # loc_system.cc 在配置的地图无效时回退到 <配置目录>/../map/0c3f8a2d-...；临时配置不在原目录，写成绝对路径
    map="$(cd "$(dirname "${config}")/../map/0c3f8a2d-41eb-416d-8fca-aaad500e2be9" 2> /dev/null && pwd)"
    if ! grep -qE "^[[:space:]]*map_path:[[:space:]]*/opt/" "${config}" || [[ -f "$(sed -n 's/^[[:space:]]*map_path:[[:space:]]*//p' "${config}" | head -1)/3d/index.txt" ]]; then
      map=""                                      # the configured map exists or is not the dog's: keep it
    fi
  fi
  if [[ -n "${map}" ]]; then
    sed -i "s|^\([[:space:]]*map_path:\).*|\1 ${map}|" "${tmp}"
  fi
  if [[ "${LOC_INS_ONLY:-0}" == "1" ]]; then
    sed -i "s|^\([[:space:]]*ins_only_mode:\)[[:space:]]*[a-z]*|\1 true|" "${tmp}"
  fi
  echo "定位临时配置 ${tmp}：$(grep -E '^[[:space:]]*(map_path|ins_only_mode):' "${tmp}" | sed 's/#.*//' | tr -s ' ' | tr '\n' ' ')" >&2
  echo "${tmp}"
}
