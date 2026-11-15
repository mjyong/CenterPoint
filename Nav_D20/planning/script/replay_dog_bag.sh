#!/bin/bash
# 一条命令回放狗上录的包（T68）。读 record_log.txt 最后一行 replay goal，sqlite3 先转成 mcap，
# 只放传感器和感知目标（狗上算好的栅格、里程计、tf 不放），栅格在容器里改订 reliable。
#
#   bash Nav_D20/planning/script/replay_dog_bag.sh <包目录> [名字]
# 没有 replay goal 就退出，用 Nav_D20/planning/tools/bag_on_map.py 看目标点。
# 对照写在 work_dirs/planning_io_<名字>/sequence.md：两列都是"包开始后几秒"（现场：record_log.txt 第一行 recording 的时刻；
# 回灌：logs/play_start.log 的时刻加 ros2 bag play 的 --delay 1 s），备注插在现场那列。
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  sed -n '2,6p' "$0"
  exit 2
fi
bag="$(cd "$1" && pwd)"
name="${2:-$(basename "$bag")}"
nav_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="$(cd "${nav_root}/.." && pwd)"
log="${bag}/record_log.txt"
if [[ ! -f "${log}" ]]; then
  echo "没有 ${log}：用 Nav_D20/planning/tools/bag_on_map.py 看目标点" >&2
  exit 1
fi
line="$(grep 'replay goal: ' "${log}" | tail -n 1 || true)"
if [[ ! "${line}" =~ replay\ goal:\ ([-0-9.]+)\ ([-0-9.]+) ]]; then
  echo "record_log.txt 里没有 replay goal：用 Nav_D20/planning/tools/bag_on_map.py 看目标点" >&2
  exit 1
fi
goal_x="${BASH_REMATCH[1]}"
goal_y="${BASH_REMATCH[2]}"

play="${bag}"
if [[ -d "${bag}_mcap" ]]; then
  play="${bag}_mcap"
else
  sqlite=0
  shopt -s nullglob
  dbs=("${bag}"/*.db3)
  shopt -u nullglob
  if (( ${#dbs[@]} )); then
    sqlite=1
  elif [[ -f "${bag}/metadata.yaml" ]] && grep -q 'storage_identifier: sqlite3' "${bag}/metadata.yaml"; then
    sqlite=1
  fi
  if [[ "${sqlite}" -eq 1 ]]; then
    python3 "${NAV_BAG_TO_MCAP:-${repo_root}/Nav_D20/planning/tools/bag_to_mcap.py}" "${bag}" "${bag}_mcap"
    play="${bag}_mcap"
  fi
fi

export PLANNING_STARTUP_MAP="${repo_root}/data/loc_maps/ad68517c-e845-42a4-b5dd-a6b7c6d2af13"
export LOC_MAP_PATH="${PLANNING_STARTUP_MAP}"
export BAG_PLAY_EXTRA="${BAG_PLAY_EXTRA:---topics /robot/topic/rdap/top_point_cloud2 /rte/rdap/rtk/inspvax /rte/rdap/rtk/imus/data_raw /perception/lidar_3d/objects}"
io="${NAV_IO_DIR:-${repo_root}/work_dirs/planning_io_${name}}"
echo "replay goal: ${goal_x} ${goal_y}"
echo "bag: ${play}"
echo "放法: ${BAG_PLAY_EXTRA}"

check="${NAV_REPLAY_CHECK:-${nav_root}/planning/script/replay_planning_check.sh}"
set +e
bash "${check}" "${name}" "${goal_x}" "${goal_y}" --bag "${play}"
code=$?
set -e

GOAL_X="${goal_x}" GOAL_Y="${goal_y}" python3 - "${bag}" "${io}" << 'PY'
import os, re, sys

bag, io = sys.argv[1], sys.argv[2]
os.makedirs(io, exist_ok=True)

def hms(h, m, s):
    return int(h) * 3600 + int(m) * 60 + float(s)

TOD = re.compile(r"(?<!\d)(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)")   # the first time of day on a line ("T17:32:35" too)

field, t0 = [], None
for line in open(os.path.join(bag, "record_log.txt"), errors="replace"):
    m = re.match(r"\[(\d{2}):(\d{2}):(\d{2})\] (.*)", line)
    if not m:
        continue
    if t0 is None and m.group(4).startswith("recording "):
        t0 = hms(*m.groups()[:3])                  # nav_cw_record.sh started ros2 bag record here
    if m.group(4).startswith("note: "):
        field.append((hms(*m.groups()[:3]), m.group(4).strip()))
journal = os.path.join(bag, "journal_cw-planning.txt")
if os.path.isfile(journal):
    for line in open(journal, errors="replace"):
        if "Crosswalk yield:" not in line:
            continue
        m = TOD.search(line)                       # the journal's own stamp (short-iso-precise or short-precise)
        if not m:
            continue
        field.append((hms(*m.groups()), "Crosswalk yield: " + line.split("Crosswalk yield:", 1)[1].strip()))
field.sort()
if t0 is None:
    t0 = field[0][0] if field else 0.0

replay = []
logs = os.path.join(io, "logs")
picked = None
if os.path.isdir(logs):
    names = ["planning.INFO", "planning.log"] + sorted(
        n for n in os.listdir(logs) if n.startswith("planning.") and "INFO" in n)
    for n in names:
        if os.path.isfile(os.path.join(logs, n)):
            picked = os.path.join(logs, n)
            break
r0 = None
start = os.path.join(logs, "play_start.log")
if os.path.isfile(start):
    m = TOD.search(open(start).read())
    if m:
        r0 = hms(*m.groups()) + 1.0                # ros2 bag play --delay 1
if picked:
    for line in open(picked, errors="replace"):
        m = re.search(r"\b(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", line)
        if m is None:
            continue
        t = hms(*m.groups())
        if r0 is None and re.match(r"[IWEF]\d{4} ", line):
            r0 = t
        if "Crosswalk yield:" in line:
            if r0 is None:
                r0 = t
            replay.append((t - r0, "Crosswalk yield: " + line.split("Crosswalk yield:", 1)[1].strip()))

rows = [(t - t0, "field", text) for t, text in field] + [(t, "replay", text) for t, text in replay]
rows.sort()
out = ["# %s" % os.path.basename(io), "",
       "goal %s %s" % (os.environ["GOAL_X"], os.environ["GOAL_Y"]), "",
       "| 包开始后 (s) | 现场 | 回灌 |", "|---|---|---|"]
for t, src, text in rows:
    cell = text.replace("|", "/")
    if src == "field":
        out.append("| %.1f | %s | |" % (t, cell))
    else:
        out.append("| %.1f | | %s |" % (t, cell))
if not rows:
    out.append("| | （现场日志和回灌日志里都没有 Crosswalk yield） | |")
path = os.path.join(io, "sequence.md")
open(path, "w").write("\n".join(out) + "\n")
print("sequence: %s" % path)
PY
exit "${code}"
