#!/usr/bin/env bash
# Are the Orin's clocks locked (sudo jetson_clocks; lost at every reboot)? One line: CPU and GPU minimum /
# maximum / current frequency and the power mode. Unlocked, the node is 2-3x slower at 10 Hz (T32: total_ms 55
# against 15-21 locked): note it next to every latency measured. Exit 0 locked, 1 not locked, 2 unknown.
#
#   orin_clocks.sh            orin_soak.sh writes it to <out>/clocks.txt, orin_foxglove.sh prints it
#
# ORIN_SYSFS (default /sys) is for the tests.
set -u
SYS=${ORIN_SYSFS:-/sys}
cpu=$SYS/devices/system/cpu/cpu0/cpufreq
gpu=$(ls -d "$SYS"/class/devfreq/17000000.* 2> /dev/null | head -1)      # .gpu (JetPack 6) or .ga10b (5)
read_f() { cat "$1" 2> /dev/null || echo "?"; }
mode=$(nvpmodel -q 2> /dev/null | sed -n 's/^NV Power Mode: //p' | head -1)
c_min=$(read_f "$cpu/scaling_min_freq") c_max=$(read_f "$cpu/scaling_max_freq") c_cur=$(read_f "$cpu/scaling_cur_freq")
g_min=? g_max=? g_cur=?
if [ -n "$gpu" ]; then
  g_min=$(read_f "$gpu/min_freq") g_max=$(read_f "$gpu/max_freq") g_cur=$(read_f "$gpu/cur_freq")
fi
detail="cpu0 $c_min / $c_max / $c_cur kHz, gpu $g_min / $g_max / $g_cur Hz (min / max / current), power mode ${mode:-?}"
if [ "$c_min" = "?" ] || [ "$g_min" = "?" ]; then
  echo "clocks unknown: $detail"
  exit 2
fi
if [ "$c_min" = "$c_max" ] && [ "$g_min" = "$g_max" ]; then
  echo "clocks locked: $detail"
  exit 0
fi
echo "clocks NOT locked (sudo jetson_clocks; latencies 2-3x higher): $detail"
exit 1
