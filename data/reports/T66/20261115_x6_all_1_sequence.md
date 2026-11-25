# 20261115_x6_all_1

图上 6（规划 crosswalk 0）。20:06:19–20:12:55，394 s。终点改过两次：开录时 17.45 −17.45，20:08:17 改成 28.91 3.97，20:08:57 改成 27.00 4.03。回灌用最后一次。没有人写的备注。

开录时已经在 cross_wait，出口距离停在 2.2 m，狗没有沿横道往前走。cross 和 cross_wait 来回：

| 时刻 | 变成 |
|---|---|
| 20:06:35 | cross_wait，等 1256 |
| 20:06:45 | cross |
| 20:07:12 | cross_wait |
| 20:07:15 | cross |
| 20:07:16 | cross_wait |
| 20:07:31 | cross |
| 20:07:39 | cross_wait |
| 20:07:48 | cross，同一秒又 cross_wait |
| 20:07:54 | cross |
| 20:08:03 | cross_wait |
| 20:08:57 | wait (vehicle)。刚过入口 0.06 m，朝向差约 8°，出口 27.7 m，等 1304 |
| 20:09:24 | go (clear)，同一秒进 cross |
| 20:10:06 | cross_wait，等 1325；20:10:11 cross |
| 20:10:21 | cross_wait；20:10:30 cross |
| 20:10:38 | cross_wait；20:10:56 cross |
| 20:11:11 | cross_wait；20:11:18 cross |
| 20:11:25 | cross_wait；20:11:31 cross |
| 20:12:06 | cross_wait（等横穿大车 1354） |
| 20:12:12 | none (`crosswalk -1`, `unreleased = 1`, 过入口 0.007 m)：狗压边线微抖动触发 `passed_` 死区 Bug，最后 42 s 门控被旁路（已在 `crosswalk_yield.cc` 修复，详析与看板见 `data/reports/T66/x6_online_analysis/`） |

20:08:57 之后出口距离停在 27.7 m（`map` 真实终点 `(80.72, 216.44)`）。`Crosswalk yield` 99 行。MPPI 失败 7 行。standstill 0 行。「没有 go」8 行。`/goal_pose` 2 条，是中途改的那两次。`/mc/motion_control` 0 条。
