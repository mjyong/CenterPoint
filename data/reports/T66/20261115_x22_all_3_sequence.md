# 20261115_x22_all_3

29.4 分钟（1,766,955,091,904 ns），1,271,296 条，db3 67 GB，20:58 写完。目录属主是 root。旁边没有 `record_log.txt`，也没有三份 `journal_*.txt`，所以没有回灌终点，也没有 `Crosswalk yield` 序列。

metadata 里看得到的：

- `/lightning/odom` 176,690 条，`/nav/location` 1,767 条，`/perception/lidar_3d/objects` 17,669 条
- `/planning/debug/crosswalk`、`path_tracking`、`plotjuggler/string` 各约 35,334 条
- `/nav_cw/shadow/cmd_vel` 和 `/nav_cw/shadow/mc/motion_control` 各 33,417 条
- `/mc/motion_control` 0 条
- `/goal_pose` 6 条
- `/planning/debug/collision_checker` 0 条
- 四路鱼眼各约 44,173 条

狗上 `journalctl -u cw-planning --since 20:25 --until 21:05` 已经没有 `Crosswalk yield` 行，补不回这段状态。笔记本上的 db3 是 0 字节，包还在狗上。
