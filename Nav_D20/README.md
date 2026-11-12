# Nav_D20

D20 的感知、定位和规划。感知用 `lidar_seg` 加 `grid_map_memory_node`，定位用 `run_loc_online`，规划是单个 `planning` 进程。

克隆后在这台机器上可以直接跑。编译好的 Humble 环境在 `/public/zjst/robot/planning_humble_env`，不在这个仓库里。`bags0921` 是指向 `/public/zjst/robot/rosbag0921` 的链接。

在 CenterPoint 仓库根目录执行：

```bash
# 播 0921 bag，并启动地面分割、局部栅格、定位和 planning。会发 /cmd_vel，默认 ROS_DOMAIN_ID=42。
bash Nav_D20/planning/script/run_planning_stack_debug.sh
```

规划怎么接话题、怎么看结果，写在 `planning/docs/ONBOARDING.md`。定位做了什么，写在 `slam/docs/OVERVIEW.md`。
