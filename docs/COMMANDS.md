# 常用命令

本机是你的电脑，服务器是 `junyongmou@10.36.136.7`。连服务器用 IPv4：`gpu1` 会走到 IPv6，握手被关掉。

仓库在服务器上：

```text
/home/engineers/junyongmou/robot/ultralytics/d20car_predict/CenterPoint
```

## 开隧道

8 号狗在 `192.168.122.100`，用户 `nvidia`。在连得上狗的电脑上开反向隧道，服务器的 `127.0.0.1:5038` 接到狗的 22 端口。断线会重连。这个终端挂着别关：

```bash
autossh -M 0 -N -4 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -p 1222 -R 5038:192.168.122.100:22 junyongmou@10.36.136.7
```

服务器上 `~/.ssh/config` 里 `Host dog` 是 `localhost`、端口 `5038`、用户 `nvidia`。隧道开着时：

```bash
ssh dog
```

不要在狗的域上起规划。规划会发 `/cmd_vel` 和 `/mc/motion_control`。调试回放的 `ROS_DOMAIN_ID` 是 42。

## 挂载目录

本机 Foxglove 只能打开本机磁盘上的文件。挂上之后，文件对话框里就能进 `work_dirs`。

在本机：

```bash
mkdir -p ~/centerpoint
sshfs -o reconnect,follow_symlinks \
  -o ssh_command='ssh -4' \
  junyongmou@10.36.136.7:/home/engineers/junyongmou/robot/ultralytics/d20car_predict/CenterPoint \
  ~/centerpoint
```

本机没有 `sshfs` 时：`sudo apt install sshfs`。

Foxglove：Open local file。要看不闪的画面，打开 `viz/planning_viz.mcap`，不要打开同目录几百 MB 的 `*_0.mcap`。

```text
~/centerpoint/work_dirs/planning_io_t49_ins/viz/planning_viz.mcap
```

布局也在这个挂载里。看规划导入 `~/centerpoint/Nav_D20/planning/script/planning_viz.json`，看定位导入 `loc_viz.json`。

用完卸掉：

```bash
fusermount -u ~/centerpoint
```
