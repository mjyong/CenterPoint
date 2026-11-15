# T66 现场手册：影子模式下逐个路口调人行横道

笔记本仓库已在 `claude/me/v01` 的 `213f8db`。这一页是现场按顺序做的命令和看什么。

出处：

| 要查的 | 在哪 |
|---|---|
| 任务原文、停止条件、报告要写什么 | `docs/SERVER_TASKS.md` 的 T66 |
| 五层表、「走一遍」、录包细节 | `docs/DOG_FIELD_GUIDE.md` 第 8.1、9 节 |
| 状态机从感知到速度怎么算 | `docs/CROSSWALK_YIELD_FLOW.md` |
| 两张地图 | `docs/map_ad68517c_crosswalks.png`、`docs/map_a70c6db6_crosswalks.png` |

规矩：影子一直开着，规划不发 `/mc/motion_control`、`/cmd_vel`；狗只由人遥控；急停在手；同事开车不超过 10 km/h，离狗至少 3 m，不朝狗站的车道开。域 0 上不发运控、焦点、模式。每一步先说要做什么，人说可以再做。只写 `/opt/seres/nav_cw/`、`/home/nvidia/`、`/tmp/`，sudo 一律 `sudo -n`。

笔记本是 PowerShell。下面远程命令都写成 `ssh dog '…'`，里面没有 `$`。含 `$` 的不要直接贴进 PowerShell。

Foxglove：`ws://192.168.122.100:8765`，Layout 导入仓库里最新的 `Nav_D20/planning/script/planning_live.json`。能从 Foxglove 发的只有 `/goal_pose`。

---

## 开工前先清今天下午留下的

T57 收工时是 `running: ours`、影子开、限速 0.3，没有换回。今天下午又叠了这些，不先清掉，T66 的定位和包都是假的：

- 定位进程在 15:41 崩过，后来重起过，激光对不上后又停发；
- 另有一个假定位在发 `/lightning/odom` 和 `map → base_link`，点在 **(-56.15, -228.26)**，是图上一条横道的坐标，不是狗现在站的地方；
- Foxglove 桥可能还在 8765。

先确认再停掉假定位（进程名 `fake_pose.py`，在 `/tmp/fake_pose.py`）。停的是那个 python，不要停 `cw-slam`：

```text
ssh dog 'ps -ef | grep fake_pose.py | grep -v grep'
ssh dog 'sudo -n kill <上面的 pid>'
```

然后再看是不是还是我们的栈。是的话，换规划之前先换回他们的（T66 第 0 步：不能在我们的规划跑着时换包）：

```text
ssh dog 'sudo -n bash /opt/seres/nav_cw/nav_cw.sh status'
ssh dog 'sudo -n bash /opt/seres/nav_cw/nav_cw.sh use-original'
```

`use-original` 之后应是 `running: theirs`。仓库里没有「T65 结果」，新门控可能还没装上，下面第 0 步 b 先查。

---

## 0. 准备

```text
mkdir data\reports\T66
```

**a. 脚本拷上去**（`git archive` 出 LF）：

```text
git archive HEAD tools/deploy/nav_cw.sh tools/deploy/nav_cw_foxglove.sh tools/deploy/nav_cw_record.sh tools/deploy/nav_cw_record.py | ssh dog "rm -rf /tmp/t66 && mkdir -p /tmp/t66 && tar -x -C /tmp/t66 && sudo -n cp /tmp/t66/tools/deploy/* /opt/seres/nav_cw/"
ssh dog "file /opt/seres/nav_cw/nav_cw.sh /opt/seres/nav_cw/nav_cw_foxglove.sh /opt/seres/nav_cw/nav_cw_record.sh /opt/seres/nav_cw/nav_cw_record.py"
ssh dog "grep -c shadow_blocked /opt/seres/nav_cw/nav_cw.sh"
```

`file` 不应出现 `CRLF`。`shadow_blocked` 至少 1 行：影子模式把「被堵就 standstill」关掉，不然人把狗按在路边、门控一放行，大约 4.5 秒后规划进 standstill。

**b. 规划是不是新门控**（人行横道上、前方来车也停）：

```text
ssh dog "grep -ral 'stops for vehicles ahead' /opt/seres/nav_cw/install/planning/planning/lib"
```

有输出：新门控，第 3 步 c 可以做。没有：旧门控，上了人行横道一律不停，第 3 步 c 不做。要装的话按 T65 第 4 步：包在服务器 `/public/zjst/robot/cp_data/nav_cw_planning_1a5ed36.tar.gz`，sha256 `afaa3921ff26c314d9137cfac5a5110596ef07fe3eae923c81f9c5dc4dc68c9f`。装的时候我们的规划必须是停的。装完 `configure` 再 `apply`，有 `NOT APPLIED` 就把 `backup/planning.t61` 挪回去，停下。

**c. INS 固定解，狗放在路网上。** `/rte/rdap/rtk/inspvax` 要 `ins_status` 3、`pos_type` 56。不是固定解就先别 `use-new`：定位只等 30 秒，超时后用配置里的位姿，对不上地图。

**d. 开关，再换上我们的：**

```text
ssh dog "sudo -n bash /opt/seres/nav_cw/nav_cw.sh shadow on"
ssh dog "sudo -n bash /opt/seres/nav_cw/nav_cw.sh speed 0.5"
ssh dog "sudo -n bash /opt/seres/nav_cw/nav_cw.sh switches"
ssh dog "sudo -n bash /opt/seres/nav_cw/nav_cw.sh use-new"
```

`switches` 要看到：shadow on，行尾有 `nav_blocked false: a person drives`；appmap ignore；限速 0.5。不能有 `NOT APPLIED`，不能有 `REHEARSAL ON`。

`use-new` 的图要全是 ok：规划在 `/nav_cw/shadow/mc/motion_control` 上，不在 `/mc/motion_control` 上。不是全 ok 就停下，它会自己停掉 `cw-planning` 并退出 3，不要再起。

日志里还要有：`Tracking: crosswalk yield on, rule wait`；`startup map … 5 crosswalks`；之后没有 `Map switch`、`Map update`。

**e. 三个窗口：**

```text
ssh dog "sudo -n bash /opt/seres/nav_cw/nav_cw_foxglove.sh"
ssh dog "journalctl -u cw-planning -f | grep --line-buffered -E 'Crosswalk yield|nav_blocked|Navigation|standstill'"
ssh -t dog
```

第三个要 `-t`，录包时才能敲备注。Foxglove 桥留在前台。

---

## 1. 选哪个路口

测的是 **ad68517c**（规划起来读的那张，5 条）。白线是路，黄块是人行横道。图：`docs/map_ad68517c_crosswalks.png`。

规划日志里的 `crosswalk N`、侧栏 `crosswalk_index` 是另一套号：

| 规划 / 侧栏 | 图上的号 | 这一趟 |
|---|---|---|
| 0 | 6 | 多半走不出 approach，路只在一端碰到 |
| 1 | 13 | 先做。路网画宽了，全局路径从白框上过 |
| 2 | 22 | 路径只擦边 |
| 3 | 32 | 先做 |
| 4 | 38 | 先做 |

包名用图上的号：`x13_walk`，不是 `x1_walk`。至少两个路口，其中一个有正常车流。发目标之前看全局路径（蓝线）是不是穿过白框。

另一张图 a70c6db6 也有 13、32、38，不是同一个路口。这次不用那张。

---

## 2. 每个路口：先空着走一遍

录包窗口里，起步前：

```text
sudo -n bash /opt/seres/nav_cw/nav_cw_record.sh --tag x13_walk
```

把 `13` 换成图上的号。到目标 1.5 m 以内，停 5 秒后它自己停。中途走出 1.5 m 就重新数。没有目标、或要提前停：在这个窗口 Ctrl-C。不要另开终端去杀进程，不然没有 `metadata.yaml`。

边走边看下面的五层。不对就在同一个窗口敲一行回车，例如 `align：狗头离线 0.5 m`。这一遍必须留下：

- align 那一刻狗头离斑马线入口线多远（人看地上）；
- 白框偏哪边、大约多少；
- 状态窗口里的状态序列；approach 最后 3 m 影子速度是不是降到大约两成（限速 0.5 时约 0.1 m/s）；align 的角速度方向对不对；
- 紫的参考线和绿的局部路线是不是跟着蓝的全局路径从白框上过。

定位偏超过 1 m，或者一直跳：这个路口不往下做，备注里写清楚，换下一个。

人怎么走（人当执行器）：

1. 在路线上离人行横道 10 m 以外起步。Foxglove 发对面的目标，蓝线要穿过白框。5 秒内 `mission_state` 不是 active 就再发一次（还在跑的任务，第一下只会被停掉）。
2. approach：沿局部路线走，速度跟着影子指令（限速 0.5 时大约 0.5 m/s），最后 3 m 慢下来。
3. 状态字变成 align（红）就停，看狗离斑马线多远，敲备注。按角速度的方向原地转，朝向差到 10° 以内就变成 wait 或 go。
4. wait（红）站着。go（绿）沿路线走过去。
5. 人行横道上 cross（绿）一直走。变成 cross_wait（红，理由 ahead）就原地停，变回 cross 再走。旧门控没有这一档，上了横道不停。
6. 出了人行横道变成 none，走到目标 1.5 m 内，录包 5 秒后自己停。这时影子速度还不是 0 是正常的：到达要离终点 0.5 m 内、朝向差 0.2 rad 内，人遥控停在旁边就到不了。

---

## 五层：上一层错了，下一层不作数

| 层 | 看什么 | 在哪看 | 对的样子 |
|---|---|---|---|
| 1 定位 | 箭头在路上，朝向和真狗一样，不跳 | 3D：`/lightning/odom`、站点缩略图 | 贴着路。偏超过 1 m，这个路口后面不看 |
| 2 地图 | 白框盖住地上的斑马线；align 时狗停在哪 | `/planning/debug/crosswalk`；人看地上 | 狗中心在入口线前约 0.8 m（狗头约 0.3 m） |
| 3 感知 | 车框贴在真车上；动的有速度箭头，停的没有；同一辆车 id 不变 | 车框；侧栏「感知的原始目标」 | 蓝 = 看到，红 = 正在等它，橙 = 看到但不停 |
| 4 门控 | 来车就等，过去约 3 秒放行 | 状态字；侧栏 `crosswalk_go`、`crosswalk_waiting_for`、`crosswalk_stale`；状态窗口里的 `Crosswalk yield` | 停着的车只看 3 秒（压着路 15 秒）；不 stale |
| 5 状态机和指令 | 序列和速度 | 影子速度、角速度曲线；`/planning/debug/path_tracking`；侧栏 `crosswalk_mode` | 见上面「人怎么走」。hold 时速度是 (0, 0, ω)，ω 不超过 0.5 rad/s |

侧栏 `crosswalk_mode`：0 none，1 approach，2 align，3 wait，4 go，5 cross，6 cross_wait。`crosswalk_index` 用上一节的表换成图上的号。

`/planning/debug/path_tracking` 里：蓝线是整条全局路径，紫线是从狗往前约 5 m 裁给局部规划的参考线，绿线是局部规划绕完障碍后要走的，箭头是终点（红 = 锁了到点朝向，Foxglove 点的目标就是红的）。

顺带看：侧栏 `nav_blocked` 一直是 0；`mission_state` 是 active。影子开着时被堵检测是关的，人把狗按住不会进 standstill。

全局缩略图 `/planning/debug/global_map_manager/occmap` 几乎整张都是障碍，看局部时关掉，免得盖住白框。

---

## 3. 同一个路口接着做

**a. 定点看**（有正常车流的路口一定做，默认 10 分钟）：

```text
sudo -n bash /opt/seres/nav_cw/nav_cw_record.sh --tag x13_watch --no-auto
```

发目标，照「走一遍」走到 wait 或 go，狗面朝对面站在停止点，人不再走。状态随车流在 wait / go 之间变。每次变成 go，判断这时候起步安不安全。不安全敲 `go 不安全：右边来车 20 m`。没车也在等，敲 `空等`。车框不对也敲。到时间在这个窗口 Ctrl-C。

**b. 有车的时候再走一遍**（人觉得安全才走）：

```text
sudo -n bash /opt/seres/nav_cw/nav_cw_record.sh --tag x13_traffic
```

wait 时站着，go 才走。

**c. 同事开车**（只有新门控才做；人定做不做）。狗已经放行、走到人行横道中间，人把狗停住：

```text
sudo -n bash /opt/seres/nav_cw/nav_cw_record.sh --tag x13_car --no-auto
```

- 车从狗前方的车道开过：应变 cross_wait，理由 ahead，车框红；车过去约 3 秒回到 cross。
- 车从狗已经走过的那条车道开过：不停，车框橙。
- 车开过来之前敲一行：`前方，从左` 或 `后面，从右`。

车不从狗站的那条车道开。

---

## 4. 每个包录完核对

录包窗口最后要有 `done:` 和 `replay goal: x y`。包在 `/home/nvidia/bags/` 下，名字是 `<日期>_x13_walk_<n>` 这种。

```text
ssh dog "ls /home/nvidia/bags/<包>"
```

要有 `metadata.yaml`、`record_log.txt`、`journal_cw-planning.txt`、`journal_cw-slam.txt`、`journal_cw-nav-perception.txt`。没有 `metadata.yaml` 就按窗口里的提示做 `ros2 bag reindex`，不要在录的时候打开 db3。

这个路口五层各一句，由人说，写进 `data/reports/T66/field_notes.md`：定位；白框偏多少、align 时狗头离线多远；感知漏了或假了什么；门控对不对；状态机和指令。

---

## 5. 收工

```text
ssh dog "sudo -n bash /opt/seres/nav_cw/nav_cw.sh use-original"
ssh dog "sudo -n bash /opt/seres/nav_cw/nav_cw.sh speed off"
ssh dog "sudo -n bash /opt/seres/nav_cw/nav_cw.sh shadow off"
ssh dog "sudo -n bash /opt/seres/nav_cw/nav_cw.sh switches"
ssh dog "sudo -n bash /opt/seres/nav_cw/nav_cw.sh status"
```

`switches` 里 shadow off，下次 `use-new` 不会带着影子。`status` 是 `running: theirs`。

---

## 6. 回来再做（服务器，不在现场）

每个包：

```text
rsync -a --partial dog:/home/nvidia/bags/<包>/ /public/zjst/robot/cp_data/<包>/
bash Nav_D20/planning/script/replay_dog_bag.sh /public/zjst/robot/cp_data/<包> t66_<包>
```

回灌在 nav-dev 容器里，读 `record_log.txt` 最后的 `replay goal`。对照写在 `work_dirs/planning_io_t66_<包>/sequence.md`。再跑 `bag_to_mcap.py`、`planning_mcap_viz.py`、`objects_in_map.py`（每个包都要 `extrinsic OK`，不是 OK 的，门控结论不作数）。

人和现场日志对不上的、定点看的次数（wait 几次、多久，go 几次，「go 不安全」「空等」各几次），写进 `data/reports/T66/<包>_sequence.md`。

---

## 什么时候停下

| 看到 | 做什么 |
|---|---|
| `shadow on` 或 `use-new` 的图不是全 ok | 停下，不往下做。规划若还在 `/mc/motion_control` 上，脚本会自己停掉 `cw-planning` |
| 定位偏超过 1 m，或一直跳 | 这个路口不测 |
| 人觉得不安全 | 急停，记下时间 |
| 回来 `objects_in_map.py` 不是 extrinsic OK | 先查外参，门控的结论不作数 |
