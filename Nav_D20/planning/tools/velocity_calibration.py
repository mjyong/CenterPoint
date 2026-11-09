#!/usr/bin/env python3
"""
速度指令标定工具

对标定三个轴(VX / VY / W)依次扫频，下发 cmd_vel 指令并采集 odometry 反馈，
建立 cmd → real 映射表，输出到 velocity_calibration.yaml。

用法:
    python3 velocity_calibration.py [--topic-cmd /cmd_vel] [--topic-odom /odom]

标定流程:
    1. 依次对 VX → VY → W 三个轴进行扫频
    2. 每个档位: 等待 settle_time 稳定 → 采样 sample_time → 计算均值
    3. 全部完成后保存 velocity_calibration.yaml
"""

import math
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from mc_interface.msg import Motion
from rclpy.node import Node

# ---- 标定参数 -----------------------------------------------------------
SWEEP_VALUES = [
    -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
]
SETTLE_TIME = 1.5   # 指令下发后等待稳定的时间 (秒)
SAMPLE_TIME = 2.5   # 采样持续时间 (秒)
STOP_TIME = 3.0     # 完成一次后停止等待的时间 (秒)
AXIS_NAMES = ["VX", "VY", "W"]


@dataclass
class CalibrationTable:
    """单个轴的标定表"""
    cmds: list = field(default_factory=list)    # 下发的指令值
    reals: list = field(default_factory=list)   # 实测的真实速度


class VelocityCalibrationNode(Node):
    def __init__(self, topic_cmd: str = "/cmd_vel", topic_odom: str = "/robot/topic/mapm/odometry"):
        super().__init__("velocity_calibration_node")

        self.topic_cmd = topic_cmd
        self.topic_odom = topic_odom

        # 发布者
        # self.cmd_pub = self.create_publisher(Twist, topic_cmd, 10)
        self.cmd_pub = self.create_publisher(Motion, "/mc/motion_control", 10)

        # 里程计订阅
        self.odom_sub = self.create_subscription(
            Odometry, topic_odom, self._odom_callback, 10,
        )

        # 标定结果
        self.tables = {
            "vx": CalibrationTable(),
            "vy": CalibrationTable(),
            "omega": CalibrationTable(),
        }

        # 状态机
        self.axis_index = 0        # 0=VX, 1=VY, 2=W
        self.sweep_index = 0       # 当前轴内的档位索引
        self.current_cmd = 0.0     # 当前下发的指令值

        # 时序控制
        self._phase = "settle"     # settle | sample | stop
        self._phase_start = 0.0    # 当前阶段开始时间
        self._buffer: deque = deque()  # 采样缓冲 (value, dt)
        self._buffer_time = 0.0

        # 里程计
        self._last_odom = None
        self._last_odom_time = 0.0

        # 定时器: 20Hz 驱动状态机
        self._timer = self.create_timer(0.05, self._tick)

        self.get_logger().info(
            f"标定节点启动, cmd_topic={topic_cmd}, odom_topic={topic_odom}"
        )
        self.get_logger().info(
            f"扫频档位: {SWEEP_VALUES} (共 {len(SWEEP_VALUES)} 档)"
        )
        self._start_next_step()

    # ---- 里程计回调 -------------------------------------------------------
    def _odom_callback(self, msg: Odometry):
        self._last_odom = msg
        self._last_odom_time = time.time()

    # ---- 主状态机 (20Hz) --------------------------------------------------
    def _tick(self):
        if self.axis_index >= len(AXIS_NAMES):
            # 全部完成
            self._timer.cancel()
            self._on_finished()
            return

        now = time.time()
        elapsed = now - self._phase_start

        if self._phase == "settle":
            self._publish_cmd()
            if elapsed >= SETTLE_TIME:
                self._enter_sample()
        elif self._phase == "sample":
            self._publish_cmd()
            self._do_sample()
            if self._buffer_time >= SAMPLE_TIME:
                self._finish_sample()
        elif self._phase == "stop":
            # 停止等待，不发指令
            if elapsed >= STOP_TIME:
                self._advance()

    # ---- 阶段切换 ---------------------------------------------------------
    def _start_next_step(self):
        """切换到下一个档位"""
        if self.axis_index >= len(AXIS_NAMES):
            return

        axis = AXIS_NAMES[self.axis_index]
        self.current_cmd = SWEEP_VALUES[self.sweep_index]

        self._publish_cmd()
        self._phase = "settle"
        self._phase_start = time.time()
        self.get_logger().info(
            f"[{axis}] 档位 {self.sweep_index + 1}/{len(SWEEP_VALUES)} "
            f"cmd={self.current_cmd:+.1f}  进入稳定等待 ({SETTLE_TIME}s)"
        )

    def _publish_cmd(self):
        """下发当前指令"""
        axis = AXIS_NAMES[self.axis_index]
        msg = Motion()
        msg.mode = 2
        msg.motion = 20
        if axis == "VX":
            msg.para = [self.current_cmd, 0.0, 0.0]
        elif axis == "VY":
            msg.para = [0.0, self.current_cmd, 0.0]
        else:
            msg.para = [0.0, 0.0, self.current_cmd]
        self.cmd_pub.publish(msg)

    def _enter_sample(self):
        """进入采样阶段"""
        self._phase = "sample"
        self._phase_start = time.time()
        self._buffer.clear()
        self._buffer_time = 0.0
        self.get_logger().info(f"  → 开始采样 ({SAMPLE_TIME}s)")

    def _enter_stop(self):
        """进入停止等待阶段"""
        self._phase = "stop"
        self._phase_start = time.time()
        self.get_logger().info(f"  → 停止等待 ({STOP_TIME}s)")

    def _do_sample(self):
        """采样一帧里程计数据"""
        if self._last_odom is None:
            return

        # 用时间戳差作为 dt（首次采样用定时器周期近似）
        if self._buffer:
            dt = self._last_odom_time - self._buffer[-1][1]
        else:
            dt = 0.05
        if dt <= 0 or dt > 1.0:
            return

        axis = AXIS_NAMES[self.axis_index]
        twist = self._last_odom.twist.twist
        if axis == "VX":
            real = twist.linear.x
        elif axis == "VY":
            real = twist.linear.y
        else:
            real = twist.angular.z

        self._buffer.append((real, self._last_odom_time))
        self._buffer_time += dt

    def _finish_sample(self):
        """完成当前档位采样，记录结果"""
        if not self._buffer:
            self.get_logger().warn("采样缓冲为空，跳过当前档位")
            self._advance()
            return

        # 加权平均
        total = 0.0
        total_time = 0.0
        prev_t = self._buffer[0][1]
        for val, t in self._buffer:
            dt = t - prev_t
            if dt <= 0:
                dt = 0.05
            total += val * dt
            total_time += dt
            prev_t = t
        mean = total / total_time if total_time > 0 else 0.0

        # 零指令强制零输出
        if abs(self.current_cmd) < 1e-5:
            mean = 0.0

        axis = AXIS_NAMES[self.axis_index]
        key = {"VX": "vx", "VY": "vy", "W": "omega"}[axis]
        self.tables[key].cmds.append(self.current_cmd)
        self.tables[key].reals.append(mean)

        self.get_logger().info(
            f"  ✓ [{axis}] cmd={self.current_cmd:+.1f}  →  real={mean:+.4f}"
        )

        self._enter_stop()

    def _advance(self):
        """推进到下一个档位 / 下一个轴"""
        self.sweep_index += 1
        if self.sweep_index >= len(SWEEP_VALUES):
            self.sweep_index = 0
            self.axis_index += 1
            if self.axis_index >= len(AXIS_NAMES):
                self.get_logger().info("全部轴标定完成!")
                return
            self.get_logger().info(
                f"--- 切换到 {AXIS_NAMES[self.axis_index]} 轴 ---"
            )

        self._start_next_step()

    # ---- 完成 & 保存 -------------------------------------------------------
    def _on_finished(self):
        """标定完成，停止指令并保存结果"""
        # 停止机器人
        # self.cmd_pub.publish(Twist())
        self.cmd_pub.publish(Motion())
        self.get_logger().info("已发送零速指令")

        self._save_yaml()
        self.get_logger().info("标定完成，即将退出。")
        rclpy.shutdown()

    def _save_yaml(self):
        """保存标定表到 YAML 文件"""
        import os

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "velocity_calibration.yaml")

        lines = []
        for key, label in [
            ("vx", "i_vx"), ("vy", "i_vy"), ("omega", "i_omega")
        ]:
            table = self.tables[key]
            cmds_str = "[" + ", ".join(f"{v:.1f}" for v in table.cmds) + "]"
            reals_str = "[" + ", ".join(f"{v:.4f}" for v in table.reals) + "]"
            lines.append(f"{label}: {cmds_str}")
            lines.append(f"o_{key}: {reals_str}")

        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        self.get_logger().info(f"标定表已保存到: {output_path}")

        # 打印摘要
        print("\n" + "=" * 60)
        print("  标定结果摘要")
        print("=" * 60)
        for key, label in [("vx", "VX"), ("vy", "VY"), ("omega", "W")]:
            table = self.tables[key]
            print(f"\n  [{label}]")
            print(f"  {'cmd':>8s}  →  {'real':>10s}")
            print(f"  {'-' * 22}")
            for c, r in zip(table.cmds, table.reals):
                print(f"  {c:+.1f}      →  {r:+.4f}")
        print("\n" + "=" * 60)


def main(args=None):
    rclpy.init(args=args)

    # 解析命令行参数
    topic_cmd = "/cmd_vel"
    topic_odom = "/robot/topic/mapm/odometry"
    remaining = sys.argv[1:] if len(sys.argv) > 1 else []
    i = 0
    while i < len(remaining):
        if remaining[i] == "--topic-cmd" and i + 1 < len(remaining):
            topic_cmd = remaining[i + 1]
            i += 2
        elif remaining[i] == "--topic-odom" and i + 1 < len(remaining):
            topic_odom = remaining[i + 1]
            i += 2
        else:
            i += 1

    node = VelocityCalibrationNode(topic_cmd=topic_cmd, topic_odom=topic_odom)

    def _sigint_handler(sig, frame):
        node.get_logger().info("收到中断信号，停止标定并保存已有数据...")
        # node.cmd_pub.publish(Twist())
        node.cmd_pub.publish(Motion())
        node._save_yaml()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
