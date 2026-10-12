#!/usr/bin/env python3
import argparse
import errno
import socket
import struct
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


CAN_FRAME_FMT = "=IB3x8s"


def clamp(value, limit):
    return max(-limit, min(limit, value))


def s16_bytes(value):
    value = int(max(-32768, min(32767, value)))
    return bytes([(value >> 8) & 0xFF, value & 0xFF])


def can_frame(can_id, data):
    data = bytes(data)
    return struct.pack(CAN_FRAME_FMT, can_id, len(data), data.ljust(8, b"\x00"))


class DirectCanCmdvelBridge(Node):
    def __init__(self, args):
        super().__init__("xw_direct_can_cmdvel_bridge")
        self.args = args
        self.sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.sock.bind((args.can, ))
        self.cmd_vx = 0.0
        self.cmd_wz = 0.0
        self.last_cmd_time = 0.0
        self.last_mode_refresh = 0.0
        self.sent_count = 0
        self.recv_count = 0
        self.create_subscription(Twist, args.topic, self.cmd_cb, 10)
        self.create_timer(1.0 / args.rate, self.tick)
        self.get_logger().info(
            f"direct CAN cmdvel bridge ready topic={args.topic} can={args.can} "
            f"rate={args.rate} command_clamp={not args.disable_command_clamp} "
            f"max_vx={args.max_vx} max_wz={args.max_wz}"
        )

    def cmd_cb(self, msg):
        if self.args.disable_command_clamp:
            self.cmd_vx = float(msg.linear.x)
            self.cmd_wz = float(msg.angular.z)
        else:
            self.cmd_vx = clamp(float(msg.linear.x), self.args.max_vx)
            self.cmd_wz = clamp(float(msg.angular.z), self.args.max_wz)
        self.last_cmd_time = time.monotonic()
        self.recv_count += 1
        if self.recv_count <= 5 or self.recv_count % 30 == 0:
            self.get_logger().info(
                f"received cmd_vel #{self.recv_count} vx={self.cmd_vx:.4f} wz={self.cmd_wz:.4f}"
            )

    def send(self, can_id, data):
        frame = can_frame(can_id, data)
        for _ in range(5):
            try:
                self.sock.send(frame)
                return True
            except OSError as e:
                if e.errno not in (errno.ENOBUFS, errno.EAGAIN):
                    raise
                time.sleep(0.004)
        self.get_logger().warn("CAN send buffer full; dropped one frame")
        return False

    def set_can_control(self):
        self.send(0x421, bytes([0x01, 0, 0, 0, 0, 0, 0, 0]))

    def set_ackermann(self):
        self.send(0x141, bytes([0x00, 0, 0, 0, 0, 0, 0, 0]))

    def send_motion(self, vx, wz):
        lin = int(vx * 1000)
        ang = int(wz * 1000)
        data = s16_bytes(lin) + s16_bytes(ang) + s16_bytes(0) + s16_bytes(0)
        if self.send(0x111, data):
            self.sent_count += 1

    def tick(self):
        now = time.monotonic()
        if now - self.last_mode_refresh > self.args.mode_refresh:
            self.set_can_control()
            self.set_ackermann()
            self.last_mode_refresh = now

        if self.last_cmd_time <= 0.0 or now - self.last_cmd_time > self.args.timeout:
            vx = 0.0
            wz = 0.0
        else:
            vx = self.cmd_vx
            wz = self.cmd_wz
        self.send_motion(vx, wz)

    def stop(self):
        for _ in range(10):
            self.send_motion(0.0, 0.0)
            time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--topic", default="/xw/cmd_vel_direct_can")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--max-vx", type=float, default=0.12)
    parser.add_argument("--max-wz", type=float, default=0.25)
    parser.add_argument("--disable-command-clamp", action="store_true")
    parser.add_argument("--timeout", type=float, default=0.25)
    parser.add_argument("--mode-refresh", type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = DirectCanCmdvelBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
