#!/usr/bin/env python3

import csv
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


class TransformWriter(Node):
    def __init__(self):
        super().__init__("transform_writer")

        self.declare_parameter("log_file_path", "/tmp/adaptive_event_core_transform_log.csv")
        self.declare_parameter("log_density", True)

        self.log_file_path = str(self.get_parameter("log_file_path").value)
        self.log_density = bool(self.get_parameter("log_density").value)

        self.current_density = 0.0
        self.current_transform = None

        self._init_log_file()

        self.create_subscription(String, "/selected_transform", self.transform_callback, 10)
        if self.log_density:
            self.create_subscription(Float32, "/event_density", self.density_callback, 10)

        self.get_logger().info(f"Logging to {self.log_file_path}")

    def _init_log_file(self):
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        if os.path.exists(self.log_file_path):
            return

        fields = ["timestamp", "ros_time", "transform"]
        if self.log_density:
            fields.append("density")

        with open(self.log_file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

    def density_callback(self, msg: Float32):
        self.current_density = float(msg.data)

    def transform_callback(self, msg: String):
        self.current_transform = msg.data
        self._append_row()

    def _append_row(self):
        now = datetime.now()
        ros_time = self.get_clock().now().nanoseconds / 1e9

        row = {
            "timestamp": now.isoformat(),
            "ros_time": f"{ros_time:.6f}",
            "transform": self.current_transform,
        }
        if self.log_density:
            row["density"] = f"{self.current_density:.6f}"

        fields = ["timestamp", "ros_time", "transform"]
        if self.log_density:
            fields.append("density")

        with open(self.log_file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(row)


def main(args=None):
    rclpy.init(args=args)
    node = TransformWriter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
