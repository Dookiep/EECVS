#!/usr/bin/env python3

import random

import numpy as np
import rclpy
from rclpy.node import Node

from adaptive_event_core.msg import Event, EventArray


class EventCameraEmulator(Node):
    def __init__(self):
        super().__init__("event_camera_emulator")

        self.declare_parameter("fire_rate", 200.0)
        self.declare_parameter("sensor_height", 480)
        self.declare_parameter("sensor_width", 640)
        self.declare_parameter("events_per_message", 200)
        self.declare_parameter("pattern_type", "random")
        self.declare_parameter("noise_level", 0.1)
        self.declare_parameter("polarity_bias", 0.5)

        self.fire_rate = float(self.get_parameter("fire_rate").value)
        self.sensor_height = int(self.get_parameter("sensor_height").value)
        self.sensor_width = int(self.get_parameter("sensor_width").value)
        self.events_per_message = int(self.get_parameter("events_per_message").value)
        self.pattern_type = str(self.get_parameter("pattern_type").value)
        self.noise_level = float(self.get_parameter("noise_level").value)
        self.polarity_bias = float(self.get_parameter("polarity_bias").value)

        self.publisher = self.create_publisher(EventArray, "/events", 10)
        self.buffer = []

        self.center_x = self.sensor_width / 2.0
        self.center_y = self.sensor_height / 2.0
        self.phase = 0.0

        self.timer = self.create_timer(1.0 / max(self.fire_rate, 1.0), self.generate_event)

        self.get_logger().info(
            f"Emulator started: rate={self.fire_rate}Hz, pattern={self.pattern_type}"
        )

    def _sample_xy(self):
        if self.pattern_type == "sinusoidal_ball":
            self.phase += 0.05
            cx = self.center_x + (self.sensor_width / 3.0) * np.sin(self.phase)
            cy = self.center_y + (self.sensor_height / 4.0) * np.sin(self.phase * 0.6)
            angle = random.uniform(0.0, 2.0 * np.pi)
            radius = random.uniform(0.0, 20.0)
            x = int(np.clip(cx + radius * np.cos(angle), 0, self.sensor_width - 1))
            y = int(np.clip(cy + radius * np.sin(angle), 0, self.sensor_height - 1))
            return x, y

        if self.pattern_type == "static_pattern":
            gx = random.choice(range(40, self.sensor_width - 40, 40))
            gy = random.choice(range(40, self.sensor_height - 40, 40))
            x = int(np.clip(gx + random.gauss(0, 8), 0, self.sensor_width - 1))
            y = int(np.clip(gy + random.gauss(0, 8), 0, self.sensor_height - 1))
            return x, y

        x = random.randint(0, self.sensor_width - 1)
        y = random.randint(0, self.sensor_height - 1)
        return x, y

    def generate_event(self):
        now_msg = self.get_clock().now().to_msg()

        x, y = self._sample_xy()

        prob = np.clip(self.polarity_bias + random.gauss(0.0, self.noise_level), 0.0, 1.0)
        polarity = bool(random.random() < prob)

        event = Event()
        event.x = x
        event.y = y
        event.ts = now_msg
        event.polarity = polarity

        self.buffer.append(event)

        if len(self.buffer) >= self.events_per_message:
            out = EventArray()
            out.header.stamp = now_msg
            out.header.frame_id = "event_camera"
            out.height = self.sensor_height
            out.width = self.sensor_width
            out.events = self.buffer
            self.publisher.publish(out)
            self.buffer = []


def main(args=None):
    rclpy.init(args=args)
    node = EventCameraEmulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
