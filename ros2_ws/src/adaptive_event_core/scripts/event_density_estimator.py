#!/usr/bin/env python3

import json
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float32, String

from adaptive_event_core.msg import EventArray


class EventDensityEstimator(Node):
    def __init__(self):
        super().__init__("event_density_estimator")

        pkg_share = Path(get_package_share_directory("adaptive_event_core"))
        default_cfg = str(pkg_share / "config" / "adaptive_thresholds.json")

        self.declare_parameter("window_duration", 0.1)
        self.declare_parameter("sensor_height", 480)
        self.declare_parameter("sensor_width", 640)
        self.declare_parameter("smoothing_window", 10)
        self.declare_parameter("hysteresis_margin", 0.05)
        self.declare_parameter("spatial_window_size", 32)
        self.declare_parameter("calibration_config_path", default_cfg)

        self.window_duration = float(self.get_parameter("window_duration").value)
        self.sensor_height = int(self.get_parameter("sensor_height").value)
        self.sensor_width = int(self.get_parameter("sensor_width").value)
        self.smoothing_window = int(self.get_parameter("smoothing_window").value)
        self.hysteresis_margin = float(self.get_parameter("hysteresis_margin").value)
        self.spatial_window_size = int(self.get_parameter("spatial_window_size").value)

        cfg_path = self.get_parameter("calibration_config_path").value
        self.sparse_threshold, self.dense_threshold = self._load_thresholds(cfg_path)

        self.event_buffer = deque()
        self.density_history = deque(maxlen=max(1, self.smoothing_window))
        self.current_transform = "DTFT"

        self.create_subscription(EventArray, "/events", self.event_callback, 10)
        self.density_pub = self.create_publisher(Float32, "/event_density", 10)
        self.transform_pub = self.create_publisher(String, "/selected_transform", 10)
        self.timer = self.create_timer(self.window_duration, self.process_window)

        self.get_logger().info(
            "Estimator ready with thresholds "
            f"sparse={self.sparse_threshold:.3f}, dense={self.dense_threshold:.3f}"
        )

    def _load_thresholds(self, cfg_path: str):
        p = Path(cfg_path)
        if not p.is_absolute() and not p.exists():
            pkg_share = Path(get_package_share_directory("adaptive_event_core"))
            p = pkg_share / cfg_path

        if not p.exists():
            self.get_logger().warn(f"Threshold config not found at {p}; using defaults")
            return 0.234, 0.630

        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if "density_thresholds" in data:
                d = data["density_thresholds"]
                return float(d["sparse_threshold"]), float(d["dense_threshold"])

            return float(data["sparse_threshold"]), float(data["dense_threshold"])
        except Exception as exc:
            self.get_logger().warn(f"Failed to parse threshold config: {exc}; using defaults")
            return 0.234, 0.630

    def event_callback(self, msg: EventArray):
        now_s = self.get_clock().now().nanoseconds / 1e9
        for ev in msg.events:
            self.event_buffer.append(
                {
                    "timestamp": now_s,
                    "x": int(ev.x),
                    "y": int(ev.y),
                    "p": 1 if ev.polarity else -1,
                }
            )

    def process_window(self):
        now_s = self.get_clock().now().nanoseconds / 1e9
        window_start = now_s - self.window_duration

        while self.event_buffer and self.event_buffer[0]["timestamp"] < window_start:
            self.event_buffer.popleft()

        if not self.event_buffer:
            self._publish_density_and_transform(0.0, "DWT")
            return

        events = list(self.event_buffer)
        density = self._compute_density_score(events)
        self.density_history.append(density)
        smooth_density = float(np.mean(self.density_history))

        new_transform = self._select_with_hysteresis(smooth_density)
        self._publish_density_and_transform(smooth_density, new_transform)

    def _publish_density_and_transform(self, density: float, transform: str):
        density_msg = Float32()
        density_msg.data = float(density)
        self.density_pub.publish(density_msg)

        if transform != self.current_transform:
            self.current_transform = transform
            msg = String()
            msg.data = transform
            self.transform_pub.publish(msg)
            self.get_logger().info(
                f"Transform changed to {transform} (density={density:.4f})"
            )

    def _compute_density_score(self, events):
        x = np.array([e["x"] for e in events], dtype=np.float64)
        y = np.array([e["y"] for e in events], dtype=np.float64)
        t = np.array([e["timestamp"] for e in events], dtype=np.float64)

        spatial_density = self._spatial_density(x, y)
        t_duration = max(float(np.max(t) - np.min(t)), 1e-6)
        event_rate = len(events) / t_duration
        temporal_variance = float(np.var(t)) if len(t) > 1 else 0.0
        clustering = self._spatial_clustering(x, y)

        spatial_norm = np.clip(spatial_density / 100.0, 0, 1)
        rate_norm = np.clip(event_rate / 10000.0, 0, 1)
        cluster_norm = np.clip(clustering, 0, 1)
        temporal_norm = np.clip(temporal_variance / 1000.0, 0, 1)

        return float(
            0.4 * spatial_norm
            + 0.3 * rate_norm
            + 0.2 * cluster_norm
            + 0.1 * temporal_norm
        )

    def _spatial_density(self, x, y):
        if x.size == 0:
            return 0.0

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        x_range = max(x_max - x_min, 1)
        y_range = max(y_max - y_min, 1)

        bins_x = max(1, int(x_range / self.spatial_window_size))
        bins_y = max(1, int(y_range / self.spatial_window_size))

        hist, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y], range=[[x_min, x_max], [y_min, y_max]])

        nonzero = np.sum(hist > 0)
        total = bins_x * bins_y
        occupancy = nonzero / total if total > 0 else 0.0
        mean_density = np.mean(hist[hist > 0]) if nonzero > 0 else 0.0
        max_possible = len(x) / nonzero if nonzero > 0 else 1.0

        return float((occupancy + (mean_density / max_possible)) / 2.0)

    def _spatial_clustering(self, x, y):
        if x.size < 2:
            return 0.0

        rng = np.random.default_rng(42)
        n_sample = min(200, x.size)
        idx = rng.choice(x.size, n_sample, replace=False)
        xs, ys = x[idx], y[idx]

        xs = (xs - np.min(xs)) / (np.max(xs) - np.min(xs) + 1e-8)
        ys = (ys - np.min(ys)) / (np.max(ys) - np.min(ys) + 1e-8)
        coords = np.column_stack([xs, ys])

        n_points = len(coords)
        n_pairs = min(100, n_points * (n_points - 1) // 2)
        if n_pairs <= 0:
            return 0.0

        dists = np.empty(n_pairs, dtype=np.float64)
        for i in range(n_pairs):
            a, b = rng.choice(n_points, 2, replace=False)
            dists[i] = np.sqrt(((coords[a] - coords[b]) ** 2).sum())

        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        return float(1.0 / (1.0 + std_dist / (mean_dist + 1e-8)))

    def _select_with_hysteresis(self, score: float):
        if self.current_transform == "DWT":
            sparse_t = self.sparse_threshold + self.hysteresis_margin
            dense_t = self.dense_threshold
        elif self.current_transform == "DCT":
            sparse_t = self.sparse_threshold
            dense_t = self.dense_threshold - self.hysteresis_margin
        else:
            sparse_t = self.sparse_threshold - self.hysteresis_margin
            dense_t = self.dense_threshold + self.hysteresis_margin

        if score <= sparse_t:
            return "DWT"
        if score >= dense_t:
            return "DCT"
        return "DTFT"


def main(args=None):
    rclpy.init(args=args)
    node = EventDensityEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
