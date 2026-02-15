# adaptive_event_core (ROS2)

ROS2 package for adaptive transform selection:

- subscribes to `/events` (custom `EventArray`)
- publishes `/event_density` (`std_msgs/Float32`)
- publishes `/selected_transform` (`std_msgs/String`)
- logs transform switches to CSV

## Build

```bash
cd core_repo/ros2_ws
colcon build --packages-select adaptive_event_core
source install/setup.bash
```

## Run

```bash
ros2 launch adaptive_event_core event_processing.launch.py use_emulator:=true
```

## Nodes

- `event_camera_emulator.py`
- `event_density_estimator.py`
- `transform_writer.py`
