# core_repo

Clean extraction of the project core:

- `offline_python/`: pure Python core for DCT/DTFT/DWT transforms, coefficient selection, density scoring, and threshold calibration.
- `ros2_ws/`: ROS2 workspace for adaptive transform selection development.

## Layout

```text
core_repo/
  offline_python/
    adaptive_core/
    scripts/
  ros2_ws/
    src/adaptive_event_core/
```

## Quick start

### Offline Python

```bash
cd core_repo/offline_python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/calibrate_thresholds.py --input ../../gt --output adaptive_core/config/adaptive_thresholds.json
python scripts/choose_transform_from_npz.py --file ../../data_example/00000.npz
```

### ROS2

```bash
cd core_repo/ros2_ws
colcon build --packages-select adaptive_event_core
source install/setup.bash
ros2 launch adaptive_event_core event_processing.launch.py use_emulator:=true
```

The ROS2 estimator reads thresholds from:
`ros2_ws/src/adaptive_event_core/config/adaptive_thresholds.json`.
