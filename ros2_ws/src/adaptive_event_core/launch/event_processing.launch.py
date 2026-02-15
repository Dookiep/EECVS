from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory("adaptive_event_core")

    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=os.path.join(pkg_dir, "config", "event_params.yaml"),
        description="Path to parameter YAML",
    )
    use_emulator_arg = DeclareLaunchArgument(
        "use_emulator",
        default_value="true",
        description="Launch emulator node",
    )
    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="ROS log level",
    )

    emulator_node = Node(
        package="adaptive_event_core",
        executable="event_camera_emulator.py",
        name="event_camera_emulator",
        parameters=[LaunchConfiguration("config_file")],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_emulator")),
    )

    estimator_node = Node(
        package="adaptive_event_core",
        executable="event_density_estimator.py",
        name="event_density_estimator",
        parameters=[LaunchConfiguration("config_file")],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        output="screen",
    )

    writer_node = Node(
        package="adaptive_event_core",
        executable="transform_writer.py",
        name="transform_writer",
        parameters=[LaunchConfiguration("config_file")],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        output="screen",
    )

    return LaunchDescription([
        config_file_arg,
        use_emulator_arg,
        log_level_arg,
        emulator_node,
        estimator_node,
        writer_node,
    ])
