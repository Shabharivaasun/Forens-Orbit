import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('forensorbit_detection')
    params_file = os.path.join(pkg_share, 'config', 'params.yaml')

    return LaunchDescription([
        Node(
            package='forensorbit_detection',
            executable='camera_capture_node',
            name='camera_capture_node',
            output='screen',
            parameters=[params_file]
        ),
        Node(
            package='forensorbit_detection',
            executable='evidence_detector_node',
            name='evidence_detector_node',
            output='screen',
            parameters=[params_file]
        ),
        Node(
            package='forensorbit_detection',
            executable='report_sender_node',
            name='report_sender_node',
            output='screen',
            parameters=[params_file]
        ),
    ])
