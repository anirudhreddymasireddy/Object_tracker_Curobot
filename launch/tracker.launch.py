from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('kf_object_tracking'),
            'config', 'params.yaml'
        ]),
        description='Full path to the ROS2 parameters file to use for the tracker node.',
    )

    tracker_node = Node(
        package='kf_object_tracking',
        executable='kf_object_tracker',
        name='kf_object_tracker',
        output='screen',
        emulate_tty=True,
        parameters=[LaunchConfiguration('params_file')],
        remappings=[
            # ('/object_localization/detections3d', '/your/detection_topic'),
        ],
    )

    visualizer_node = Node(
        package='kf_object_tracking',
        executable='tracker_visualizer',
        name='tracker_visualizer',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        params_file_arg,
        tracker_node,
        visualizer_node,
    ])
