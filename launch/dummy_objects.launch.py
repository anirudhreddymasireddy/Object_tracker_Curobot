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
            'config', 'params.yaml',
        ]),
        description='Parameters file for the tracker node.',
    )

    scenario_arg = DeclareLaunchArgument(
        'scenario',
        default_value='0',
        description=(
            'Which dummy-object scenario to run: '
            '0=all four objects, 1=linear, 2=circular, 3=diagonal, 4=zigzag'
        ),
    )

    dummy_node = Node(
        package='kf_object_tracking',
        executable='dummy_objects_publisher',
        name='dummy_objects_publisher',
        output='screen',
        parameters=[{
            'publish_rate': 10.0,
            'noise_std':    0.05,
            'frame_id':     'map',
            'scenario':     LaunchConfiguration('scenario'),
        }],
    )

    tracker_node = Node(
        package='kf_object_tracking',
        executable='kf_object_tracker',
        name='kf_object_tracker',
        output='screen',
        emulate_tty=True,
        parameters=[LaunchConfiguration('params_file')],
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
        scenario_arg,
        dummy_node,
        tracker_node,
        visualizer_node,
    ])
