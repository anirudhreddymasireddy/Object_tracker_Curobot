from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        DeclareLaunchArgument(
            'detection_topic',
            default_value='/detectnet/detections',
            description='Detection2DArray topic to subscribe to.',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/detectnet/image_raw',
            description='Camera image topic to overlay detections on.',
        ),
        DeclareLaunchArgument(
            'display_fps',
            default_value='30.0',
            description='Target display refresh rate (Hz).',
        ),

        Node(
            package='kf_object_tracking',
            executable='detection2d_visualizer',
            name='detection2d_visualizer',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'detection_topic': LaunchConfiguration('detection_topic'),
                'image_topic':     LaunchConfiguration('image_topic'),
                'display_fps':     LaunchConfiguration('display_fps'),
            }],
        ),
    ])
