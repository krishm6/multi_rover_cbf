from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='ugv_experiment',
            executable='rover_1_controller.py',
            name='rover_1_controller',
            output='screen'
        ),

        Node(
            package='ugv_experiment',
            executable='rover_2_controller.py',
            name='rover_2_controller',
            output='screen'
        ),

        Node(
            package='ugv_experiment',
            executable='rover_3_controller.py',
            name='rover_3_controller',
            output='screen'
        ),

    ])
