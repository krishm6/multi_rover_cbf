from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([

        Node(
            package='ugv_experiment',
            executable='cbf_controller_1.py',
            name='cbf_controller_1',
            output='screen'
        ),

        Node(
            package='ugv_experiment',
            executable='cbf_controller_2.py',
            name='cbf_controller_2',
            output='screen'
        ),

        Node(
            package='ugv_experiment',
            executable='cbf_controller_3.py',
            name='cbf_controller_3',
            output='screen'
        ),
        Node(
            package='ugv_experiment',
            executable='cbf_trajectory_2.py',
            name='cbf_trajectory_logger',
            output='screen'
        ),
    ])
