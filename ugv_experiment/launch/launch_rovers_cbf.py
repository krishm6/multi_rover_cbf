#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo

def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg=['Launching CBF-QP theoretical trackers and plotter...']),

        # Rover 1 CBF tracker
        Node(
            package='ugv_experiment',
            executable='rover_1_cbf_tracker.py',
            name='rover_1_cbf_tracker',
            output='screen',
            emulate_tty=True,
            parameters=[{'use_sim_time': True}]
        ),

        # Rover 2 CBF tracker
        Node(
            package='ugv_experiment',
            executable='rover_2_cbf_tracker.py',
            name='rover_2_cbf_tracker',
            output='screen',
            emulate_tty=True,
            parameters=[{'use_sim_time': True}]
        ),

        # Rover 3 CBF tracker
        Node(
            package='ugv_experiment',
            executable='rover_3_cbf_tracker.py',
            name='rover_3_cbf_tracker',
            output='screen',
            emulate_tty=True,
            parameters=[{'use_sim_time': True}]
        ),

        # CBF trajectory plotter (waits for all rovers to finish)
        Node(
            package='ugv_experiment',
            executable='cbf_tracker_trajectories.py',
            name='cbf_tracker_trajectories',
            output='screen',
            emulate_tty=True,
            parameters=[{'use_sim_time': True}]
        ),

        LogInfo(msg=['All CBF nodes started. Waiting for rovers to reach goals...'])
    ])
