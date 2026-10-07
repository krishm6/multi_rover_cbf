#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo

def generate_launch_description():
    return LaunchDescription([
        # Log info
        LogInfo(msg=['Launching Rover Blind Trackers...']),
        
        # Rover 1 Tracker
        Node(
            package='ugv_experiment',
            executable='rover_1_tracker.py',
            name='rover_1_tracker',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': True
            }]
        ),
        
        # Rover 2 Tracker
        Node(
            package='ugv_experiment',
            executable='rover_2_tracker.py',
            name='rover_2_tracker',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': True
            }]
        ),
        
        # Rover 3 Tracker
        Node(
            package='ugv_experiment',
            executable='rover_3_tracker.py',
            name='rover_3_tracker',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': True
            }]
        ),
        
        # Trajectory Logger
        Node(
            package='ugv_experiment',
            executable='tracker_trajectories.py',
            name='tracker_trajectories',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': True
            }]
        ),
        
        LogInfo(msg=['All trackers launched! Press Ctrl+C to stop.'])
    ])
