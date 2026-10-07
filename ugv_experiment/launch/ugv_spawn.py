import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    UGV_MODEL = os.environ['UGV_MODEL']
 
    urdf_path_1 = os.path.join(
        get_package_share_directory('ugv_experiment'),
        'models',
        UGV_MODEL,
        'model_1.sdf'
    )
    urdf_path_2 = os.path.join(
        get_package_share_directory('ugv_experiment'),
        'models',
        UGV_MODEL,
        'model_2.sdf'
    )
    urdf_path_3 = os.path.join(
        get_package_share_directory('ugv_experiment'),
        'models',
        UGV_MODEL,
        'model_3.sdf'
    )

    # Declare the launch arguments
    declare_x_position_cmd = DeclareLaunchArgument(
        'x_pose', default_value='0.0',
        description='Specify namespace of the robot')

    declare_y_position_cmd = DeclareLaunchArgument(
        'y_pose', default_value='0.0',
        description='Specify namespace of the robot')

    start_gazebo_ros_spawner_cmd_1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        namespace='rover_1',
        arguments=[
            '-entity', 'ugv_rover_1',
            '-file', urdf_path_1,
            '-x','0.0',
            '-y','0.0',
            '-z','0.1',
            '-Y','0.785398'
        ],
        output='screen',
    )
    
    start_gazebo_ros_spawner_cmd_2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        namespace='rover_2',
        arguments=[
            '-entity', 'ugv_rover_2',
            '-file', urdf_path_2,
            '-x','3.0',
            '-y','0.0',
            '-z','0.1',
            '-Y','1.570796'
        ],
        output='screen',
    )
    
    start_gazebo_ros_spawner_cmd_3 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        namespace='rover_3',
        arguments=[
            '-entity', 'ugv_rover_3',
            '-file', urdf_path_3,
            '-x','0.0',
            '-y','4.0',
            '-z','0.1',
            '-Y','-0.197396'
        ],
        output='screen',
    )
    

    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_x_position_cmd)
    ld.add_action(declare_y_position_cmd)

    # Add any conditioned actions
    ld.add_action(start_gazebo_ros_spawner_cmd_1)
    ld.add_action(start_gazebo_ros_spawner_cmd_2)
    ld.add_action(start_gazebo_ros_spawner_cmd_3)
    return ld
