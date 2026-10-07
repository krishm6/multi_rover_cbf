
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess

def generate_launch_description():
    # Get the directory of the launch file
    launch_file_dir = os.path.join(get_package_share_directory('ugv_experiment'), 'launch')
    # Get the directory of the gazebo_ros package
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # Get the use_sim_time parameter from the launch file
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Get the world file
    world = os.path.join(
        get_package_share_directory('ugv_experiment'),
        'worlds',
        'experiment.world'
    )

    # Include the gzserver launch file
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    # Include the gzclient launch file
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )
            
    # Include the robot_state_publisher launch file
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Include the spawn_ugv launch file
    spawn_ugv_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'ugv_spawn.py')
        )
    )

    # Create a launch description
    ld = LaunchDescription()

    # Add the commands to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_ugv_cmd)

    return ld
