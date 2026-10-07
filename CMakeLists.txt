cmake_minimum_required(VERSION 3.5)

project(ugv_experiment)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)


install(DIRECTORY launch urdf worlds models 
    DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/rover_1_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/rover_2_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/rover_3_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/blue_cylinder_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/trajectory_logger.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/cbf_controller_1.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/cbf_controller_2.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/cbf_controller_3.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/cbf_trajectory.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/rover_1_tracker.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/rover_2_tracker.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/rover_3_tracker.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/tracker_trajectories.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/rover_1_cbf_tracker.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/rover_2_cbf_tracker.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/rover_3_cbf_tracker.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/cbf_tracker_trajectories.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/cbf_trajectory_2.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(ament_cmake)
ament_export_dependencies(ament_cmake_python)
ament_export_dependencies(rosidl_default_generators)

ament_package()

