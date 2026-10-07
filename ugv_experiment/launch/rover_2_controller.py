#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

class Rover2Controller(Node):
    def __init__(self):
        super().__init__('rover_2_controller')

        # ===============================
        # States
        # ===============================
        self.p2x = 0.0
        self.p2y = 0.0
        self.theta = 0.0

        self.p1x = 0.0
        self.p1y = 0.0

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.have_odom = False
        self.have_leader = False
        self.have_goal = False
        self.goal_reached = False
        self.GOAL_TOLERANCE = 0.17
        self.profile_saved = False

        # ===== FIX 1: MATCH MATLAB PARAMETERS =====
        # Feedback linearization gain (MATLAB uses 0.05-0.2)
        self.l = 0.085  # Changed from 0.6
       
        # ===== FIX 2: REDUCE ANGULAR VELOCITY LIMIT =====
        self.speed_max = 0.5      # Keep as is
        self.omega_max = 1.0     # Changed from 3.0 to 0.8 (MATLAB value)
       
        # ===== FIX 3: ADD LOW-PASS FILTER =====
        self.alpha = 0.3  # Smoothing factor
        self.speed_filtered = 0.0
        self.omega_filtered = 0.0

        # ===== FIX 4: REMOVE ALIGNMENT PHASE =====
        # MATLAB doesn't have separate alignment phase
        # self.aligned = False
        # self.heading_initialised = False
        # self.theta_des = 0

        # Relative position offsets
        self.r12_x = -2.0
        self.r12_y = 0.0

        # Logging
        self.time_log = []
        self.v_raw_log = []
        self.v_sat_log = []
        self.w_raw_log = []
        self.w_sat_log = []
        

        # ===============================
        # Subscribers
        # ===============================
        self.create_subscription(
            Odometry,
            '/rover_2/odom',
            self.odom_callback,
            10
        )

        self.create_subscription(
            Twist,
            '/leader_state',
            self.leader_callback,
            10
        )

        # ===============================
        # Publishers
        # ===============================
        self.cmd_pub = self.create_publisher(
            Twist,
            '/rover_2/cmd_vel',
            10
        )
       
        self.goal_pub = self.create_publisher(
            Bool,
            '/rover_2/goal_reached',
            10
        )

        # ===============================
        # Control Timer
        # ===============================
        # ===== FIX 5: INCREASE CONTROL FREQUENCY =====
        self.timer = self.create_timer(
            0.02,  # Changed from 0.1 to 0.02 (50 Hz)
            self.control_loop
        )

        self.get_logger().info('Rover 2 Controller Started')

    def odom_callback(self, msg):
        self.p2x = msg.pose.pose.position.x
        self.p2y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.theta = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.have_odom = True

    def leader_callback(self, msg):
        if self.have_goal:
            self.p1x = msg.linear.x
            self.p1y = msg.linear.y
            return

        self.p1x = msg.linear.x
        self.p1y = msg.linear.y
        self.goal_x = msg.angular.x
        self.goal_y = msg.angular.y
        self.have_leader = True
        self.have_goal = True

    def control_loop(self):
        if not self.have_odom or not self.have_leader:
            return

        # Goal for rover 2: p1* + r12
        goal_2_x = self.goal_x + self.r12_x
        goal_2_y = self.goal_y + self.r12_y
       
        dx = goal_2_x - self.p2x
        dy = goal_2_y - self.p2y
        distance = math.sqrt(dx**2 + dy**2)
       
        # Goal reached check
        if distance < self.GOAL_TOLERANCE:
            self.goal_reached = True
           
        if self.goal_reached:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.get_logger().info(f"Goal reached. Distance = {distance:.3f} m")
           
            msg = Bool()
            msg.data = True
            self.goal_pub.publish(msg)
            
            if self.profile_saved:
                return
           
            # Save data with filtered values
            np.savez(
                f"rover_2_profile.npz",
                t=np.array(self.time_log),
                v_raw=np.array(self.v_raw_log),
                w_raw=np.array(self.w_raw_log),
                v_sat=np.array(self.v_sat_log),
                w_sat=np.array(self.w_sat_log)
            )
            self.profile_saved = True
            return

        # ===============================
        # LQ CONTROL: u2 = -(x1 + 2*x2)
        # x1 = p1 - p1*, x2 = (p2 - p1) - r12
        # ===============================
        x1_x = self.p1x - self.goal_x
        x1_y = self.p1y - self.goal_y

        x2_x = (self.p2x - self.p1x) - self.r12_x
        x2_y = (self.p2y - self.p1y) - self.r12_y

        u2_x = -(x1_x + 2.0 * x2_x)
        u2_y = -(x1_y + 2.0 * x2_y)

        # ===== FIX 6: FEEDBACK LINEARIZATION (MATLAB METHOD) =====
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)

        # 1. Linear velocity: project desired velocity onto heading
        speed_raw = cos_theta * u2_x + sin_theta * u2_y

        # 2. Angular velocity: transverse error divided by l
        # This is PURE feedback linearization, NOT look-ahead point tracking
        omega_raw = (-sin_theta * u2_x + cos_theta * u2_y) / self.l


        omega = max(min(omega_raw, self.omega_max), -self.omega_max)
        
        omega_ref = 10.0      # rad/s where slowdown begins
        speed_min = 0.05
        speed_max = 0.3

        scale = np.exp(-abs(omega_raw)/omega_ref)

        speed = speed_min + (speed_max-speed_min)*scale

        speed = min(speed, speed_raw)

        # Log data
        self.time_log.append(self.get_clock().now().nanoseconds * 1e-9)
        self.v_raw_log.append(speed_raw)
        self.w_raw_log.append(omega_raw)
        self.v_sat_log.append(speed)
        self.w_sat_log.append(omega)

        # ===============================
        # Publish cmd_vel
        # ===============================
        cmd = Twist()
        cmd.linear.x = float(speed_raw)
        cmd.angular.z = float(omega_raw)
        self.cmd_pub.publish(cmd)
       
        self.get_logger().info(
            f'p2=({self.p2x:.2f},{self.p2y:.2f}) '
            f'goal=({goal_2_x:.2f},{goal_2_y:.2f}) '
            f'v={speed:.3f} w={omega:.3f}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = Rover2Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
