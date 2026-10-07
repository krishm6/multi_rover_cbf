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

class Rover1Controller(Node):
    def __init__(self):
        super().__init__('rover_1_controller')

        # ===============================
        # States
        # ===============================
        self.px = 0.0
        self.py = 0.0
        self.theta = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0

        self.have_odom = False
        self.have_goal = False
        self.goal_reached = False
        self.GOAL_TOLERANCE = 0.3
        self.profile_saved = False

        # ===== FIX 1: MATCH MATLAB PARAMETERS =====
        self.l = 0.085  # Changed from 0.8
        self.speed_max = 0.5
        self.omega_max = 1.0  # Changed from 3.0
       
        # ===== FIX 2: ADD LOW-PASS FILTER =====
        self.alpha = 0.3
        self.speed_filtered = 0.0
        self.omega_filtered = 0.0

        # ===== FIX 3: REMOVE ALIGNMENT PHASE =====
       
        # ===============================
        # LOGGING - EXACTLY LIKE ROVER 2 & 3
        # ===============================
        self.time_log = []
        self.v_raw_log = []
        self.w_raw_log = []
        self.v_sat_log = []
        self.w_sat_log = []

        # ===============================
        # Subscribers
        # ===============================
        self.odom_sub = self.create_subscription(
            Odometry,
            '/rover_1/odom',
            self.odom_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        # ===============================
        # Publishers
        # ===============================
        self.cmd_pub = self.create_publisher(
            Twist,
            '/rover_1/cmd_vel',
            10
        )

        self.leader_pub = self.create_publisher(
            Twist,
            '/leader_state',
            10
        )
       
        self.goal_pub = self.create_publisher(
            Bool,
            '/rover_1/goal_reached',
            10
        )

        # ===============================
        # Control Timer
        # ===============================
        # ===== FIX 4: INCREASE CONTROL FREQUENCY =====
        self.timer = self.create_timer(
            0.02,  # Changed from 0.1
            self.control_loop
        )

        self.get_logger().info('Rover 1 Controller Started')

    # ==================================
    # ODOM CALLBACK
    # ==================================
    def odom_callback(self, msg):
        self.px = msg.pose.pose.position.x
        self.py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.theta = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.have_odom = True

    # ==================================
    # GOAL CALLBACK
    # ==================================
    def goal_callback(self, msg):
        if self.have_goal:
            return
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.have_goal = True

    # ==================================
    # CONTROL LOOP
    # ==================================
    def control_loop(self):
        if not self.have_odom or not self.have_goal:
            return

        dx = self.goal_x - self.px
        dy = self.goal_y - self.py
        distance = math.sqrt(dx**2 + dy**2)

        if distance < self.GOAL_TOLERANCE:
            self.goal_reached = True

        if self.goal_reached:       
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
           
            leader_msg = Twist()
            leader_msg.linear.x = self.px
            leader_msg.linear.y = self.py
            leader_msg.angular.x = self.goal_x
            leader_msg.angular.y = self.goal_y
            self.leader_pub.publish(leader_msg)
           
            self.get_logger().info(f"Goal reached. Distance = {distance:.3f} m")
            msg = Bool()
            msg.data = True
            self.goal_pub.publish(msg)
           
            if self.profile_saved:
                return
            # ===== SAVE EXACTLY LIKE ROVER 2 & 3 =====
            np.savez(
                f"rover_1_profile.npz",
                t=np.array(self.time_log),
                v_raw=np.array(self.v_raw_log),
                w_raw=np.array(self.w_raw_log),
                v_sat=np.array(self.v_sat_log),
                w_sat=np.array(self.w_sat_log)
            )
            self.profile_saved = True
            return

        # ===============================
        # LQ CONTROL: u1 = -2*x1
        # ===============================
        x1_x = self.px - self.goal_x
        x1_y = self.py - self.goal_y

        u1_x = -2.0 * x1_x
        u1_y = -2.0 * x1_y

        # ===== FEEDBACK LINEARIZATION (MATLAB METHOD) =====
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)

        speed_raw = cos_theta * u1_x + sin_theta * u1_y

        omega_raw = (-sin_theta * u1_x + cos_theta * u1_y) / self.l

        omega = max(min(omega_raw, self.omega_max), -self.omega_max)
        
        omega_ref = 100.0      # rad/s where slowdown begins
        speed_min = 0.05
        speed_max = 0.3

        scale = np.exp(-abs(omega_raw)/omega_ref)

        speed = speed_min + (speed_max-speed_min)*scale

        speed = min(speed, speed_raw)

        # ===== LOGGING - EXACTLY LIKE ROVER 2 & 3 =====
        self.time_log.append(self.get_clock().now().nanoseconds * 1e-9)
        self.v_raw_log.append(speed_raw)
        self.w_raw_log.append(omega_raw)
        self.v_sat_log.append(speed)
        self.w_sat_log.append(omega)

        # ===============================
        # Publish cmd_vel
        # ===============================
        cmd = Twist()
        cmd.linear.x = speed_raw
        cmd.angular.z = omega_raw
        self.cmd_pub.publish(cmd)

        # ===============================
        # Publish leader_state
        # ===============================
        leader_msg = Twist()
        leader_msg.linear.x = self.px
        leader_msg.linear.y = self.py
        leader_msg.angular.x = self.goal_x
        leader_msg.angular.y = self.goal_y
        self.leader_pub.publish(leader_msg)

        self.get_logger().info(
            f'p1=({self.px:.2f},{self.py:.2f}) '
            f'goal=({self.goal_x:.2f},{self.goal_y:.2f})'
            f'v={speed:.3f} w={omega:.3f}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = Rover1Controller()
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

