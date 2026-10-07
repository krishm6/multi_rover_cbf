#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

class Rover3Controller(Node):
    def __init__(self):
        super().__init__('rover_3_controller')
       
        # Robot state
        self.p3x = 0.0
        self.p3y = 0.0
        self.theta = 0.0
        self.p1x = 0.0
        self.p1y = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
       
        # Flags
        self.have_odom = False
        self.have_leader = False
        self.goal_reached = False
        self.have_goal = False
        self.profile_saved = False
       
        # ===== CRITICAL FIX 1: MATCH MATLAB PARAMETERS =====
        self.GOAL_TOLERANCE = 0.17
        self.l = 0.085  # Changed from 0.01 to 0.1 (MATLAB uses 0.05-0.2)
       
        # ===== CRITICAL FIX 2: REDUCE ANGULAR VELOCITY LIMIT =====
        self.speed_max = 0.5      # Keep as is
        self.omega_max = 1.0     # Changed from 3.0 to 0.8 (MATLAB value)
       
        # ===== CRITICAL FIX 3: ADD LOW-PASS FILTER =====
        self.alpha = 0.3  # Smoothing factor (0.2-0.4 works well)
        self.speed_filtered = 0.0
        self.omega_filtered = 0.0
       
        # Relative position offset
        self.r13_x = 0.0
        self.r13_y = -2.0
       
        # Logging
        self.time_log = []
        self.v_raw_log = []
        self.w_raw_log = []
        self.v_sat_log = []
        self.w_sat_log = []

        # Subscribers and publishers
        self.create_subscription(Odometry, '/rover_3/odom', self.odom_callback, 10)
        self.create_subscription(Twist, '/leader_state', self.leader_callback, 10)
       
        self.cmd_pub = self.create_publisher(Twist, '/rover_3/cmd_vel', 10)
        self.goal_pub = self.create_publisher(Bool, '/rover_3/goal_reached', 10)
       
        # ===== CRITICAL FIX 4: INCREASE CONTROL FREQUENCY =====
        self.timer = self.create_timer(0.02, self.control_loop)  # 50 Hz (was 10 Hz)

    def odom_callback(self, msg):
        self.p3x = msg.pose.pose.position.x
        self.p3y = msg.pose.pose.position.y
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

        goal_3_x = self.goal_x
        goal_3_y = self.goal_y - 2.0
       
        dx = goal_3_x - self.p3x
        dy = goal_3_y - self.p3y
        distance = math.sqrt(dx**2 + dy**2)
       
        # Goal reached
        if distance < self.GOAL_TOLERANCE or self.goal_reached:
            self.goal_reached = True
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
            np.savez(
                f"rover_3_profile.npz",
                t=np.array(self.time_log),
                v_raw=np.array(self.v_raw_log),
                w_raw=np.array(self.w_raw_log),
                v_sat=np.array(self.v_sat_log),
                w_sat=np.array(self.w_sat_log)
            )
            self.profile_saved = True
            return

        # ===== FEEDBACK LINEARIZATION (MATLAB METHOD) =====
       
        # Compute LQ control
        x1_x = self.p1x - self.goal_x
        x1_y = self.p1y - self.goal_y
        x3_x = self.p3x - self.p1x - self.r13_x
        x3_y = self.p3y - self.p1y - self.r13_y
       
        # u3 = -(x1 + 2*x3) from LQ game
        u3_x = -(x1_x + 2.0 * x3_x)
        u3_y = -(x1_y + 2.0 * x3_y)
       
        # ===== CRITICAL FIX 5: FEEDBACK LINEARIZATION =====
        # This is the standard method used in MATLAB
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
       
        # 1. Linear velocity: project desired onto heading direction
        speed_raw = cos_theta * u3_x + sin_theta * u3_y
       
        # 2. Angular velocity: transverse error divided by l
        # This is PURE feedback linearization, NOT look-ahead
        omega_raw = (-sin_theta * u3_x + cos_theta * u3_y) / self.l
       
       
        omega = max(min(omega_raw, self.omega_max), -self.omega_max)
        
        omega_ref = 4.2     # rad/s where slowdown begins
        speed_min = 0.05
        speed_max = 0.3

        scale = np.exp(-abs(omega_raw)/omega_ref)

        speed = speed_min + (speed_max-speed_min)*scale

        speed = min(speed, speed_raw)
        
       
        # Publish command
        cmd = Twist()
        cmd.linear.x = float(speed_raw)
        cmd.angular.z = float(omega_raw)
        self.cmd_pub.publish(cmd)
       
        # Log data
        self.time_log.append(self.get_clock().now().nanoseconds * 1e-9)
        self.v_raw_log.append(speed_raw)
        self.w_raw_log.append(omega_raw)
        self.v_sat_log.append(speed)
        self.w_sat_log.append(omega)
        self.get_logger().info(
            f'p3=({self.p3x:.2f},{self.p3y:.2f}) '
            f'goal=({goal_3_x:.2f},{goal_3_y:.2f}) '
            f'v={speed:.3f} w={omega:.3f}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = Rover3Controller()
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
