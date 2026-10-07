#!/usr/bin/env python3

import csv
import pandas as pd
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


class TrajectoryLogger(Node):

    def __init__(self):

        super().__init__('trajectory_logger')
        self.r1_done = False
        self.r2_done = False
        self.r3_done = False

        # Rover 1 CSV
        self.file1 = open('rover_1.csv', 'w', newline='')
        self.writer1 = csv.writer(self.file1)
        self.writer1.writerow(['time', 'x', 'y'])

        # Rover 2 CSV
        self.file2 = open('rover_2.csv', 'w', newline='')
        self.writer2 = csv.writer(self.file2)
        self.writer2.writerow(['time', 'x', 'y'])

        # Rover 3 CSV
        self.file3 = open('rover_3.csv', 'w', newline='')
        self.writer3 = csv.writer(self.file3)
        self.writer3.writerow(['time', 'x', 'y'])
        
        self.dt = 0.01

        self.V_THRESHOLD = 0.02
        self.W_THRESHOLD = 0.05

        # -----------------------
        # START DETECTION
        # -----------------------
        self.started_1 = False
        self.started_2 = False
        self.started_3 = False
        
        self.start_idx_1 = None
        self.start_idx_2 = None
        self.start_idx_3 = None
        
        self.sample_count_1 = 0
        self.sample_count_2 = 0
        self.sample_count_3 = 0
        
        # -----------------------
        # LOGS
        # -----------------------
        self.time_log_1 = []

        self.x_log_1 = []
        self.y_log_1 = []

        self.v_log_1 = []
        self.w_log_1 = []
        
        self.time_log_2 = []

        self.x_log_2 = []
        self.y_log_2 = []

        self.v_log_2 = []
        self.w_log_2 = []
        
        self.time_log_3 = []

        self.x_log_3 = []
        self.y_log_3 = []

        self.v_log_3 = []
        self.w_log_3 = []
        
        self.D_SAFE = 0.7

        # Subscribers
        self.sub1 = self.create_subscription(
            Odometry,
            '/rover_1/odom',
            self.rover1_callback,
            10
        )

        self.sub2 = self.create_subscription(
            Odometry,
            '/rover_2/odom',
            self.rover2_callback,
            10
        )

        self.sub3 = self.create_subscription(
            Odometry,
            '/rover_3/odom',
            self.rover3_callback,
            10
        )
        
        self.goal_sub1 = self.create_subscription(
            Bool,
            '/rover_1/goal_reached',
            self.goal1_callback,
            10
        )

        self.goal_sub2 = self.create_subscription(
            Bool,
            '/rover_2/goal_reached',
            self.goal2_callback,
            10
        )

        self.goal_sub3 = self.create_subscription(
            Bool,
            '/rover_3/goal_reached',
            self.goal3_callback,
            10
        )

        self.get_logger().info("Trajectory Logger Started")

    def rover1_callback(self, msg):

        t = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self.writer1.writerow([t, x, y])
        
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        
        if (
            not self.started_1 and abs(v) > self.V_THRESHOLD      
        ):

            self.started_1 = True
            self.start_idx_1 = self.sample_count_1

            self.get_logger().info(
                f"Motion detected at sample {self.start_idx_1}"
            )
        if self.started_1:

            t_rel = (
                self.sample_count_1 - self.start_idx_1
            ) * self.dt

            self.time_log_1.append(t_rel)

            self.x_log_1.append(x)
            self.y_log_1.append(y)

            self.v_log_1.append(v)
            self.w_log_1.append(w)

        self.sample_count_1 += 1

    def rover2_callback(self, msg):

        t = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self.writer2.writerow([t, x, y])
        
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        
        if (
            not self.started_2 and abs(v) > self.V_THRESHOLD   
        ):
            self.started_2 = True
            self.start_idx_2 = self.sample_count_2

            self.get_logger().info(
                f"Motion detected at sample {self.start_idx_2}"
            )
        if self.started_2:

            t_rel = (
                self.sample_count_2 - self.start_idx_2
            ) * self.dt

            self.time_log_2.append(t_rel)

            self.x_log_2.append(x)
            self.y_log_2.append(y)

            self.v_log_2.append(v)
            self.w_log_2.append(w)

        self.sample_count_2 += 1

    def rover3_callback(self, msg):

        t = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self.writer3.writerow([t, x, y])
        
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        
        if (
            not self.started_3 and abs(v) > self.V_THRESHOLD
        ):

            self.started_3 = True
            self.start_idx_3 = self.sample_count_3

            self.get_logger().info(
                f"Motion detected at sample {self.start_idx_3}"
            )
        if self.started_3:

            t_rel = (
                self.sample_count_3 - self.start_idx_3
            ) * self.dt

            self.time_log_3.append(t_rel)

            self.x_log_3.append(x)
            self.y_log_3.append(y)

            self.v_log_3.append(v)
            self.w_log_3.append(w)

        self.sample_count_3 += 1
        
    def goal1_callback(self, msg):

        self.r1_done = msg.data
        self.check_finished()


    def goal2_callback(self, msg):

        self.r2_done = msg.data
        self.check_finished()


    def goal3_callback(self, msg):

        self.r3_done = msg.data
        self.check_finished()
    
    def check_finished(self):

        if (
            self.r1_done and
            self.r2_done and
            self.r3_done
        ):

            self.get_logger().info(
                "All rovers reached goals."
            )

            self.destroy_node()
            self.save_logs()
            self.plot_trajectories()
            rclpy.shutdown()
            
    def save_logs(self):

        if len(self.time_log_1) == 0:
            self.get_logger().warn(
                "No trajectory recorded for rover 1"
            )
            return
                
        if len(self.time_log_2) == 0:
            self.get_logger().warn(
                "No trajectory recorded for rover 2"
            )
            return
                
        if len(self.time_log_3) == 0:
            self.get_logger().warn(
                "No trajectory recorded for rover 3"
            )
            return
                 
                
        T_final_1 = self.time_log_1[-1]
        T_final_2 = self.time_log_2[-1]
        T_final_3 = self.time_log_3[-1]

        self.get_logger().info(
            f"Final mission time of rover 1 = {T_final_1:.3f}"
        )
            
        self.get_logger().info(
            f"Final mission time of rover 2 = {T_final_2:.3f}"
        )
            
        self.get_logger().info(
            f"Final mission time of rover 3 = {T_final_3:.3f}"
        )

        np.savez(
            "rover_1_log.npz",

            t=np.array(self.time_log_1),

            x=np.array(self.x_log_1),
            y=np.array(self.y_log_1),

            v=np.array(self.v_log_1),
            w=np.array(self.w_log_1),

            final_time=T_final_1,
            dt=self.dt
        )
            
        np.savez(
            "rover_2_log.npz",

            t=np.array(self.time_log_2),

            x=np.array(self.x_log_2),
            y=np.array(self.y_log_2),

            v=np.array(self.v_log_2),
            w=np.array(self.w_log_2),

            final_time=T_final_2,
            dt=self.dt
        )

        np.savez(
            "rover_3_log.npz",

            t=np.array(self.time_log_3),

            x=np.array(self.x_log_3),
            y=np.array(self.y_log_3),

            v=np.array(self.v_log_3),
            w=np.array(self.w_log_3),

            final_time=T_final_3,
            dt=self.dt
        )
            
        self.get_logger().info(
            "Saved rover_1_log.npz"
        )
            
        self.get_logger().info(
            "Saved rover_2_log.npz"
        )
            
        self.get_logger().info(
            "Saved rover_3_log.npz"
        )
    
    
    def plot_cbf_logs(self, rover_id):

        data = np.loadtxt(
            f"rover_{rover_id}_cbf_logs.csv",
            delimiter=",",
            skiprows=1
        )

        t = data[:,0]

    # =====================================================
    # Distance
    # =====================================================

        plt.figure(figsize=(10,5))

        plt.plot(t, data[:,5], linewidth=2,
                 label="Neighbour 1")

        plt.plot(t, data[:,6], linewidth=2,
                 label="Neighbour 2")

        plt.axhline(
            self.D_SAFE,
            color='r',
            linestyle='--',
            linewidth=2,
            label="Safety Distance"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title(f"Rover {rover_id} Inter-Rover Distances")
        plt.grid(True)
        plt.legend()

        plt.savefig(
            f"rover{rover_id}_distance.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

    # =====================================================
    # Barrier Function
    # =====================================================

        plt.figure(figsize=(10,5))

        plt.plot(t, data[:,7], linewidth=2,
                 label="Neighbour 1")

        plt.plot(t, data[:,8], linewidth=2,
                 label="Neighbour 2")

        plt.axhline(
            0,
            color='r',
            linestyle='--',
            linewidth=2,
            label="h = 0"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Barrier Function")
        plt.title(f"Rover {rover_id} Barrier Functions")
        plt.grid(True)
        plt.legend()

        plt.savefig(
            f"rover{rover_id}_barrier.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

    # =====================================================
    # Nominal vs QP Control
    # =====================================================

        plt.figure(figsize=(10,5))

        u_nom = np.sqrt(data[:,1]**2 + data[:,2]**2)
        u_qp  = np.sqrt(data[:,3]**2 + data[:,4]**2)

        plt.plot(
            t,
            u_nom,
            '--',
            linewidth=2,
            label="Nominal"
        )

        plt.plot(
            t,
            u_qp,
            linewidth=2,
            label="QP"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Control Magnitude")
        plt.title(f"Rover {rover_id} Nominal vs Safe Control")
        plt.grid(True)
        plt.legend()

        plt.savefig(
            f"rover{rover_id}_control.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

    # =====================================================
    # Safety Link
    # =====================================================

        plt.figure(figsize=(10,5))

        plt.step(
            t,
            data[:,9],
            where='post',
            linewidth=2,
            label="Neighbour 1"
        )

        plt.step(
            t,
            data[:,10],
            where='post',
            linewidth=2,
            label="Neighbour 2"
       )

        plt.yticks([0,1],["OFF","ON"])

        plt.xlabel("Time (s)")
        plt.ylabel("Safety Link")
        plt.title(f"Rover {rover_id} Safety Link")
        plt.grid(True)
        plt.legend()

        plt.savefig(
            f"rover{rover_id}_safety_link.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
    
    def plot_trajectories(self):

        r1 = np.load("rover_1_log.npz")
        r2 = np.load("rover_2_log.npz")
        r3 = np.load("rover_3_log.npz")
        
        T1 = float(r1["final_time"])
        T2 = float(r2["final_time"])
        T3 = float(r3["final_time"])
        
        plt.figure(figsize=(10,8))
        
        alpha = np.array([1.0, 1.0, 1.0])
        I2 = np.eye(2)
        zero2 = np.zeros((2,2))

# B matrices (6x2)
        B1 = np.vstack([I2, -I2, -I2])
        B2 = np.vstack([zero2, I2, zero2])
        B3 = np.vstack([zero2, zero2, I2])
            
        def control_law(t, x):
            x1 = x[0:2]
            x2 = x[2:4]
            x3 = x[4:6]

            u1 = -2 * x1
            u2 = -(x1 + 2*x2)
            u3 = -(x1 + 2*x3)

            return u1, u2, u3

# =============================
# Dynamics
# =============================
        def dynamics(t, x):
            u1, u2, u3 = control_law(t, x)
            dx = B1 @ u1 + B2 @ u2 + B3 @ u3
            return dx

    # SAME INITIAL CONDITIONS AS GAZEBO

        p1_0 = np.array([-1.0,1.0])
        p2_0 = np.array([-5.0, 5.0])
        p3_0 = np.array([-5.0, 5.0])

        p1_star = np.array([5.0, 5.0])

        r12 = np.array([-2.0, 0.0])
        r13 = np.array([0.0, -2.0])

        x1_0 = p1_0 - p1_star
        x2_0 = (p2_0 - p1_0) - r12
        x3_0 = (p3_0 - p1_0) - r13
        
        X0 = np.concatenate([
            x1_0,
            x2_0,
            x3_0
        ])
        Tmax = max(T1, T2, T3)
        
        t_eval = np.arange(0, Tmax, self.dt)

        sol = solve_ivp(
            dynamics,
            [0, Tmax],
            X0,
            t_eval=t_eval,method='RK45'
        )

        # indices corresponding to each simulation duration

        x1_theory = sol.y[0:2, :]
        x2_theory = sol.y[2:4, :]
        x3_theory = sol.y[4:6, :]
        
        p1_theory = x1_theory + p1_star.reshape(-1,1)
        p2_theory = x2_theory + p1_theory + r12.reshape(-1,1)
        p3_theory = x3_theory + p1_theory + r13.reshape(-1,1)
        
        p1_gazebo = np.column_stack((r1["x"], r1["y"]))
        p2_gazebo = np.column_stack((r2["x"], r2["y"]))
        p3_gazebo = np.column_stack((r3["x"], r3["y"]))

    # =============================
    # OVERLAY PLOT
    # =============================
        
        plt.plot(
            p1_theory[0,:],
            p1_theory[1,:],
            '--',
            linewidth=2,
            label='Rover 1 Theory'
        )

        plt.plot(
            p1_gazebo[:,0],
            p1_gazebo[:,1],
            linewidth=2,
            label='Rover 1 Gazebo'
        )

        plt.plot(
            p2_theory[0,:],
            p2_theory[1,:],
            '--',
            linewidth=2,
            label='Rover 2 Theory'
        )

        plt.plot(
            p2_gazebo[:,0],
            p2_gazebo[:,1],
            linewidth=2,
            label='Rover 2 Gazebo'
        ) 

        plt.plot(
            p3_theory[0,:],
            p3_theory[1,:],
            '--',
            linewidth=2,
            label='Rover 3 Theory'
        )

        plt.plot(
            p3_gazebo[:,0],
            p3_gazebo[:,1],
            linewidth=2,
            label='Rover 3 Gazebo'
        )

# ----------------------
# Start markers
# ----------------------
        plt.scatter(
            p1_gazebo[0,0],
            p1_gazebo[0,1],
            s=100,
            marker='o'
        )

        plt.scatter(
            p2_gazebo[0,0],
            p2_gazebo[0,1],
            s=100,
            marker='o'
        )

        plt.scatter(
            p3_gazebo[0,0],
            p3_gazebo[0,1],
            s=100,
            marker='o'
        )

# ----------------------
# End markers
# ----------------------
        plt.scatter(
            p1_gazebo[-1,0],
            p1_gazebo[-1,1],
            s=100,
            marker='x'
        )

        plt.scatter(
            p2_gazebo[-1,0],
            p2_gazebo[-1,1],
            s=100,
            marker='x'
        )

        plt.scatter(
            p3_gazebo[-1,0],
            p3_gazebo[-1,1],
            s=100,
            marker='x'
        )
     

        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Multi-Rover Trajectories")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(
            "trajectory_plot.png",
            dpi=300,
            bbox_inches="tight"
        )
        
        plt.show()
        
        plt.figure(figsize=(10,5))

        for i in [1,2,3]:
            data = np.load(f"rover_{i}_profile.npz")

            plt.plot(data["t"], data["v_raw"], '--', linewidth=1.5,
                    label=f"R{i} Raw")
            plt.plot(data["t"], data["v_sat"], linewidth=2,
                    label=f"R{i} Saturated")

        plt.xlabel("Time (s)")
        plt.ylabel("Linear Velocity (m/s)")
        plt.title("Linear Velocity: Raw vs Saturated")
        plt.grid(True)
        plt.legend()
        
        plt.savefig("linear_velocity_profiles.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        plt.figure(figsize=(10,5))

        for i in [1,2,3]:
            data = np.load(f"rover_{i}_profile.npz")

            plt.plot(data["t"], data["w_raw"], '--', linewidth=1.5,
                     label=f"R{i} Raw")
            plt.plot(data["t"], data["w_sat"], linewidth=2,
                     label=f"R{i} Saturated")

        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (rad/s)")
        plt.title("Angular Velocity: Raw vs Saturated")
        plt.grid(True)
        plt.legend()
        
        plt.savefig("angular_velocity_profiles.png", dpi=300, bbox_inches="tight")
        plt.show()
        self.plot_cbf_logs(1)
        self.plot_cbf_logs(2)
        self.plot_cbf_logs(3)

    def destroy_node(self):

        self.file1.close()
        self.file2.close()
        self.file3.close()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = TrajectoryLogger()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
