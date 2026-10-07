#!/usr/bin/env python3

"""

cbf_theory_comparison.py



ROS2 node that:

  1. Logs all three rovers' odom (x,y,v,w) during the run, exactly like TrajectoryLogger.

  2. Waits until all three rovers report goal_reached.

  3. Runs theoretical CBF-QP simulation with DDMR realization matching the actual controller.

  4. Produces:

       - trajectory_comparison.png   (Gazebo vs CBF-QP DDMR theory - single plot)

       - linear_velocity.png         (raw vs saturated, Gazebo)

       - angular_velocity.png        (raw vs saturated, Gazebo)

"""



import math

import csv

import numpy as np

import matplotlib.pyplot as plt



import rclpy

from rclpy.node import Node

from nav_msgs.msg import Odometry

from std_msgs.msg import Bool



def quaternion_to_yaw(x, y, z, w):

    siny_cosp = 2.0 * (w * z + x * y)

    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(siny_cosp, cosy_cosp)



# ============================================================

# SIMULATION PARAMETERS  (match cbf_controller_1 exactly)

# ============================================================

DT              = 0.02

T_SIM           = 60.0        # hard ceiling; early-stop at goal



ROBOT_RADIUS    = 0.25

SAFETY_MARGIN   = 0.20

D_SAFE          = ROBOT_RADIUS + ROBOT_RADIUS + SAFETY_MARGIN   # 0.70 m



R_VIS           = 1.5

R_TRIGGER       = 1.0

R_RELEASE       = 1.3

GAMMA_CBF       = 5.0

PSI_ACTIVATE_TOL = 1e-9

PSI_RELEASE     = 0.25

MIN_ACTIVE_TIME       = 0.10

RELEASE_CONFIRM_TIME  = 0.10



MIN_ACTIVE_STEPS      = max(1, math.ceil(MIN_ACTIVE_TIME      / DT))

RELEASE_CONFIRM_STEPS = max(1, math.ceil(RELEASE_CONFIRM_TIME / DT))



GOAL_TOLERANCE  = 0.3



# ============================================================

# MISSION GEOMETRY  (match Gazebo spawn / trajectory_logger)

# ============================================================

p1_star = np.array([ 5.0,  5.0])

r12     = np.array([-2.0,  0.0])

r13     = np.array([ 0.0, -2.0])



target_2 = p1_star + r12

target_3 = p1_star + r13

TARGETS = [p1_star, target_2, target_3]



# ============================================================

# DDMR PARAMETERS (matching cbf_controller_1)

# ============================================================

L_DDMR          = 0.20          # Changed from 0.15 to match controller

V_MAX_DDMR      = 0.30

W_MAX_DDMR      = 0.80

OMEGA_REF       = 5.0           # rad/s where slowdown begins

SPEED_MIN       = 0.05

SPEED_MAX       = 0.3

INITIAL_HEADING_DEG = [45.0, 90.0, -11.31]



PAIRS       = [(0, 1), (0, 2), (1, 2)]

PAIR_NAMES  = ['1-2', '1-3', '2-3']

COLORS      = ['tab:blue', 'tab:orange', 'tab:green']



# ============================================================

# NOMINAL LQ CONTROL

# ============================================================

def nominal_control(p):

    x1 = p[:, 0] - p1_star

    x2 = (p[:, 1] - p[:, 0]) - r12

    x3 = (p[:, 2] - p[:, 0]) - r13

    u1 = -2.0 * x1

    u2 = -(x1 + 2.0 * x2)

    u3 = -(x1 + 2.0 * x3)

    return np.column_stack([u1, u2, u3])



# ============================================================

# CBF-QP SOLVER

# ============================================================

def solve_projection_qp_2d(u_nom, A_rows, b_vals):

    if not A_rows:

        return u_nom.copy()

    TOL = 1e-10



    def feasible(u):

        for a, b in zip(A_rows, b_vals):

            if a @ u > b + TOL:

                return False

        return True



    candidates = []

    if feasible(u_nom):

        candidates.append(u_nom.copy())



    for a, b in zip(A_rows, b_vals):

        denom = a @ a

        if denom <= TOL:

            continue

        cand = u_nom - ((a @ u_nom - b) / denom) * a

        if feasible(cand):

            candidates.append(cand)



    n = len(A_rows)

    for ell in range(n):

        for m in range(ell + 1, n):

            M = np.array([A_rows[ell], A_rows[m]])

            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]

            if abs(det) <= TOL:

                continue

            cand = np.linalg.solve(M, np.array([b_vals[ell], b_vals[m]]))

            if feasible(cand):

                candidates.append(cand)



    if not candidates:

        zero = np.zeros(2)

        return zero if feasible(zero) else u_nom.copy()



    return min(candidates, key=lambda c: float(np.sum((c - u_nom) ** 2)))



def apply_cbf_qp_distributed(p, u_nom, pair_active):

    u_safe = np.zeros((2, 3))

    for i in range(3):

        A_rows, b_vals = [], []

        for q, (pi, pj) in enumerate(PAIRS):

            if not pair_active[q]:

                continue

            j = pj if i == pi else (pi if i == pj else None)

            if j is None:

                continue

            rel  = p[:, i] - p[:, j]

            dist = np.linalg.norm(rel)

            h_ij = dist ** 2 - D_SAFE ** 2

            A_rows.append(-2.0 * rel)

            b_vals.append(0.5 * GAMMA_CBF * h_ij)

        u_safe[:, i] = solve_projection_qp_2d(u_nom[:, i], A_rows, b_vals)

    return u_safe



# ============================================================

# THEORETICAL CBF-QP SIMULATION WITH DDMR REALIZATION

# ============================================================

def run_cbf_simulation_with_ddmr(p0):

    """

    Runs CBF-QP simulation with DDMR realization matching the actual controller.

    p0 : (2, 3) — initial positions taken from first Gazebo odom samples.

    """

    # Initialize DDMR states (x, y, theta) for each rover

    state = np.zeros((3, 3))

    state[:, 0] = p0[0, :]  # x positions

    state[:, 1] = p0[1, :]  # y positions

    state[:, 2] = np.radians(INITIAL_HEADING_DEG)  # initial headings



    pair_active = [False, False, False]

    hold_counter = [0, 0, 0]

    rel_counter = [0, 0, 0]



    t_log = []

    state_log = []  # DDMR states [x, y, theta] for each rover

    u_nom_log = []

    u_safe_log = []  # Single-integrator safe velocities

    u_ddmr_log = []  # DDMR [v, omega] commands



    t = 0.0

    while t < T_SIM:

        # Get current positions

        p_current = state[:, :2].T  # (2, 3)



        # Nominal LQ control

        u_nom = nominal_control(p_current)



        # Check distances and CBF conditions

        dist = np.zeros(3)

        h = np.zeros(3)

        psi = np.zeros(3)

        vis = np.zeros(3, dtype=bool)



        for q, (i, j) in enumerate(PAIRS):

            rel = p_current[:, i] - p_current[:, j]

            dist[q] = np.linalg.norm(rel)

            h[q] = dist[q] ** 2 - D_SAFE ** 2

            psi[q] = (2.0 * rel @ u_nom[:, i] + 0.5 * GAMMA_CBF * h[q])

            vis[q] = dist[q] <= R_VIS



        # Pair activation logic

        for q in range(3):

            if not pair_active[q]:

                rel_counter[q] = 0

                if (vis[q] and dist[q] <= R_TRIGGER and psi[q] < -PSI_ACTIVATE_TOL):

                    pair_active[q] = True

                    hold_counter[q] = MIN_ACTIVE_STEPS

                    rel_counter[q] = 0

            elif hold_counter[q] > 0:

                hold_counter[q] -= 1

                rel_counter[q] = 0

            else:

                if dist[q] >= R_RELEASE and psi[q] >= PSI_RELEASE:

                    rel_counter[q] += 1

                    if rel_counter[q] >= RELEASE_CONFIRM_STEPS:

                        pair_active[q] = False

                        hold_counter[q] = 0

                        rel_counter[q] = 0

                else:

                    rel_counter[q] = 0



        # Get CBF-safe reference velocities (single-integrator)

        u_safe = apply_cbf_qp_distributed(p_current, u_nom, pair_active)



        # === DDMR REALIZATION (matching controller exactly) ===

        u_ddmr = np.zeros((2, 3))  # [v, omega] for each rover

        for i in range(3):

            # Feedback linearization

            cos_theta = math.cos(state[i, 2])

            sin_theta = math.sin(state[i, 2])



            speed_raw = cos_theta * u_safe[0, i] + sin_theta * u_safe[1, i]

            speed_raw = max(0.0, speed_raw)



            omega_raw = (-sin_theta * u_safe[0, i] + cos_theta * u_safe[1, i]) / L_DDMR

            omega = np.clip(omega_raw, -W_MAX_DDMR, W_MAX_DDMR)



            # Exponential velocity scaling (matching controller)

            scale = np.exp(-abs(omega_raw) / OMEGA_REF)

            speed = SPEED_MIN + (SPEED_MAX - SPEED_MIN) * scale

            speed = min(speed, speed_raw)

            speed = np.clip(speed, -V_MAX_DDMR, V_MAX_DDMR)



            u_ddmr[0, i] = speed

            u_ddmr[1, i] = omega



        # Logging

        t_log.append(t)

        state_log.append(state.copy())

        u_nom_log.append(u_nom.copy())

        u_safe_log.append(u_safe.copy())

        u_ddmr_log.append(u_ddmr.copy())



        # Update DDMR state using exact propagation

        for i in range(3):

            v = u_ddmr[0, i]

            omega = u_ddmr[1, i]



            if abs(omega) > 1e-10:

                state[i, 0] += (v / omega) * (math.sin(state[i, 2] + omega * DT) - math.sin(state[i, 2]))

                state[i, 1] -= (v / omega) * (math.cos(state[i, 2] + omega * DT) - math.cos(state[i, 2]))

            else:

                state[i, 0] += DT * v * math.cos(state[i, 2])

                state[i, 1] += DT * v * math.sin(state[i, 2])

            state[i, 2] += omega * DT



        t += DT



        # Check if all rovers reached goals

        all_reached = True

        for i in range(3):

            if np.linalg.norm(state[i, :2] - TARGETS[i]) > GOAL_TOLERANCE:

                all_reached = False

                break

        if all_reached:

            break



    return {

        't': np.array(t_log),

        'state': np.array(state_log),      # (N, 3, 3) - [step, rover, [x, y, theta]]

        'u_nom': np.array(u_nom_log),      # (N, 2, 3)

        'u_safe': np.array(u_safe_log),    # (N, 2, 3) - single-integrator safe velocities

        'u_ddmr': np.array(u_ddmr_log),    # (N, 2, 3) - [step, rover, [v, omega]]

    }



# ============================================================

# PLOTS

# ============================================================

def plot_trajectory_comparison(theory_result, gz_x, gz_y):

    """Single plot comparing Gazebo vs CBF-QP DDMR theory"""

    

    fig, ax = plt.subplots(figsize=(12, 10))

    

    # Plot theoretical trajectories (DDMR realization)

    for i in range(3):

        xy = theory_result['state'][:, i, :2]  # (N, 2)

        ax.plot(xy[:, 0], xy[:, 1], 

                color=COLORS[i], linestyle='--', linewidth=2.5,

                label=f'Rover {i+1} Theory (DDMR)')

        ax.scatter(xy[0, 0], xy[0, 1], s=100, color=COLORS[i], marker='s',

                  edgecolors='black', linewidth=1.5, zorder=5)

        ax.scatter(xy[-1, 0], xy[-1, 1], s=100, color=COLORS[i], marker='o',

                  edgecolors='black', linewidth=1.5, zorder=5)

    

    # Plot Gazebo trajectories

    for i in range(3):

        if gz_x[i] is not None and len(gz_x[i]) > 0:

            ax.plot(gz_x[i], gz_y[i], 

                    color=COLORS[i], linewidth=2,

                    label=f'Rover {i+1} Gazebo')

            ax.scatter(gz_x[i][0], gz_y[i][0], s=100, color=COLORS[i], 

                      marker='s', alpha=0.7, edgecolors='black', linewidth=1.5, zorder=4)

            ax.scatter(gz_x[i][-1], gz_y[i][-1], s=100, color=COLORS[i],

                      marker='o', alpha=0.7, edgecolors='black', linewidth=1.5, zorder=4)

    

    # Plot targets

    for i, target in enumerate(TARGETS):

        ax.scatter(*target, s=200, color=COLORS[i], marker='*', 

                  edgecolors='black', linewidth=1.5, zorder=6,

                  label=f'Target {i+1}')

    

    ax.set_xlabel('X [m]', fontsize=12)

    ax.set_ylabel('Y [m]', fontsize=12)

    ax.set_title('Gazebo vs CBF-QP DDMR Theory', fontsize=14, fontweight='bold')

    ax.set_aspect('equal')

    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=10, loc='best')

    

    # Add a text box with simulation info

    textstr = f'DT = {DT*1000:.1f} ms\n'

    textstr += f'D_safe = {D_SAFE:.2f} m\n'

    textstr += f'R_vis = {R_VIS:.1f} m\n'

    textstr += f'γ = {GAMMA_CBF:.1f}\n'

    textstr += f'L = {L_DDMR:.2f} m'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,

            verticalalignment='top', bbox=props)

    

    plt.tight_layout()

    plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')

    plt.show()

    plt.close()



def plot_velocity_profiles(v_raw, v_sat, w_raw, w_sat, t_logs):

    """Plot velocity profiles (raw vs saturated)"""

    

    # Linear velocity

    plt.figure(figsize=(12, 5))

    for i in range(3):

        if t_logs[i] is None or len(t_logs[i]) == 0:

            continue

        t = np.array(t_logs[i])

        plt.plot(t, v_raw[i], '--', linewidth=1.5,

                 color=COLORS[i], label=f'R{i+1} raw', alpha=0.7)

        plt.plot(t, v_sat[i], '-', linewidth=2.5,

                 color=COLORS[i], label=f'R{i+1} saturated')

    plt.xlabel('Time [s]', fontsize=12)

    plt.ylabel('Linear velocity [m/s]', fontsize=12)

    plt.title('Linear velocity: raw vs saturated (Gazebo)', fontsize=14)

    plt.grid(True, alpha=0.3)

    plt.legend(fontsize=10)

    plt.tight_layout()

    plt.savefig('linear_velocity.png', dpi=300, bbox_inches='tight')

    plt.show()

    plt.close()



    # Angular velocity

    plt.figure(figsize=(12, 5))

    for i in range(3):

        if t_logs[i] is None or len(t_logs[i]) == 0:

            continue

        t = np.array(t_logs[i])

        plt.plot(t, w_raw[i], '--', linewidth=1.5,

                 color=COLORS[i], label=f'R{i+1} raw', alpha=0.7)

        plt.plot(t, w_sat[i], '-', linewidth=2.5,

                 color=COLORS[i], label=f'R{i+1} saturated')

    plt.xlabel('Time [s]', fontsize=12)

    plt.ylabel('Angular velocity [rad/s]', fontsize=12)

    plt.title('Angular velocity: raw vs saturated (Gazebo)', fontsize=14)

    plt.grid(True, alpha=0.3)

    plt.legend(fontsize=10)

    plt.tight_layout()

    plt.savefig('angular_velocity.png', dpi=300, bbox_inches='tight')

    plt.show()

    plt.close()



# ============================================================

# ROS2 NODE

# ============================================================

class CbfTheoryComparison(Node):



    def __init__(self):

        super().__init__('cbf_theory_comparison')



        # goal flags

        self.r1_done = False

        self.r2_done = False

        self.r3_done = False

        self.finished = False



        self.dt = 0.01

        self.V_THRESHOLD = 0.02



        # per-rover logs

        self.started    = [False, False, False]

        self.start_idx  = [None,  None,  None]

        self.sample_cnt = [0,     0,     0]



        self.t_log  = [[], [], []]

        self.x_log  = [[], [], []]

        self.y_log  = [[], [], []]

        self.v_raw  = [[], [], []]

        self.v_sat  = [[], [], []]

        self.w_raw  = [[], [], []]

        self.w_sat  = [[], [], []]



        # first odom positions for CBF sim initial conditions

        self.first_pos  = [None, None, None]



        # CSV writers

        self.csv_files   = []

        self.csv_writers = []

        for i in range(1, 4):

            f = open(f'rover_{i}.csv', 'w', newline='')

            w = csv.writer(f)

            w.writerow(['time', 'x', 'y'])

            self.csv_files.append(f)

            self.csv_writers.append(w)



        # subscribers

        for i, rid in enumerate([1, 2, 3]):

            self.create_subscription(

                Odometry,

                f'/rover_{rid}/odom',

                lambda msg, idx=i: self._odom_cb(msg, idx),

                10

            )

            self.create_subscription(

                Bool,

                f'/rover_{rid}/goal_reached',

                lambda msg, idx=i: self._goal_cb(msg, idx),

                10

            )



        self.get_logger().info('CbfTheoryComparison node started -- waiting for rovers.')



    def _odom_cb(self, msg, idx):

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        x = msg.pose.pose.position.x

        y = msg.pose.pose.position.y

        v = msg.twist.twist.linear.x

        w = msg.twist.twist.angular.z



        # record first position for CBF sim ICs

        if self.first_pos[idx] is None:

            self.first_pos[idx] = np.array([x, y])



        self.csv_writers[idx].writerow([t, x, y])



        if not self.started[idx] and abs(v) > self.V_THRESHOLD:

            self.started[idx]   = True

            self.start_idx[idx] = self.sample_cnt[idx]



        if self.started[idx]:

            t_rel = (self.sample_cnt[idx] - self.start_idx[idx]) * self.dt

            self.t_log[idx].append(t_rel)

            self.x_log[idx].append(x)

            self.y_log[idx].append(y)

            self.v_raw[idx].append(v)

            self.v_sat[idx].append(v)   # odom reports actual velocity

            self.w_raw[idx].append(w)

            self.w_sat[idx].append(w)



        self.sample_cnt[idx] += 1



    def _goal_cb(self, msg, idx):

        if msg.data:

            if idx == 0:

                self.r1_done = True

            elif idx == 1:

                self.r2_done = True

            elif idx == 2:

                self.r3_done = True

        self._check_finished()



    def _check_finished(self):

        if self.finished:

            return

        if not (self.r1_done and self.r2_done and self.r3_done):

            return



        self.finished = True

        self.get_logger().info('All rovers reached goals -- running analysis.')

        self._run_analysis()

        self.destroy_node()

        rclpy.shutdown()



    def _run_analysis(self):

        # close CSV files

        for f in self.csv_files:

            f.close()



        # ------ build initial conditions from first odom readings ------

        p0 = np.zeros((2, 3))

        for i in range(3):

            if self.first_pos[i] is not None:

                p0[:, i] = self.first_pos[i]

            else:

                # fallback to hardcoded spawn positions

                defaults = [

                    np.array([-5.0, -5.0]),

                    np.array([-3.0, -7.0]),

                    np.array([-7.0,  -3.0])

                ]

                p0[:, i] = defaults[i]

                self.get_logger().warn(

                    f'No odom received for rover {i+1}, '

                    f'using default spawn position.'

                )



        # ------ theoretical CBF-QP with DDMR realization ------

        self.get_logger().info('Running theoretical CBF-QP with DDMR realization...')

        theory_result = run_cbf_simulation_with_ddmr(p0)

        self.get_logger().info(

            f'Simulation done: {len(theory_result["t"])} steps, '

            f'T={theory_result["t"][-1]:.2f} s'

        )



        # ------ package Gazebo trajectory data ------

        gz_x = [

            np.array(self.x_log[i]) if self.x_log[i] else None

            for i in range(3)

        ]

        gz_y = [

            np.array(self.y_log[i]) if self.y_log[i] else None

            for i in range(3)

        ]



        # ------ plots ------

        self.get_logger().info('Generating plots...')

        plot_trajectory_comparison(theory_result, gz_x, gz_y)

        plot_velocity_profiles(

            self.v_raw, self.v_sat,

            self.w_raw, self.w_sat,

            self.t_log

        )

        self.get_logger().info(

            'Plots saved: trajectory_comparison.png, '

            'linear_velocity.png, angular_velocity.png'

        )



    def destroy_node(self):

        for f in self.csv_files:

            try:

                f.close()

            except Exception:

                pass

        super().destroy_node()



# ============================================================

# MAIN

# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = CbfTheoryComparison()

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
