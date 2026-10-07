#!/usr/bin/env python3
"""
cbf_theory_comparison.py  --  multi-disc footprint-aware CBF-QP

ROS2 node that:
  1. Logs all three rovers odom (x,y,v,w) during the run.
  2. Waits until all three rovers report goal_reached.
  3. Runs the theoretical CBF-QP simulation with a MULTI-DISC footprint
     model and DDMR realisation matching the actual controller.
  4. Produces:
       trajectory_comparison.png
       linear_velocity.png
       angular_velocity.png

CHANGE vs previous version (single-disc):
  Each rover is now represented by N_DISCS = 4 overlapping discs placed
  at body-frame offsets [+0.09, +0.03, -0.03, -0.09] m along the
  longitudinal axis with disc radius 0.05 m.
  During every control cycle the disc centres are transformed to the world
  frame using the current DDMR pose (x, y, theta) and a CBF constraint is
  generated for EVERY disc pair between neighbouring rovers.  All 16
  constraints per rover pair are added to the QP; the solver returns the
  minimum-norm correction that satisfies all of them simultaneously.
  The DDMR realization, velocity saturation, visibility / trigger /
  release logic, and tracker architecture are unchanged.
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
# SIMULATION PARAMETERS  (match cbf_controller exactly)
# ============================================================
DT    = 0.02
T_SIM = 60.0

# Center-to-center safe distance -- still used for visibility /
# trigger / release checks (unchanged from original).
ROBOT_RADIUS  = 0.25
SAFETY_MARGIN = 0.20
D_SAFE        = ROBOT_RADIUS + ROBOT_RADIUS + SAFETY_MARGIN   # 0.70 m

R_VIS            = 2.5
R_TRIGGER        = 2.0
R_RELEASE        = 2.2
GAMMA_CBF        = 5.0
PSI_ACTIVATE_TOL = 1e-9
PSI_RELEASE      = 0.25
MIN_ACTIVE_TIME       = 0.10
RELEASE_CONFIRM_TIME  = 0.10

MIN_ACTIVE_STEPS      = max(1, math.ceil(MIN_ACTIVE_TIME      / DT))
RELEASE_CONFIRM_STEPS = max(1, math.ceil(RELEASE_CONFIRM_TIME / DT))

GOAL_TOLERANCE = 0.3

# ============================================================
# MULTI-DISC FOOTPRINT PARAMETERS  (NEW)
#
# Four discs placed along the rover longitudinal axis (body-frame x).
# The safety distance for a disc pair is 2 * DISC_RADIUS = 0.10 m.
# The disc positions span ±0.09 m which covers the rover chassis.
# ============================================================
DISC_OFFSETS = np.array([
    [ 0.09, 0.0],
    [ 0.03, 0.0],
    [-0.03, 0.0],
    [-0.09, 0.0],
])                           # (4, 2) in body frame
DISC_RADIUS = 0.05           # m
DISC_D_SAFE = 2.0 * DISC_RADIUS   # 0.10 m  -- per disc-pair safe distance
N_DISCS     = len(DISC_OFFSETS)


def get_disc_centers(px, py, theta):
    """
    Transform all 4 disc centres from body frame to world frame.
    Returns (N_DISCS, 2) array.
    """
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    # (2, 4) -> transpose -> (4, 2) and add rover position
    return np.array([px, py]) + (R @ DISC_OFFSETS.T).T


# ============================================================
# MISSION GEOMETRY
# ============================================================
p1_star = np.array([ 5.0,  5.0])
r12     = np.array([-2.0,  0.0])
r13     = np.array([ 0.0, -2.0])

target_2 = p1_star + r12
target_3 = p1_star + r13
TARGETS  = [p1_star, target_2, target_3]

# ============================================================
# DDMR PARAMETERS
# ============================================================
L_DDMR     = 0.20
V_MAX_DDMR = 0.30
W_MAX_DDMR = 0.80
OMEGA_REF  = 5.0
SPEED_MIN  = 0.05
SPEED_MAX  = 0.30
INITIAL_HEADING_DEG = [45.0, 90.0, -11.31]

PAIRS      = [(0, 1), (0, 2), (1, 2)]
PAIR_NAMES = ['1-2', '1-3', '2-3']
COLORS     = ['tab:blue', 'tab:orange', 'tab:green']


# ============================================================
# NOMINAL LQ CONTROL
# ============================================================
def nominal_control(p):
    """p: (2, 3). Returns u: (2, 3)."""
    x1 = p[:, 0] - p1_star
    x2 = (p[:, 1] - p[:, 0]) - r12
    x3 = (p[:, 2] - p[:, 0]) - r13
    return np.column_stack([-2.0 * x1,
                             -(x1 + 2.0 * x2),
                             -(x1 + 2.0 * x3)])


# ============================================================
# CBF-QP SOLVER  (unchanged)
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
            M   = np.array([A_rows[ell], A_rows[m]])
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


# ============================================================
# MULTI-DISC CBF-QP  (NEW -- replaces apply_cbf_qp_distributed)
#
# For each active rover pair (i, j):
#   For each disc pair (k from rover i, l from rover j):
#     h_kl   = ||c_ik - c_jl||^2 - DISC_D_SAFE^2
#     Constraint on rover i's control u_i:
#       -2*(c_ik - c_jl)' u_i  <=  (gamma/2) * h_kl
# ============================================================
def apply_cbf_qp_multidisc(p, thetas, u_nom, pair_active):
    """
    p       : (2, 3) rover centre positions
    thetas  : (3,)   current DDMR headings
    u_nom   : (2, 3) single-integrator nominal velocities
    Returns u_safe : (2, 3)
    """
    u_safe = np.zeros((2, 3))

    for i in range(3):
        A_rows, b_vals = [], []

        # Pre-compute rover i disc centres (used for every pair it appears in)
        centers_i = get_disc_centers(p[0, i], p[1, i], thetas[i])

        for q, (pi, pj) in enumerate(PAIRS):
            if not pair_active[q]:
                continue
            j = pj if i == pi else (pi if i == pj else None)
            if j is None:
                continue

            centers_j = get_disc_centers(p[0, j], p[1, j], thetas[j])

            # Build one constraint per disc pair (k from i, l from j)
            for k in range(N_DISCS):
                for l in range(N_DISCS):
                    rel    = centers_i[k] - centers_j[l]
                    dist_kl = np.linalg.norm(rel)
                    h_kl   = dist_kl ** 2 - DISC_D_SAFE ** 2
                    # -2*(c_ik - c_jl)' u_i <= (gamma/2)*h_kl
                    A_rows.append(-2.0 * rel)
                    b_vals.append(0.5 * GAMMA_CBF * h_kl)

        u_safe[:, i] = solve_projection_qp_2d(u_nom[:, i], A_rows, b_vals)

    return u_safe


# ============================================================
# MULTI-DISC PSI  (NEW -- for hysteresis activation check)
#
# Returns the minimum (most critical) psi across all disc pairs
# between rovers i and j.  Activation fires when this min < 0.
# ============================================================
def min_disc_psi(p, thetas, u_nom, i, j):
    centers_i = get_disc_centers(p[0, i], p[1, i], thetas[i])
    centers_j = get_disc_centers(p[0, j], p[1, j], thetas[j])
    min_psi = float('inf')
    for k in range(N_DISCS):
        for l in range(N_DISCS):
            rel    = centers_i[k] - centers_j[l]
            dist_kl = np.linalg.norm(rel)
            h_kl   = dist_kl ** 2 - DISC_D_SAFE ** 2
            psi_kl = 2.0 * rel @ u_nom[:, i] + 0.5 * GAMMA_CBF * h_kl
            if psi_kl < min_psi:
                min_psi = psi_kl
    return min_psi


# ============================================================
# THEORETICAL CBF-QP + DDMR SIMULATION
# ============================================================
def run_cbf_simulation_with_ddmr(p0):
    """
    p0 : (2, 3)  initial rover positions (from first Gazebo odom samples).
    """
    # DDMR state:  state[i] = [x, y, theta]
    state       = np.zeros((3, 3))
    state[:, 0] = p0[0, :]
    state[:, 1] = p0[1, :]
    state[:, 2] = np.radians(INITIAL_HEADING_DEG)

    pair_active  = [False, False, False]
    hold_counter = [0, 0, 0]
    rel_counter  = [0, 0, 0]

    t_log      = []
    state_log  = []
    u_nom_log  = []
    u_safe_log = []
    u_ddmr_log = []

    t = 0.0
    while t < T_SIM:
        p_current = state[:, :2].T    # (2, 3)
        thetas    = state[:, 2]       # (3,)

        u_nom = nominal_control(p_current)

        # ----------------------------------------------------------
        # Hysteresis state machine
        # Visibility / trigger / release still use rover CENTER-to-CENTER
        # distance (unchanged).
        # Psi check uses MINIMUM disc-pair psi (more conservative).
        # ----------------------------------------------------------
        dist = np.zeros(3)
        vis  = np.zeros(3, dtype=bool)
        psi  = np.zeros(3)

        for q, (i, j) in enumerate(PAIRS):
            rel     = p_current[:, i] - p_current[:, j]
            dist[q] = np.linalg.norm(rel)
            vis[q]  = dist[q] <= R_VIS
            # Use min disc-pair psi from rover i's perspective
            psi[q]  = min_disc_psi(p_current, thetas, u_nom, i, j)

        for q in range(3):
            if not pair_active[q]:
                rel_counter[q] = 0
                if (vis[q]
                        and dist[q] <= R_TRIGGER
                        and psi[q] < -PSI_ACTIVATE_TOL):
                    pair_active[q]  = True
                    hold_counter[q] = MIN_ACTIVE_STEPS
                    rel_counter[q]  = 0
            elif hold_counter[q] > 0:
                hold_counter[q] -= 1
                rel_counter[q]   = 0
            else:
                if dist[q] >= R_RELEASE and psi[q] >= PSI_RELEASE:
                    rel_counter[q] += 1
                    if rel_counter[q] >= RELEASE_CONFIRM_STEPS:
                        pair_active[q]  = False
                        hold_counter[q] = 0
                        rel_counter[q]  = 0
                else:
                    rel_counter[q] = 0

        # ----------------------------------------------------------
        # Multi-disc CBF-QP  (NEW: passes thetas, uses disc pairs)
        # ----------------------------------------------------------
        u_safe = apply_cbf_qp_multidisc(
            p_current, thetas, u_nom, pair_active
        )

        # ----------------------------------------------------------
        # DDMR realisation  (unchanged)
        # ----------------------------------------------------------
        u_ddmr = np.zeros((2, 3))
        for i in range(3):
            cos_theta = math.cos(state[i, 2])
            sin_theta = math.sin(state[i, 2])

            speed_raw = cos_theta * u_safe[0, i] + sin_theta * u_safe[1, i]
            speed_raw = max(0.0, speed_raw)

            omega_raw = (-sin_theta * u_safe[0, i]
                         + cos_theta * u_safe[1, i]) / L_DDMR
            omega     = np.clip(omega_raw, -W_MAX_DDMR, W_MAX_DDMR)

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

        # DDMR propagation (exact ZOH)
        for i in range(3):
            v   = u_ddmr[0, i]
            omg = u_ddmr[1, i]
            th  = state[i, 2]
            if abs(omg) > 1e-10:
                state[i, 0] += (v / omg) * (math.sin(th + omg * DT) - math.sin(th))
                state[i, 1] -= (v / omg) * (math.cos(th + omg * DT) - math.cos(th))
            else:
                state[i, 0] += DT * v * math.cos(th)
                state[i, 1] += DT * v * math.sin(th)
            state[i, 2]  += omg * DT

        t += DT

        # Early stop when all rovers reach their goals
        if all(np.linalg.norm(state[i, :2] - TARGETS[i]) < GOAL_TOLERANCE
               for i in range(3)):
            break

    return {
        't':       np.array(t_log),
        'state':   np.array(state_log),     # (N, 3, 3)
        'u_nom':   np.array(u_nom_log),     # (N, 2, 3)
        'u_safe':  np.array(u_safe_log),    # (N, 2, 3)
        'u_ddmr':  np.array(u_ddmr_log),   # (N, 2, 3)
    }


# ============================================================
# PLOTS  (unchanged)
# ============================================================
def plot_trajectory_comparison(theory_result, gz_x, gz_y):
    fig, ax = plt.subplots(figsize=(12, 10))

    for i in range(3):
        xy = theory_result['state'][:, i, :2]
        ax.plot(xy[:, 0], xy[:, 1],
                color=COLORS[i], linestyle='--', linewidth=2.5,
                label=f'Rover {i+1} Theory (multi-disc DDMR)')
        ax.scatter(xy[0, 0], xy[0, 1], s=100, color=COLORS[i],
                   marker='s', edgecolors='black', linewidth=1.5, zorder=5)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=100, color=COLORS[i],
                   marker='o', edgecolors='black', linewidth=1.5, zorder=5)

    for i in range(3):
        if gz_x[i] is not None and len(gz_x[i]) > 0:
            ax.plot(gz_x[i], gz_y[i],
                    color=COLORS[i], linewidth=2,
                    label=f'Rover {i+1} Gazebo')
            ax.scatter(gz_x[i][0], gz_y[i][0], s=100, color=COLORS[i],
                       marker='s', alpha=0.7, edgecolors='black', zorder=4)
            ax.scatter(gz_x[i][-1], gz_y[i][-1], s=100, color=COLORS[i],
                       marker='o', alpha=0.7, edgecolors='black', zorder=4)

    for i, target in enumerate(TARGETS):
        ax.scatter(*target, s=200, color=COLORS[i], marker='*',
                   edgecolors='black', linewidth=1.5, zorder=6,
                   label=f'Target {i+1}')

    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title('Gazebo vs CBF-QP DDMR Theory (multi-disc footprint)',
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    textstr = (f'DT = {DT*1000:.1f} ms\n'
               f'Discs/rover = {N_DISCS}\n'
               f'Disc radius = {DISC_RADIUS*100:.0f} cm\n'
               f'Disc D_safe = {DISC_D_SAFE*100:.0f} cm\n'
               f'Offsets = ±9, ±3 cm\n'
               f'γ = {GAMMA_CBF:.1f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_velocity_profiles(v_raw, v_sat, w_raw, w_sat, t_logs):
    plt.figure(figsize=(12, 5))
    for i in range(3):
        if not t_logs[i]:
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

    plt.figure(figsize=(12, 5))
    for i in range(3):
        if not t_logs[i]:
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
# ROS2 NODE  (unchanged structure)
# ============================================================
class CbfTheoryComparison(Node):

    def __init__(self):
        super().__init__('cbf_theory_comparison')

        self.r1_done  = False
        self.r2_done  = False
        self.r3_done  = False
        self.finished = False

        self.dt          = 0.01
        self.V_THRESHOLD = 0.02

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

        self.first_pos = [None, None, None]

        self.csv_files   = []
        self.csv_writers = []
        for i in range(1, 4):
            f = open(f'rover_{i}.csv', 'w', newline='')
            w = csv.writer(f)
            w.writerow(['time', 'x', 'y'])
            self.csv_files.append(f)
            self.csv_writers.append(w)

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

        self.get_logger().info(
            'CbfTheoryComparison (multi-disc) started -- waiting for rovers.'
        )

    def _odom_cb(self, msg, idx):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z

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
            self.v_sat[idx].append(v)
            self.w_raw[idx].append(w)
            self.w_sat[idx].append(w)

        self.sample_cnt[idx] += 1

    def _goal_cb(self, msg, idx):
        if msg.data:
            if idx == 0:
                self.r1_done = True
            elif idx == 1:
                self.r2_done = True
            else:
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
        for f in self.csv_files:
            f.close()

        p0 = np.zeros((2, 3))
        defaults = [
            np.array([-5.0, -5.0]),
            np.array([-3.0, -7.0]),
            np.array([-7.0, -3.0])
        ]
        for i in range(3):
            if self.first_pos[i] is not None:
                p0[:, i] = self.first_pos[i]
            else:
                p0[:, i] = defaults[i]
                self.get_logger().warn(
                    f'No odom for rover {i+1}, using default spawn.'
                )

        self.get_logger().info(
            'Running multi-disc CBF-QP simulation with DDMR realization...'
        )
        theory = run_cbf_simulation_with_ddmr(p0)
        self.get_logger().info(
            f'Done: {len(theory["t"])} steps, T={theory["t"][-1]:.2f}s'
        )

        gz_x = [np.array(self.x_log[i]) if self.x_log[i] else None
                for i in range(3)]
        gz_y = [np.array(self.y_log[i]) if self.y_log[i] else None
                for i in range(3)]

        self.get_logger().info('Generating plots...')
        plot_trajectory_comparison(theory, gz_x, gz_y)
        plot_velocity_profiles(
            self.v_raw, self.v_sat,
            self.w_raw, self.w_sat,
            self.t_log
        )
        self.get_logger().info(
            'Saved: trajectory_comparison.png, '
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
