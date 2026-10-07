#!/usr/bin/env python3
"""
rover_1_cbf_tracker.py – integrated look-ahead CBF-QP generator + tracker
Now uses holonomic CBF-QP in the global frame, then DDMR realisation,
matching the behaviour of cbf_theory_comparison.py.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import csv

# ============================================================
# MISSION PARAMETERS
# ============================================================
P1_0    = np.array([0.0, 0.0])
P2_0    = np.array([3.0, 0.0])
P3_0    = np.array([0.0, 4.0])
P1_STAR = np.array([ 5.0,  5.0])
R12     = np.array([-2.0,  0.0])
R13     = np.array([ 0.0, -2.0])
P2_STAR = P1_STAR + R12   # (3, 5)
P3_STAR = P1_STAR + R13   # (5, 3)

PAIRS = [(0, 1), (0, 2), (1, 2)]

# ============================================================
# CBF / SAFETY PARAMETERS
# ============================================================
ROBOT_RADIUS     = 0.25
SAFETY_MARGIN    = 0.30
ERROR_MARGIN     = 0.20
D_SAFE           = ROBOT_RADIUS + ROBOT_RADIUS + SAFETY_MARGIN  # 1.00 m
R_VIS            = 2.5
R_TRIGGER        = 2.0       # increased to force activation (like old code)
R_RELEASE        = 2.2
GAMMA_CBF        = 5.0
PSI_ACTIVATE_TOL = 1e-9
PSI_RELEASE      = 0.25
MIN_ACTIVE_TIME       = 0.10
RELEASE_CONFIRM_TIME  = 0.10

# ============================================================
# SI SIMULATION PARAMETERS
# ============================================================
SI_DT       = 0.005
SI_T_MAX    = 60.0
SI_GOAL_TOL = 0.02

_SI_MIN_ACTIVE_STEPS      = max(1, math.ceil(MIN_ACTIVE_TIME / SI_DT))
_SI_RELEASE_CONFIRM_STEPS = max(1, math.ceil(RELEASE_CONFIRM_TIME / SI_DT))

# ============================================================
# DDMR / TRACKER PARAMETERS (unified)
# ============================================================
L_DDMR        = 0.202        # look-ahead distance (solver and tracker)
K_DDMR        = 2.0           # tracking gain (solver and tracker)
V_MAX         = 0.30
W_MAX         = 0.80
DDMR_DT       = 0.005          # solver step (same as control rate)
DDMR_T_MAX    = 300.0
DDMR_GOAL_TOL = 0.15
INIT_HEADING_R1 = math.radians(45.0)

# ============================================================
# TRACKER PARAMETERS
# ============================================================
L_DDMR_TRACK    = 0.202 # same as solver
K_DDMR_TRACK    = 2.0
V_MAX_TRACK     = 0.30
W_MAX_TRACK     = 0.80
LOOK_AHEAD_DIST = 0.03
GOAL_TOLERANCE  = 0.20
CONTROL_DT      = 0.005

def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

# ============================================================
# NOMINAL LQ CONTROL
# ============================================================
def nominal_control(p):
    x1 = p[:, 0] - P1_STAR
    x2 = (p[:, 1] - p[:, 0]) - R12
    x3 = (p[:, 2] - p[:, 0]) - R13
    u1 = -2.0 * x1
    u2 = -(x1 + 2.0 * x2)
    u3 = -(x1 + 2.0 * x3)
    return np.column_stack([u1, u2, u3])

# ============================================================
# CBF-QP SOLVER (holonomic, 2D velocity vector)
# ============================================================
def solve_projection_qp_2d(u_nom, A_rows, b_vals):
    if not A_rows:
        return u_nom.copy()
    TOL = 1e-10

    u_nom = np.array(u_nom)
    A_rows = [np.array(a) for a in A_rows]
    b_vals = np.array(b_vals)

    def feasible(u):
        for a, b in zip(A_rows, b_vals):
            if np.dot(a, u) > b + TOL:
                return False
        return True

    candidates = []
    if feasible(u_nom):
        candidates.append(u_nom.copy())

    for a, b in zip(A_rows, b_vals):
        denom = np.dot(a, a)
        if denom <= TOL:
            continue
        scale = (np.dot(a, u_nom) - b) / denom
        cand = u_nom - scale * a
        if feasible(cand):
            candidates.append(cand)

    n = len(A_rows)
    for ell in range(n):
        for m in range(ell + 1, n):
            M = np.array([A_rows[ell], A_rows[m]])
            det = np.linalg.det(M)
            if abs(det) <= TOL:
                continue
            cand = np.linalg.solve(M, np.array([b_vals[ell], b_vals[m]]))
            if feasible(cand):
                candidates.append(cand)

    if not candidates:
        zero = np.zeros_like(u_nom)
        return zero if feasible(zero) else u_nom.copy()

    best = min(candidates, key=lambda c: np.sum((c - u_nom)**2))
    return best

# ============================================================
# HOLONOMIC CBF-QP (full 2D velocities) then DDMR realisation
# ============================================================
def apply_cbf_qp_holonomic(state, u_nom_xy, pair_active):
    """
    u_nom_xy : (2, 3) – nominal velocities in global frame
    returns u_safe_xy (2,3) – safe velocities in global frame
    """
    u_safe = np.zeros((2, 3))
    for i in range(3):
        A_rows, b_vals = [], []
        # optional velocity limits in 2D? Usually we don't limit here,
        # because DDMR realisation will handle speed limits later.

        xi, yi, _ = state[i]
        p_i = np.array([xi, yi])

        for q, (pi, pj) in enumerate(PAIRS):
            if not pair_active[q]:
                continue
            if i not in (pi, pj):
                continue

            j = pj if pi == i else pi
            xj, yj, _ = state[j]
            p_j = np.array([xj, yj])

            rel = p_i - p_j
            dist_center = np.linalg.norm(rel)
            h_ij = dist_center**2 - D_SAFE**2

            # Constraint: -2*rel·u_safe_i <= 0.5*gamma*h_ij
            # i.e., a·u <= b with a = -2*rel, b = 0.5*gamma*h_ij
            a = -2.0 * rel
            b = 0.5 * GAMMA_CBF * h_ij
            A_rows.append(a)
            b_vals.append(b)

        u_nom_i = u_nom_xy[:, i]
        u_safe[:, i] = solve_projection_qp_2d(u_nom_i, A_rows, b_vals)
    return u_safe

# ============================================================
# DDMR realisation (same as cbf_theory_comparison.py)
# ============================================================
def ddmr_realisation(state, u_safe_xy):
    """
    Convert safe 2D velocities to DDMR commands (v, omega) using feedback linearisation
    and speed scaling matching the real controller.
    """
    u_ddmr = np.zeros((2, 3))  # [v, omega]
    for i in range(3):
        cos_theta = math.cos(state[i, 2])
        sin_theta = math.sin(state[i, 2])

        speed_raw = cos_theta * u_safe_xy[0, i] + sin_theta * u_safe_xy[1, i]
        speed_raw = max(0.0, speed_raw)   # no backward motion

        omega_raw = (-sin_theta * u_safe_xy[0, i] + cos_theta * u_safe_xy[1, i]) / L_DDMR
        omega = np.clip(omega_raw, -W_MAX, W_MAX)

        # Speed scaling (matching original controller)
        OMEGA_REF = 5.0
        SPEED_MIN = 0.05
        SPEED_MAX = V_MAX
        scale = np.exp(-abs(omega_raw) / OMEGA_REF)
        speed = SPEED_MIN + (SPEED_MAX - SPEED_MIN) * scale
        speed = min(speed, speed_raw)
        speed = np.clip(speed, 0.0, V_MAX)

        u_ddmr[0, i] = speed
        u_ddmr[1, i] = omega
    return u_ddmr

# ============================================================
# CBF-QP SIMULATION WITH HOLONOMIC CBF + DDMR REALISATION
# ============================================================
def run_cbf_simulation_with_ddmr(p0):
    state = np.zeros((3, 3))
    state[:, 0] = p0[0, :]
    state[:, 1] = p0[1, :]
    state[:, 2] = np.radians([45.0, 0.0, 0.0])

    pair_active = [False, False, False]
    hold_counter = [0, 0, 0]
    rel_counter = [0, 0, 0]

    t_log = []
    state_log = []
    u_ddmr_log = []

    goals = np.array([P1_STAR, P2_STAR, P3_STAR])

    csv_file = open('cbf_activations.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'time', 'pair_idx', 'dist_center', 'h_ij', 'psi', 'active',
        'robot', 'v_nom_xy_x', 'v_nom_xy_y', 'v_safe_xy_x', 'v_safe_xy_y',
        'goal_x', 'goal_y', 'dist_to_goal'
    ])

    t = 0.0
    while t < SI_T_MAX:
        p_current = state[:, :2].T
        u_nom_xy = nominal_control(p_current)   # (2,3)

        # Compute distances, h, psi (psi uses u_nom_xy)
        dist = np.zeros(3)
        h = np.zeros(3)
        psi = np.zeros(3)
        vis = np.zeros(3, dtype=bool)

        for q, (i, j) in enumerate(PAIRS):
            pi = p_current[:, i]
            pj = p_current[:, j]
            rel = pi - pj
            dist[q] = np.linalg.norm(rel)
            h[q] = dist[q]**2 - D_SAFE**2
            # psi = 2*rel·u_nom_i + 0.5*gamma*h
            psi[q] = 2.0 * np.dot(rel, u_nom_xy[:, i]) + 0.5 * GAMMA_CBF * h[q]
            vis[q] = dist[q] <= R_VIS

        # Activation logic
        for q in range(3):
            if not pair_active[q]:
                rel_counter[q] = 0
                if (vis[q] and dist[q] <= R_TRIGGER and psi[q] < -PSI_ACTIVATE_TOL):
                    pair_active[q] = True
                    hold_counter[q] = _SI_MIN_ACTIVE_STEPS
                    rel_counter[q] = 0
            elif hold_counter[q] > 0:
                hold_counter[q] -= 1
                rel_counter[q] = 0
            else:
                if dist[q] >= R_RELEASE and psi[q] >= PSI_RELEASE:
                    rel_counter[q] += 1
                    if rel_counter[q] >= _SI_RELEASE_CONFIRM_STEPS:
                        pair_active[q] = False
                        hold_counter[q] = 0
                        rel_counter[q] = 0
                else:
                    rel_counter[q] = 0

        # Holonomic CBF-QP
        u_safe_xy = apply_cbf_qp_holonomic(state, u_nom_xy, pair_active)

        # DDMR realisation
        u_ddmr = ddmr_realisation(state, u_safe_xy)

        # Logging (per robot, per pair – we log only for active pairs for clarity)
        for q, (i, j) in enumerate(PAIRS):
            for robot in (i, j):
                dist_to_goal = np.linalg.norm(state[robot, :2] - goals[robot])
                csv_writer.writerow([
                    t, q, dist[q], h[q], psi[q], int(pair_active[q]),
                    robot,
                    u_nom_xy[0, robot], u_nom_xy[1, robot],
                    u_safe_xy[0, robot], u_safe_xy[1, robot],
                    goals[robot][0], goals[robot][1], dist_to_goal
                ])

        t_log.append(t)
        state_log.append(state.copy())
        u_ddmr_log.append(u_ddmr.copy())

        # Update state using DDMR kinematics
        for i in range(3):
            v = u_ddmr[0, i]
            omega = u_ddmr[1, i]
            if abs(omega) > 1e-10:
                state[i, 0] += (v / omega) * (math.sin(state[i, 2] + omega * SI_DT) - math.sin(state[i, 2]))
                state[i, 1] -= (v / omega) * (math.cos(state[i, 2] + omega * SI_DT) - math.cos(state[i, 2]))
            else:
                state[i, 0] += SI_DT * v * math.cos(state[i, 2])
                state[i, 1] += SI_DT * v * math.sin(state[i, 2])
            state[i, 2] += omega * SI_DT

        t += SI_DT
        if np.all([np.linalg.norm(state[i,:2] - goals[i]) < GOAL_TOLERANCE for i in range(3)]):
            break

    csv_file.close()
    print(f"Activation log saved to cbf_activations.csv")

    # Final checks
    if len(state_log) > 0:
        min_center = min([np.linalg.norm(state_log[k][i, :2] - state_log[k][j, :2])
                          for k in range(len(state_log)) for i in range(3) for j in range(i+1,3)])
        print(f"Minimum center distance during simulation: {min_center:.3f} m")
        safe_ok = min_center >= D_SAFE - 1e-6
        print(f"[CBF check] D_SAFE = {D_SAFE:.3f} m -> "
              f"{'OK (constraint respected)' if safe_ok else 'VIOLATED'}")
    else:
        print("No states logged.")

    # Final waypoints logging
    print("[Final waypoints] ---------------------------------------------")
    final_rows = []
    for i in range(3):
        final_state = state_log[-1][i] if state_log else state[i]
        goal = goals[i]
        err = np.linalg.norm(final_state[:2] - goal)
        row = {
            'robot': i + 1,
            'goal_x': goal[0], 'goal_y': goal[1],
            'final_x': final_state[0], 'final_y': final_state[1],
            'final_theta': final_state[2],
            'goal_error': err,
            'within_tol': bool(err < GOAL_TOLERANCE),
        }
        final_rows.append(row)
        print(f"  Rover {i+1}: goal=({goal[0]:.3f}, {goal[1]:.3f})  "
              f"final=({final_state[0]:.3f}, {final_state[1]:.3f})  "
              f"error={err:.3f} m  {'[OK]' if row['within_tol'] else '[NOT REACHED]'}")

    with open('cbf_final_waypoints.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['robot', 'goal_x', 'goal_y', 'final_x', 'final_y',
                    'final_theta', 'goal_error', 'within_tol'])
        for row in final_rows:
            w.writerow([row['robot'], row['goal_x'], row['goal_y'],
                        row['final_x'], row['final_y'], row['final_theta'],
                        row['goal_error'], row['within_tol']])
    print("[Final waypoints] Saved to cbf_final_waypoints.csv")
    print("[Final waypoints] ---------------------------------------------")

    return {
        't': np.array(t_log),
        'state': np.array(state_log),
        'u_ddmr': np.array(u_ddmr_log),
    }

# ============================================================
# RUN GENERATOR AT MODULE LOAD AND SAVE WAYPOINTS
# ============================================================
print("[CBF-Lookahead R1] Generating all rover waypoints (holonomic CBF)...")
p0 = np.array([
    [0.0, 0.0],
    [3.0, 0.0],
    [0.0, 4.0]
]).T
result = run_cbf_simulation_with_ddmr(p0)
print(f"[CBF-Lookahead R1] Simulation done: {len(result['t'])} steps, T={result['t'][-1]:.2f}s")

state_log = result['state']
u_ddmr_log = result['u_ddmr']

for i in range(3):
    x = state_log[:, i, 0]
    y = state_log[:, i, 1]
    theta = state_log[:, i, 2]
    v = u_ddmr_log[:, 0, i]
    omega = u_ddmr_log[:, 1, i]
    arc = np.zeros(len(x))
    for k in range(1, len(x)):
        arc[k] = arc[k-1] + math.hypot(x[k]-x[k-1], y[k]-y[k-1])
    filename = f'cbf_rover_{i+1}_waypoints.npz'
    np.savez(filename, x=x, y=y, theta=theta, v=v, omega=omega, arc=arc)
    print(f"[CBF-Lookahead R1] Saved {filename} ({len(x)} points)")

_WP_X = state_log[:, 0, 0]
_WP_Y = state_log[:, 0, 1]
_WP_THETA = state_log[:, 0, 2]
_WP_V = u_ddmr_log[:, 0, 0]
_WP_OMEGA = u_ddmr_log[:, 1, 0]
_N_WP = len(_WP_X)

print(f"[CBF-Lookahead R1] Rover 1 waypoints loaded: {_N_WP} points.")

# ============================================================
# PHASE 2: ROS2 LOOK-AHEAD TRACKER NODE (unchanged)
# ============================================================
class Rover1CbfTracker(Node):
    def __init__(self):
        super().__init__('rover_1_cbf_tracker')
        self.wp_x = _WP_X
        self.wp_y = _WP_Y
        self.wp_theta = _WP_THETA
        self.wp_v = _WP_V
        self.wp_omega = _WP_OMEGA
        self.n_wp = _N_WP

        self.px = self.py = self.theta = 0.0
        self.have_odom = False
        self.goal_reached = False
        self.profile_saved = False
        self.wp_idx = 0

        self.time_log = []
        self.v_raw_log = []
        self.w_raw_log = []
        self.v_sat_log = []
        self.w_sat_log = []

        self.odom_time_log = []
        self.odom_x_log = []
        self.odom_y_log = []
        self.odom_v_log = []
        self.odom_w_log = []

        self.waypoint_time_log = []
        self.waypoint_x_log = []
        self.waypoint_y_log = []
        self.waypoint_v_log = []
        self.waypoint_w_log = []

        self.create_subscription(Odometry, '/rover_1/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/rover_1/cmd_vel', 10)
        self.goal_pub = self.create_publisher(Bool, '/rover_1/goal_reached', 10)
        self.timer = self.create_timer(CONTROL_DT, self.control_loop)

        self.get_logger().info(
            f'Rover 1 CBF tracker ready -- {self.n_wp} waypoints loaded. '
            f'Goal = ({P1_STAR[0]:.1f}, {P1_STAR[1]:.1f})'
        )

    def odom_callback(self, msg):
        self.px = msg.pose.pose.position.x
        self.py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.theta = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.have_odom = True
        t = self.get_clock().now().nanoseconds * 1e-9
        self.odom_time_log.append(t)
        self.odom_x_log.append(self.px)
        self.odom_y_log.append(self.py)
        self.odom_v_log.append(msg.twist.twist.linear.x)
        self.odom_w_log.append(msg.twist.twist.angular.z)

    def _advance_waypoint_index(self):
        search_end = min(self.wp_idx + 250, self.n_wp)

        dx = self.wp_x[self.wp_idx:search_end] - self.px
        dy = self.wp_y[self.wp_idx:search_end] - self.py
        local_closest = int(np.argmin(dx*dx + dy*dy))

        closest_idx = self.wp_idx + local_closest

        forward_steps = 8
        self.wp_idx = min(closest_idx + forward_steps, self.n_wp - 1)

    def _save_logs(self):
        if self.profile_saved:
            return
        np.savez('rover_1_cbf_profile.npz',
                 t=np.array(self.time_log),
                 v_raw=np.array(self.v_raw_log),
                 w_raw=np.array(self.w_raw_log),
                 v_sat=np.array(self.v_sat_log),
                 w_sat=np.array(self.w_sat_log))
        np.savez('rover_1_cbf_odom_log.npz',
                 t=np.array(self.odom_time_log),
                 x=np.array(self.odom_x_log),
                 y=np.array(self.odom_y_log),
                 v=np.array(self.odom_v_log),
                 w=np.array(self.odom_w_log))
        np.savez('rover_1_cbf_tracked_waypoints.npz',
                 t=np.array(self.waypoint_time_log),
                 x=np.array(self.waypoint_x_log),
                 y=np.array(self.waypoint_y_log),
                 v=np.array(self.waypoint_v_log),
                 w=np.array(self.waypoint_w_log))
        self.profile_saved = True
        self.get_logger().info('Logs saved: rover_1_cbf_*')

        if len(self.waypoint_x_log) > 0 and len(self.odom_x_log) > 0:
            final_wp = (self.waypoint_x_log[-1], self.waypoint_y_log[-1])
            final_odom = (self.odom_x_log[-1], self.odom_y_log[-1])
            err = math.hypot(final_odom[0] - P1_STAR[0], final_odom[1] - P1_STAR[1])
            with open('rover_1_cbf_final_waypoint.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['goal_x', 'goal_y', 'final_tracked_wp_x', 'final_tracked_wp_y',
                            'final_odom_x', 'final_odom_y', 'goal_error'])
                w.writerow([P1_STAR[0], P1_STAR[1], final_wp[0], final_wp[1],
                            final_odom[0], final_odom[1], err])
            self.get_logger().info(
                f'[Final waypoint] goal=({P1_STAR[0]:.3f},{P1_STAR[1]:.3f}) '
                f'final_odom=({final_odom[0]:.3f},{final_odom[1]:.3f}) '
                f'error={err:.3f} m -> saved rover_1_cbf_final_waypoint.csv'
            )

    def control_loop(self):
        if not self.have_odom:
            return

        dist_to_goal = math.hypot(self.px - self.wp_x[-1], self.py - self.wp_y[-1])
        if dist_to_goal < GOAL_TOLERANCE:
            self.goal_reached = True

        if self.goal_reached:
            self.cmd_pub.publish(Twist())
            msg = Bool(); msg.data = True
            self.goal_pub.publish(msg)
            self.get_logger().info(f'Goal reached. dist={dist_to_goal:.3f}m')
            self._save_logs()
            return

        self._advance_waypoint_index()
        idx = self.wp_idx

        tx = self.wp_x[idx]
        ty = self.wp_y[idx]

        lx = self.px + L_DDMR_TRACK * math.cos(self.theta)
        ly = self.py + L_DDMR_TRACK * math.sin(self.theta)
        ex = tx - lx
        ey = ty - ly

        v_ref = self.wp_v[idx]
        th_ref = self.wp_theta[idx]

        c = math.cos(self.theta)
        s = math.sin(self.theta)

        e_long = c * ex + s * ey
        e_lat  = -s * ex + c * ey

        v_raw = v_ref + K_DDMR_TRACK * e_long

        omega_raw = (
            self.wp_omega[idx]
            + (K_DDMR_TRACK / L_DDMR_TRACK) * e_lat
        )
        v_sat = float(np.clip(v_raw, 0.0, V_MAX_TRACK))
        omega_sat = float(np.clip(omega_raw, -W_MAX_TRACK, W_MAX_TRACK))

        cmd = Twist()
        cmd.linear.x = v_sat
        cmd.angular.z = omega_sat
        self.cmd_pub.publish(cmd)

        now = self.get_clock().now().nanoseconds * 1e-9
        self.time_log.append(now)
        self.v_raw_log.append(v_raw)
        self.w_raw_log.append(omega_raw)
        self.v_sat_log.append(v_sat)
        self.w_sat_log.append(omega_sat)

        self.waypoint_time_log.append(now)
        self.waypoint_x_log.append(tx)
        self.waypoint_y_log.append(ty)
        self.waypoint_v_log.append(v_ref)
        self.waypoint_w_log.append(float(self.wp_omega[idx]))

        self.get_logger().info(
            f'p1=({self.px:.2f},{self.py:.2f}) '
            f'wp={idx}/{self.n_wp-1} '
            f'v={v_sat:.3f} w={omega_sat:.3f}'
        )

# ============================================================
# MAIN
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = Rover1CbfTracker()
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
