#!/usr/bin/env python3
"""
rover_1_tracker.py

PHASE 1 (runs at startup, before ROS):
  Simulates Rover 1 SI dynamics:
      u1 = -2*(p1 - p1_star)
  to produce a full SPATIAL path for Rover 1.

  Then runs the DDMR look-ahead solver at its own rate (DDMR_DT=0.02s)
  using ARC-LENGTH lookup on the SI path -- so the DDMR timing is
  completely decoupled from the SI timing. The DDMR solver keeps running
  until the simulated rover physically reaches the goal.

PHASE 2 (ROS2 node -- pure tracker, no online controller):
  Follows the pre-computed waypoints using look-ahead feedback
  linearization. Only needs own odom.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

# ============================================================
# MISSION GEOMETRY
# ============================================================
P1_0    = np.array([-6.0,  -5.0])   # Rover 1 spawn
P1_STAR = np.array([ 10.0,  10.0])   # Rover 1 goal

# ============================================================
# SI SIMULATION PARAMETERS
# ============================================================
SI_DT      = 0.005    # fine enough for accurate spatial path
SI_T_MAX   = 30.0
SI_GOAL_TOL = 0.02    # stop SI when rover 1 is very close to goal

# ============================================================
# DDMR SOLVER PARAMETERS
# ============================================================
L_DDMR          = 0.15    # look-ahead distance [m]
K_DDMR          = 2.0     # tracking gain [1/s]
V_MAX           = 0.30    # m/s
W_MAX           = 0.80    # rad/s
DDMR_DT         = 0.02    # DDMR solver step (matches ROS control rate)
DDMR_T_MAX      = 300.0   # generous ceiling for DDMR solver
DDMR_GOAL_TOL   = 0.05    # DDMR solver stops here
INITIAL_HEADING = math.radians(45.0)   # rover 1 initial heading

# ============================================================
# TRACKER PARAMETERS
# ============================================================
LOOK_AHEAD_DIST = 0.20
GOAL_TOLERANCE  = 0.30
CONTROL_DT      = 0.02


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# ============================================================
# PHASE 1A: SI SIMULATION -> SPATIAL PATH
# ============================================================
def compute_si_path():
    """
    Returns the full spatial SI path for Rover 1.
    p_si  : (N, 2) positions
    u_si  : (N, 2) velocities (= SI control u1)
    """
    p = P1_0.copy().astype(float)
    t = 0.0

    p_log, u_log = [], []

    while t < SI_T_MAX:
        u = -2.0 * (p - P1_STAR)
        p_log.append(p.copy())
        u_log.append(u.copy())

        if np.linalg.norm(p - P1_STAR) < SI_GOAL_TOL:
            break

        p = p + SI_DT * u
        t += SI_DT

    return np.array(p_log), np.array(u_log)


# ============================================================
# PHASE 1B: ARC-LENGTH PARAMETERISATION
# ============================================================
def build_arc_length(p_si):
    """
    Cumulative arc length along the SI path.
    Returns arc (N,) starting at 0.
    """
    N = len(p_si)
    arc = np.zeros(N)
    for k in range(1, N):
        arc[k] = arc[k - 1] + np.linalg.norm(p_si[k] - p_si[k - 1])
    return arc


def path_point_at_arc(arc, p_si, u_si, target_arc):
    """
    Linearly interpolate position and velocity at a given arc length.
    """
    target_arc = float(np.clip(target_arc, arc[0], arc[-1]))
    idx = int(np.searchsorted(arc, target_arc, side='left'))
    idx = min(idx, len(arc) - 1)

    if idx == 0 or arc[idx] == arc[idx - 1]:
        return p_si[idx].copy(), u_si[idx].copy()

    alpha = (target_arc - arc[idx - 1]) / (arc[idx] - arc[idx - 1])
    pos = p_si[idx - 1] + alpha * (p_si[idx] - p_si[idx - 1])
    vel = u_si[idx - 1] + alpha * (u_si[idx] - u_si[idx - 1])
    return pos, vel


def nearest_arc_ahead(arc, p_si, state_xy, prev_arc):
    """
    Find the arc-length of the SI point nearest to state_xy,
    searching only forward from prev_arc (monotonic).
    """
    i_start = max(0, int(np.searchsorted(arc, prev_arc)) - 1)
    best_arc = prev_arc
    best_dist = float('inf')
    for i in range(i_start, len(p_si)):
        d = np.linalg.norm(p_si[i] - state_xy)
        if d < best_dist:
            best_dist = d
            best_arc = arc[i]
        elif d > best_dist + 0.5:
            break
    return best_arc


# ============================================================
# PHASE 1C: DDMR LOOK-AHEAD SOLVER (arc-length based)
# ============================================================
def ddmr_exact_step(state, v, omega, dt):
    x, y, theta = state
    if abs(omega) > 1e-10:
        nx = x + (v / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
        ny = y - (v / omega) * (math.cos(theta + omega * dt) - math.cos(theta))
    else:
        nx = x + dt * v * math.cos(theta)
        ny = y + dt * v * math.sin(theta)
    return np.array([nx, ny, theta + dt * omega])


def compute_ddmr_waypoints(p_si, u_si):
    """
    Runs the DDMR look-ahead solver at DDMR_DT using arc-length
    lookup on the SI spatial path.  Completely decoupled from SI_DT.
    """
    arc = build_arc_length(p_si)

    state = np.array([P1_0[0], P1_0[1], INITIAL_HEADING])
    curr_arc = 0.0   # arc-length position of the nearest SI point

    wp_x, wp_y, wp_theta, wp_v, wp_omega = [], [], [], [], []

    t = 0.0
    while t < DDMR_T_MAX:
        x, y, th = state

        # 1. Find nearest SI point (arc-length), monotonically
        curr_arc = nearest_arc_ahead(arc, p_si, state[:2], curr_arc)

        # 2. Look-ahead point: L_DDMR metres ahead along arc
        la_arc = min(curr_arc + L_DDMR, arc[-1])
        la_pos, la_vel = path_point_at_arc(arc, p_si, u_si, la_arc)

        # 3. Actual look-ahead point on the DDMR rover
        lx = x + L_DDMR * math.cos(th)
        ly = y + L_DDMR * math.sin(th)

        # 4. Error
        ex = la_pos[0] - lx
        ey = la_pos[1] - ly

        # 5. Reference speed (capped) and direction from SI velocity
        v_ref_mag = float(np.linalg.norm(la_vel))
        if v_ref_mag > 1e-6:
            v_ref_dir = la_vel / v_ref_mag
        else:
            v_ref_dir = np.array([math.cos(th), math.sin(th)])
        v_ref_mag = min(v_ref_mag, V_MAX)

        # 6. Desired look-ahead velocity
        vdx = v_ref_mag * v_ref_dir[0] + K_DDMR * ex
        vdy = v_ref_mag * v_ref_dir[1] + K_DDMR * ey

        # 7. Inverse decoupling
        c = math.cos(th)
        s = math.sin(th)
        v_raw = c * vdx + s * vdy
        omega_raw = (-s * vdx + c * vdy) / L_DDMR

        v_cmd = float(np.clip(v_raw, -V_MAX, V_MAX))
        omega_cmd = float(np.clip(omega_raw, -W_MAX, W_MAX))

        wp_x.append(x);    wp_y.append(y)
        wp_theta.append(th)
        wp_v.append(v_cmd); wp_omega.append(omega_cmd)

        # 8. Goal check on DDMR rover position
        if np.linalg.norm(state[:2] - P1_STAR) < DDMR_GOAL_TOL:
            print(f"[DDMR solver R1] Goal reached at t={t:.2f}s")
            break

        state = ddmr_exact_step(state, v_cmd, omega_cmd, DDMR_DT)
        t += DDMR_DT

    return (np.array(wp_x), np.array(wp_y),
            np.array(wp_theta), np.array(wp_v), np.array(wp_omega))


# ============================================================
# RUN PHASE 1 AT MODULE LOAD
# ============================================================
print("[DDMR solver R1] Computing SI path...")
_p1_si, _u1_si = compute_si_path()
print(f"[DDMR solver R1] SI done: {len(_p1_si)} points, "
      f"path length={build_arc_length(_p1_si)[-1]:.2f}m, "
      f"final dist to goal={np.linalg.norm(_p1_si[-1]-P1_STAR):.4f}m")

print("[DDMR solver R1] Running DDMR arc-length solver...")
_WP_X, _WP_Y, _WP_THETA, _WP_V, _WP_OMEGA = \
    compute_ddmr_waypoints(_p1_si, _u1_si)
_N_WP = len(_WP_X)
print(f"[DDMR solver R1] {_N_WP} waypoints generated. "
      f"Final wp=({_WP_X[-1]:.3f},{_WP_Y[-1]:.3f}), "
      f"goal=({P1_STAR[0]},{P1_STAR[1]})")


# ============================================================
# PHASE 2: ROS2 LOOK-AHEAD TRACKER NODE
# ============================================================
class Rover1DdmrTracker(Node):

    def __init__(self):
        super().__init__('rover_1_ddmr_tracker')

        self.wp_x = _WP_X
        self.wp_y = _WP_Y
        self.wp_theta = _WP_THETA
        self.wp_v = _WP_V
        self.wp_omega = _WP_OMEGA
        self.n_wp = _N_WP

        self.px = 0.0
        self.py = 0.0
        self.theta = 0.0
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

        self.create_subscription(
            Odometry, '/rover_1/odom', self.odom_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, '/rover_1/cmd_vel', 10)
        self.goal_pub = self.create_publisher(Bool, '/rover_1/goal_reached', 10)
        self.timer = self.create_timer(CONTROL_DT, self.control_loop)

        self.get_logger().info(
            f'Rover 1 DDMR tracker ready -- {self.n_wp} waypoints loaded. '
            f'Goal=({P1_STAR[0]:.1f},{P1_STAR[1]:.1f})'
        )

    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    def _advance_waypoint_index(self):
        while self.wp_idx < self.n_wp - 1:
            dx = self.wp_x[self.wp_idx] - self.px
            dy = self.wp_y[self.wp_idx] - self.py
            if math.sqrt(dx * dx + dy * dy) > LOOK_AHEAD_DIST:
                break
            self.wp_idx += 1

    # ----------------------------------------------------------
    def _save_and_flag(self):
        if self.profile_saved:
            return
        np.savez(
            'rover_1_profile.npz',
            t=np.array(self.time_log),
            v_raw=np.array(self.v_raw_log),
            w_raw=np.array(self.w_raw_log),
            v_sat=np.array(self.v_sat_log),
            w_sat=np.array(self.w_sat_log)
        )
        np.savez(
            'rover_1_odom_log.npz',
            t=np.array(self.odom_time_log),
            x=np.array(self.odom_x_log),
            y=np.array(self.odom_y_log),
            v=np.array(self.odom_v_log),
            w=np.array(self.odom_w_log)
        )
        np.savez(
            'rover_1_waypoints.npz',
            t=np.array(self.waypoint_time_log),
            x=np.array(self.waypoint_x_log),
            y=np.array(self.waypoint_y_log),
            v=np.array(self.waypoint_v_log),
            w=np.array(self.waypoint_w_log)
        )
        self.profile_saved = True
        self.get_logger().info('Logs saved: rover_1_profile/odom_log/waypoints.npz')

    # ----------------------------------------------------------
    def control_loop(self):
        if not self.have_odom:
            return

        dist_to_goal = math.sqrt(
            (self.px - float(self.wp_x[-1])) ** 2 +
            (self.py - float(self.wp_y[-1])) ** 2
        )
        if dist_to_goal < GOAL_TOLERANCE:
            self.goal_reached = True

        if self.goal_reached:
            self.cmd_pub.publish(Twist())
            msg = Bool()
            msg.data = True
            self.goal_pub.publish(msg)
            self.get_logger().info(f'Goal reached. dist={dist_to_goal:.3f} m')
            self._save_and_flag()
            return

        self._advance_waypoint_index()
        idx = self.wp_idx

        tx = float(self.wp_x[idx])
        ty = float(self.wp_y[idx])

        lx = self.px + L_DDMR * math.cos(self.theta)
        ly = self.py + L_DDMR * math.sin(self.theta)

        ex = tx - lx
        ey = ty - ly

        v_ref = float(self.wp_v[idx])
        vdx = v_ref * math.cos(float(self.wp_theta[idx])) + K_DDMR * ex
        vdy = v_ref * math.sin(float(self.wp_theta[idx])) + K_DDMR * ey

        c = math.cos(self.theta)
        s = math.sin(self.theta)
        v_raw = c * vdx + s * vdy
        omega_raw = (-s * vdx + c * vdy) / L_DDMR

        v_sat = float(np.clip(v_raw, -V_MAX, V_MAX))
        omega_sat = float(np.clip(omega_raw, -W_MAX, W_MAX))

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
        self.waypoint_x_log.append(float(self.wp_x[idx]))
        self.waypoint_y_log.append(float(self.wp_y[idx]))
        self.waypoint_v_log.append(float(self.wp_v[idx]))
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
    node = Rover1DdmrTracker()
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
