#!/usr/bin/env python3

import math
import logging
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class CbfController1(Node):
    def __init__(self):
        super().__init__('cbf_controller_1')

        # ===== FILE LOGGER =====
        self.event_logger = logging.getLogger('cbf_event_logger_1')
        self.event_logger.setLevel(logging.INFO)
        if not self.event_logger.handlers:
            file_handler = logging.FileHandler('cbf_events_1.log')
            formatter = logging.Formatter(
                '%(asctime)s.%(msecs)03d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.event_logger.addHandler(file_handler)

        # ===============================
        # States
        # ===============================
        self.px = 0.0
        self.py = 0.0
        self.theta = 0.0

        # ===== HARDCODED GOAL (MATCHES THEORY) =====
        self.goal_x = 5.0
        self.goal_y = 5.0
        self.have_goal = True   # Always have goal

        self.have_odom = False
        self.goal_reached = False
        self.GOAL_TOLERANCE = 0.3
        self.profile_saved = False

        # ===== DDMR parameters =====
        self.l = 0.20
        self.speed_max = 0.5
        self.omega_max = 0.8

        # ===== Low‑pass filter =====
        self.alpha = 0.3
        self.speed_filtered = 0.0
        self.omega_filtered = 0.0

        # ===== Multi‑disc footprint parameters (NEW) =====
        self.DISC_OFFSETS = [
            ( 0.09, 0.0),
            ( 0.03, 0.0),
            (-0.03, 0.0),
            (-0.09, 0.0),
        ]
        self.DISC_RADIUS = 0.05
        self.DISC_D_SAFE = 2.0 * self.DISC_RADIUS   # 0.10 m
        self.N_DISCS = len(self.DISC_OFFSETS)

        # ===== Logging =====
        self.time_log = []
        self.v_raw_log = []
        self.w_raw_log = []
        self.v_sat_log = []
        self.w_sat_log = []
        self.u_nom_x_log = []
        self.u_nom_y_log = []
        self.u_qp_x_log = []
        self.u_qp_y_log = []
        self.distance_logs = {2: [], 3: []}
        self.barrier_logs = {2: [], 3: []}
        self.safety_link_active = {2: False, 3: False}
        self.safety_link_logs = {2: [], 3: []}
        self.current_distance = {2: np.nan, 3: np.nan}
        self.current_barrier = {2: np.nan, 3: np.nan}
        self.current_safety_link = {2: 0, 3: 0}

        # ===============================
        # Subscribers
        # ===============================
        self.odom_sub = self.create_subscription(
            Odometry,
            '/rover_1/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/rover_1/scan",
            self.scan_callback,
            10
        )

        # Goal subscription kept but ignored (hardcoded)
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
        self.timer = self.create_timer(0.02, self.control_loop)

        self.cluster_threshold = 0.10
        self.min_cluster_points = 2

        self.visible_neighbors = {}

        # ===== CBF-QP Parameters (unchanged) =====
        self.CONTROL_DT = 0.02
        self.ROBOT_RADIUS = 0.25
        self.SAFETY_MARGIN = 0.2
        self.D_SAFE = self.ROBOT_RADIUS + self.ROBOT_RADIUS + self.SAFETY_MARGIN
        self.R_VIS = 2.5
        self.R_TRIGGER = 2.0
        self.R_RELEASE = 2.2
        self.GAMMA_CBF = 5.0
        self.PSI_ACTIVATE_TOL = 1e-9
        self.PSI_RELEASE = 0.25
        self.MIN_ACTIVE_TIME = 0.10
        self.RELEASE_CONFIRM_TIME = 0.10
        self.MIN_ACTIVE_STEPS = max(1, math.ceil(self.MIN_ACTIVE_TIME / self.CONTROL_DT))
        self.RELEASE_CONFIRM_STEPS = max(1, math.ceil(self.RELEASE_CONFIRM_TIME / self.CONTROL_DT))

        self.other_rover_ids = [2, 3]
        self.rover_tracks = {}
        for rid in self.other_rover_ids:
            self.rover_tracks[rid] = {
                'pair_active': False,
                'hold_counter': 0,
                'release_counter': 0,
            }

        # ===== Neighbour positions now include theta (CHANGED) =====
        self.other_rover_positions = {
            rid: {'x': 0.0, 'y': 0.0, 'theta': 0.0, 'have_odom': False}
            for rid in self.other_rover_ids
        }

        self.rover_odom_subs = {rid: None for rid in self.other_rover_ids}

        self.get_logger().info('Rover 1 Controller Started (multi‑disc CBF, goal hardcoded to (5,5))')

    # ============================================================
    #  CALLBACKS
    # ============================================================
    def odom_callback(self, msg):
        self.px = msg.pose.pose.position.x
        self.py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.theta = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.have_odom = True

    def goal_callback(self, msg):
        # Ignore external goal – we use hardcoded (5,5)
        pass

    def rover_2_odom_callback(self, msg):
        self.other_rover_positions[2]['x'] = msg.pose.pose.position.x
        self.other_rover_positions[2]['y'] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.other_rover_positions[2]['theta'] = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.other_rover_positions[2]['have_odom'] = True

    def rover_3_odom_callback(self, msg):
        self.other_rover_positions[3]['x'] = msg.pose.pose.position.x
        self.other_rover_positions[3]['y'] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.other_rover_positions[3]['theta'] = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.other_rover_positions[3]['have_odom'] = True

    def scan_callback(self, msg):
        self.latest_scan = msg
        self.identify_visible_robots()

    # ------------------------------------------------------------
    #  LiDAR clustering (EXACTLY AS ORIGINAL – unchanged)
    # ------------------------------------------------------------
    def extract_clusters(self):
        if not hasattr(self, "latest_scan"):
            return []
        scan = self.latest_scan
        clusters = []
        current_cluster = []
        angle = scan.angle_min
        prev_point = None
        for r in scan.ranges:
            if np.isinf(r) or np.isnan(r):
                if len(current_cluster) >= self.min_cluster_points:
                    clusters.append(current_cluster)
                current_cluster = []
                prev_point = None
                angle += scan.angle_increment
                continue
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            point = (x, y, r, angle)
            if prev_point is None:
                current_cluster.append(point)
            else:
                d = np.hypot(x - prev_point[0], y - prev_point[1])
                if d < self.cluster_threshold:
                    current_cluster.append(point)
                else:
                    if len(current_cluster) >= self.min_cluster_points:
                        clusters.append(current_cluster)
                    current_cluster = [point]
            prev_point = point
            angle += scan.angle_increment
        if len(current_cluster) >= self.min_cluster_points:
            clusters.append(current_cluster)
        return clusters

    def cluster_center(self, cluster):
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        xc = np.mean(xs)
        yc = np.mean(ys)
        r = np.hypot(xc, yc)
        bearing = np.arctan2(yc, xc)
        return r, bearing

    def identify_visible_robots(self):
        self.visible_neighbors = {}
        clusters = self.extract_clusters()
        self.event_logger.info(f"Clusters identified: {len(clusters)}")

        in_range_clusters = [
            c for c in clusters
            if self.cluster_center(c)[0] <= self.R_VIS
        ]

        self.update_neighbor_subscriptions(
            any_cluster_in_range=bool(in_range_clusters)
        )

        candidate_pairs = []
        for cluster_index, cluster in enumerate(in_range_clusters):
            measured_range, measured_bearing = self.cluster_center(cluster)

            for rid in self.other_rover_ids:
                if not self.other_rover_positions[rid]["have_odom"]:
                    continue

                dx = self.other_rover_positions[rid]["x"] - self.px
                dy = self.other_rover_positions[rid]["y"] - self.py
                expected_range = np.hypot(dx, dy)

                if expected_range > self.R_VIS:
                    continue

                expected_bearing = math.atan2(dy, dx) - self.theta
                expected_bearing = math.atan2(
                    math.sin(expected_bearing),
                    math.cos(expected_bearing)
                )
                range_error = expected_range - measured_range
                bearing_error = math.atan2(
                    math.sin(expected_bearing - measured_bearing),
                    math.cos(expected_bearing - measured_bearing)
                )
                cost = range_error ** 2 + 0.5 * (expected_range * bearing_error) ** 2
                candidate_pairs.append((cost, rid, cluster_index))

        candidate_pairs.sort(key=lambda item: item[0])
        matched_rovers = set()
        matched_clusters = set()
        MAX_MATCH_COST = 2.0

        for cost, rid, cluster_index in candidate_pairs:
            if cost > MAX_MATCH_COST:
                break
            if rid in matched_rovers or cluster_index in matched_clusters:
                continue
            matched_rovers.add(rid)
            matched_clusters.add(cluster_index)
            self.visible_neighbors[rid] = True

        if self.visible_neighbors:
            self.event_logger.info(
                f"Visible neighbors detected: {list(self.visible_neighbors.keys())}"
            )

    def update_neighbor_subscriptions(self, any_cluster_in_range):
        for rid in self.other_rover_ids:
            already_subscribed = self.rover_odom_subs[rid] is not None

            if any_cluster_in_range and not already_subscribed:
                callback = (
                    self.rover_2_odom_callback if rid == 2
                    else self.rover_3_odom_callback
                )
                self.rover_odom_subs[rid] = self.create_subscription(
                    Odometry,
                    f'/rover_{rid}/odom',
                    callback,
                    10
                )
                self.get_logger().info(
                    f'Cluster detected in range -- subscribing to rover {rid} odom.'
                )
            elif not any_cluster_in_range and already_subscribed:
                self.destroy_subscription(self.rover_odom_subs[rid])
                self.rover_odom_subs[rid] = None
                self.other_rover_positions[rid]['have_odom'] = False
                self.get_logger().info(
                    f'No clusters in range -- unsubscribing from rover {rid} odom.'
                )

    # ============================================================
    #  MULTI‑DISC CBF‑QP (NEW)
    # ============================================================
    def get_disc_centers(self, px, py, theta):
        c = math.cos(theta)
        s = math.sin(theta)
        centers = []
        for ox, oy in self.DISC_OFFSETS:
            wx = px + c * ox - s * oy
            wy = py + s * ox + c * oy
            centers.append((wx, wy))
        return centers

    def apply_cbf_qp(self, u1_x, u1_y):
        constraint_rows = []
        constraint_bounds = []

        for rid in self.other_rover_ids:
            other = self.other_rover_positions[rid]
            if not other['have_odom']:
                continue

            # Centre‑to‑centre distance (for trigger/release)
            dx = self.px - other['x']
            dy = self.py - other['y']
            distance = math.hypot(dx, dy)

            # Disc centres
            own_centers = self.get_disc_centers(self.px, self.py, self.theta)
            other_centers = self.get_disc_centers(other['x'], other['y'], other['theta'])

            min_h = float('inf')
            min_psi = float('inf')
            for k in range(self.N_DISCS):
                for l in range(self.N_DISCS):
                    rel_x = own_centers[k][0] - other_centers[l][0]
                    rel_y = own_centers[k][1] - other_centers[l][1]
                    dist_sq = rel_x*rel_x + rel_y*rel_y
                    h = dist_sq - self.DISC_D_SAFE**2
                    psi = 2.0*(rel_x*u1_x + rel_y*u1_y) + 0.5*self.GAMMA_CBF*h
                    if h < min_h:
                        min_h = h
                    if psi < min_psi:
                        min_psi = psi

            visible = self.visible_neighbors.get(rid, False)
            self._update_pair_mode(rid, distance, min_psi, visible)

            self.current_distance[rid] = distance
            self.current_barrier[rid] = min_h
            self.current_safety_link[rid] = int(self.rover_tracks[rid]['pair_active'])

            if self.rover_tracks[rid]['pair_active']:
                for k in range(self.N_DISCS):
                    for l in range(self.N_DISCS):
                        rel_x = own_centers[k][0] - other_centers[l][0]
                        rel_y = own_centers[k][1] - other_centers[l][1]
                        constraint_rows.append((-2.0 * rel_x, -2.0 * rel_y))
                        constraint_bounds.append(
                            0.5 * self.GAMMA_CBF * (
                                (rel_x*rel_x + rel_y*rel_y) - self.DISC_D_SAFE**2
                            )
                        )

        u_safe_x, u_safe_y = self.solve_projection_qp_2d(
            u1_x, u1_y, constraint_rows, constraint_bounds
        )
        return u_safe_x, u_safe_y

    def _update_pair_mode(self, rid, distance_ij, min_psi, visible):
        track = self.rover_tracks[rid]

        if not track['pair_active']:
            track['release_counter'] = 0
            if (visible
                    and distance_ij <= self.R_TRIGGER
                    and min_psi < -self.PSI_ACTIVATE_TOL):
                self.get_logger().info(f'Collision Check Activated for Pair 1 and {rid}')
                track['pair_active'] = True
                track['hold_counter'] = self.MIN_ACTIVE_STEPS
                track['release_counter'] = 0
                self.event_logger.info(
                    f"SAFETY FILTER TRIGGERED for rover {rid} "
                    f"(distance={distance_ij:.3f} m, min_psi={min_psi:.4f})"
                )
        elif track['hold_counter'] > 0:
            track['hold_counter'] -= 1
            track['release_counter'] = 0
        else:
            if distance_ij >= self.R_RELEASE and min_psi >= self.PSI_RELEASE:
                track['release_counter'] += 1
                if track['release_counter'] >= self.RELEASE_CONFIRM_STEPS:
                    track['pair_active'] = False
                    track['hold_counter'] = 0
                    track['release_counter'] = 0
                    self.event_logger.info(
                        f"SAFETY FILTER RELEASED for rover {rid} "
                        f"(distance={distance_ij:.3f} m, min_psi={min_psi:.4f})"
                    )
            else:
                track['release_counter'] = 0

    # ============================================================
    #  QP SOLVER (unchanged – still efficient for many constraints)
    # ============================================================
    def solve_projection_qp_2d(self, u_nom_x, u_nom_y, A_rows, b_values):
        if not A_rows:
            return u_nom_x, u_nom_y

        tolerance = 1e-10

        def feasible(u):
            for (a_x, a_y), b in zip(A_rows, b_values):
                if a_x * u[0] + a_y * u[1] > b + tolerance:
                    return False
            return True

        candidates = []
        if feasible((u_nom_x, u_nom_y)):
            candidates.append((u_nom_x, u_nom_y))

        for (a_x, a_y), b in zip(A_rows, b_values):
            denom = a_x ** 2 + a_y ** 2
            if denom <= tolerance:
                continue
            scale = (a_x * u_nom_x + a_y * u_nom_y - b) / denom
            candidate = (u_nom_x - scale * a_x, u_nom_y - scale * a_y)
            if feasible(candidate):
                candidates.append(candidate)

        num_constraints = len(A_rows)
        for ell in range(num_constraints):
            for m in range(ell + 1, num_constraints):
                a1_x, a1_y = A_rows[ell]
                a2_x, a2_y = A_rows[m]
                determinant = a1_x * a2_y - a2_x * a1_y
                if abs(determinant) <= tolerance:
                    continue
                b1 = b_values[ell]
                b2 = b_values[m]
                candidate_x = (b1 * a2_y - b2 * a1_y) / determinant
                candidate_y = (a1_x * b2 - a2_x * b1) / determinant
                candidate = (candidate_x, candidate_y)
                if feasible(candidate):
                    candidates.append(candidate)

        if not candidates:
            if feasible((0.0, 0.0)):
                self.get_logger().warn(
                    'CBF-QP projection candidates were empty; using zero control.'
                )
                return 0.0, 0.0
            self.get_logger().error(
                'Local CBF-QP is infeasible; holding zero command.'
            )
            return 0.0, 0.0

        best_candidate = min(
            candidates,
            key=lambda c: (c[0] - u_nom_x) ** 2 + (c[1] - u_nom_y) ** 2
        )
        return best_candidate

    # ============================================================
    #  CONTROL LOOP (unchanged except CBF call)
    # ============================================================
    def control_loop(self):
        if not self.have_odom:
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
            np.savez(
                f"rover_1_profile.npz",
                t=np.array(self.time_log),
                v_raw=np.array(self.v_raw_log),
                w_raw=np.array(self.w_raw_log),
                v_sat=np.array(self.v_sat_log),
                w_sat=np.array(self.w_sat_log)
            )
            np.savetxt(
                "rover_1_cbf_logs.csv",
                np.column_stack((
                    self.time_log,
                    self.u_nom_x_log,
                    self.u_nom_y_log,
                    self.u_qp_x_log,
                    self.u_qp_y_log,
                    self.distance_logs[2],
                    self.distance_logs[3],
                    self.barrier_logs[2],
                    self.barrier_logs[3],
                    self.safety_link_logs[2],
                    self.safety_link_logs[3]
                )),
                delimiter=",",
                header=(
                    "time,"
                    "u_nom_x,u_nom_y,"
                    "u_qp_x,u_qp_y,"
                    "distance_r2,distance_r3,"
                    "h_r2,h_r3,"
                    "link_r2,link_r3"
                ),
                comments=""
            )
            self.profile_saved = True
            return

        # ---- LQ nominal control ----
        x1_x = self.px - self.goal_x
        x1_y = self.py - self.goal_y
        u1_x = -2.0 * x1_x
        u1_y = -2.0 * x1_y
        u_nom_x, u_nom_y = u1_x, u1_y

        # ---- multi‑disc CBF‑QP safety filter ----
        u1_x, u1_y = self.apply_cbf_qp(u1_x, u1_y)

        # ---- feedback linearisation (MATLAB method) ----
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)

        speed_raw = cos_theta * u1_x + sin_theta * u1_y
        speed_raw = max(0.0, speed_raw)

        omega_raw = (-sin_theta * u1_x + cos_theta * u1_y) / self.l
        omega = max(min(omega_raw, self.omega_max), -self.omega_max)

        omega_ref = 5.0
        speed_min = 0.05
        speed_max = 0.3
        scale = np.exp(-abs(omega_raw) / omega_ref)
        speed = speed_min + (speed_max - speed_min) * scale
        speed = min(speed, speed_raw)

        # ---- logging ----
        self.time_log.append(self.get_clock().now().nanoseconds * 1e-9)
        self.v_raw_log.append(speed_raw)
        self.w_raw_log.append(omega_raw)
        self.v_sat_log.append(speed)
        self.w_sat_log.append(omega)
        self.u_nom_x_log.append(u_nom_x)
        self.u_nom_y_log.append(u_nom_y)
        self.u_qp_x_log.append(u1_x)
        self.u_qp_y_log.append(u1_y)
        self.distance_logs[2].append(self.current_distance[2])
        self.distance_logs[3].append(self.current_distance[3])
        self.barrier_logs[2].append(self.current_barrier[2])
        self.barrier_logs[3].append(self.current_barrier[3])
        self.safety_link_logs[2].append(self.current_safety_link[2])
        self.safety_link_logs[3].append(self.current_safety_link[3])

        # ---- publish commands ----
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

        leader_msg = Twist()
        leader_msg.linear.x = self.px
        leader_msg.linear.y = self.py
        leader_msg.angular.x = self.goal_x
        leader_msg.angular.y = self.goal_y
        self.leader_pub.publish(leader_msg)

        self.get_logger().info(
            f'p1=({self.px:.2f},{self.py:.2f}) '
            f'goal=({self.goal_x:.2f},{self.goal_y:.2f}) '
            f'v={speed:.3f} w={omega:.3f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = CbfController1()
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
