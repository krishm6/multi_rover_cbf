#!/usr/bin/env python3
"""
cbf_trajectory_plotter.py

ROS node that waits for all rovers to reach their goals, then generates:

1. CBF-QP theoretical (dashed) vs Gazebo (solid) trajectories.
2. Inter-rover distances from synchronized Gazebo odometry logs.
3. Trigger and release distance reference lines on the distance plot.

If interrupted with Ctrl+C, it plots whatever data is available.

Required contents in each rover_i_cbf_odom_log.npz:
    t or time : timestamp array [s]
    x         : x-position array [m]
    y         : y-position array [m]

Set R_TRIGGER and R_RELEASE below to the same footprint/safety parameters
used by your CBF tracker.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


# ==============================================================
# CBF TRACKER FOOTPRINT / SAFETY PARAMETERS
# Put the SAME values used in your CBF tracker here.
# ==============================================================

D_SAFE    = 0.80
R_TRIGGER = 2.00
R_RELEASE = 2.20


class CbfTrajectoryPlotter(Node):
    def __init__(self):
        super().__init__('cbf_trajectory_plotter')

        self.r1_done = False
        self.r2_done = False
        self.r3_done = False

        self.finished = False
        self.plot_triggered = False

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

        self.get_logger().info(
            'CBF trajectory plotter waiting for all rovers to reach goals...'
        )

    def goal1_callback(self, msg):
        if msg.data:
            self.r1_done = True
            self.get_logger().info('Rover 1 reached goal')
            self.check_finished()

    def goal2_callback(self, msg):
        if msg.data:
            self.r2_done = True
            self.get_logger().info('Rover 2 reached goal')
            self.check_finished()

    def goal3_callback(self, msg):
        if msg.data:
            self.r3_done = True
            self.get_logger().info('Rover 3 reached goal')
            self.check_finished()

    def check_finished(self):
        if self.finished:
            return

        if self.r1_done and self.r2_done and self.r3_done:
            self.finished = True

            self.get_logger().info(
                'All rovers reached goals. Generating CBF plots...'
            )

            self.plot_trajectories()
            self.plot_inter_rover_distances()

            self.destroy_node()
            rclpy.shutdown()

    def plot_trajectories(self):
        """
        Plot theoretical CBF-QP tracked trajectories and Gazebo trajectories.
        Can also be called after Ctrl+C with partial data.
        """

        if self.plot_triggered:
            return

        self.plot_triggered = True

        colors = ['tab:blue', 'tab:orange', 'tab:green']
        rover_names = ['R1', 'R2', 'R3']

        fig, ax = plt.subplots(figsize=(15, 12))

        all_x = []
        all_y = []

        for i in range(1, 4):
            color = colors[i - 1]
            rover_name = rover_names[i - 1]

            # ==========================================================
            # THEORETICAL CBF-QP TRAJECTORY
            # ==========================================================
            waypoint_file = f'rover_{i}_cbf_tracked_waypoints.npz'

            if os.path.exists(waypoint_file):
                data = np.load(waypoint_file)

                x_theory = np.asarray(data['x'], dtype=float)
                y_theory = np.asarray(data['y'], dtype=float)

                if len(x_theory) > 0 and len(y_theory) > 0:
                    all_x.extend(x_theory)
                    all_y.extend(y_theory)

                    ax.plot(
                        x_theory,
                        y_theory,
                        color=color,
                        linestyle='--',
                        linewidth=2.8,
                        label=f'{rover_name} CBF Theory',
                        zorder=3
                    )

                    # Hollow square = theoretical initial pose
                    ax.scatter(
                        x_theory[0],
                        y_theory[0],
                        s=160,
                        facecolors='none',
                        edgecolors=color,
                        linewidths=2.8,
                        marker='s',
                        zorder=8
                    )

                    # Hollow circle = theoretical final pose
                    ax.scatter(
                        x_theory[-1],
                        y_theory[-1],
                        s=180,
                        facecolors='white',
                        edgecolors=color,
                        linewidths=2.8,
                        marker='o',
                        zorder=8
                    )

                else:
                    self.get_logger().warn(
                        f'{waypoint_file} is empty – skipping theory for Rover {i}'
                    )

            else:
                self.get_logger().warn(
                    f'{waypoint_file} not found – skipping theory for Rover {i}'
                )

            # ==========================================================
            # GAZEBO ODOMETRY TRAJECTORY
            # ==========================================================
            odom_file = f'rover_{i}_cbf_odom_log.npz'

            if os.path.exists(odom_file):
                data = np.load(odom_file)

                if 'x' not in data or 'y' not in data:
                    self.get_logger().warn(
                        f'{odom_file} does not contain x/y arrays'
                    )
                    continue

                x_gazebo = np.asarray(data['x'], dtype=float)
                y_gazebo = np.asarray(data['y'], dtype=float)

                if len(x_gazebo) > 0 and len(y_gazebo) > 0:
                    all_x.extend(x_gazebo)
                    all_y.extend(y_gazebo)

                    ax.plot(
                        x_gazebo,
                        y_gazebo,
                        color=color,
                        linestyle='-',
                        linewidth=2.3,
                        label=f'{rover_name} Gazebo',
                        zorder=4
                    )

                    # Filled square = Gazebo initial pose
                    ax.scatter(
                        x_gazebo[0],
                        y_gazebo[0],
                        s=140,
                        color=color,
                        marker='s',
                        edgecolors='black',
                        linewidths=1.0,
                        zorder=9
                    )

                    # Filled X = Gazebo final pose
                    ax.scatter(
                        x_gazebo[-1],
                        y_gazebo[-1],
                        s=210,
                        color=color,
                        marker='X',
                        edgecolors='black',
                        linewidths=1.0,
                        zorder=10
                    )

                else:
                    self.get_logger().warn(
                        f'{odom_file} is empty – skipping Gazebo for Rover {i}'
                    )

            else:
                self.get_logger().warn(
                    f'{odom_file} not found – run CBF tracker first'
                )

        ax.set_xlabel('X [m]', fontsize=16)
        ax.set_ylabel('Y [m]', fontsize=16)

        ax.set_title(
            'CBF-QP Theoretical vs Gazebo Trajectories',
            fontsize=20,
            fontweight='bold'
        )

        ax.tick_params(axis='both', labelsize=14)
        ax.set_aspect('equal', adjustable='box')

        # Major grid spacing = 1 m
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))

        # Minor grid spacing = 0.25 m
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))

        ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.55)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.35)

        if all_x and all_y:
            margin = 0.5
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        # ==============================================================
        # LEGEND 1: TRAJECTORIES, TOP-LEFT
        # ==============================================================

        trajectory_legend = ax.legend(
            fontsize=16,
            title_fontsize=18,
            loc='upper left',
            bbox_to_anchor=(0.02, 1.00),
            borderaxespad=0.0,
            title='Trajectories',
            framealpha=0.95
        )

        ax.add_artist(trajectory_legend)

        # ==============================================================
        # LEGEND 2: INITIAL / FINAL POSE MARKERS, BELOW IT
        # ==============================================================

        marker_handles = [
            Line2D(
                [0], [0],
                marker='s',
                color='black',
                markerfacecolor='none',
                markersize=11,
                linewidth=0,
                label='Theory initial pose'
            ),
            Line2D(
                [0], [0],
                marker='s',
                color='black',
                markerfacecolor='black',
                markersize=10,
                linewidth=0,
                label='Gazebo initial pose'
            ),
            Line2D(
                [0], [0],
                marker='o',
                color='black',
                markerfacecolor='white',
                markersize=11,
                linewidth=0,
                label='Theory final pose'
            ),
            Line2D(
                [0], [0],
                marker='X',
                color='black',
                markerfacecolor='black',
                markersize=11,
                linewidth=0,
                label='Gazebo final pose'
            )
        ]

        marker_legend = ax.legend(
            handles=marker_handles,
            fontsize=16,
            title_fontsize=18,
            loc='lower right',
            bbox_to_anchor=(0.98, 0.02),
            borderaxespad=0.0,
            title='Pose markers',
            framealpha=0.95
        )

        ax.add_artist(marker_legend)

        plt.tight_layout()

        plt.savefig(
            'cbf_trajectory_comparison.jpg',
            dpi=600,
            bbox_inches='tight',
            format='jpeg',
            pil_kwargs={'quality': 95}
        )

        self.get_logger().info(
            'Trajectory plot saved as cbf_trajectory_comparison.jpg'
        )

        plt.show()

    def load_odom_log(self, filename):
        """
        Load one CBF odometry log and prepare it for interpolation.

        Removes invalid values, sorts time, and removes duplicate timestamps.
        """

        if not os.path.exists(filename):
            self.get_logger().error(f'{filename} not found')
            return None

        data = np.load(filename)

        if 't' in data:
            t = np.asarray(data['t'], dtype=float)
        elif 'time' in data:
            t = np.asarray(data['time'], dtype=float)
        else:
            self.get_logger().error(
                f'{filename} has no "t" or "time" array. '
                'Save timestamps in the odometry logger.'
            )
            return None

        if 'x' not in data or 'y' not in data:
            self.get_logger().error(
                f'{filename} must contain x and y arrays.'
            )
            return None

        x = np.asarray(data['x'], dtype=float)
        y = np.asarray(data['y'], dtype=float)

        if len(t) != len(x) or len(t) != len(y):
            self.get_logger().error(
                f'{filename}: t, x, and y arrays have different lengths.'
            )
            return None

        valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)

        t = t[valid]
        x = x[valid]
        y = y[valid]

        if len(t) < 2:
            self.get_logger().error(
                f'{filename}: insufficient valid samples.'
            )
            return None

        order = np.argsort(t)

        t = t[order]
        x = x[order]
        y = y[order]

        unique_t, unique_indices = np.unique(t, return_index=True)

        t = unique_t
        x = x[unique_indices]
        y = y[unique_indices]

        if len(t) < 2:
            self.get_logger().error(
                f'{filename}: insufficient unique timestamp samples.'
            )
            return None

        return {
            't': t,
            'x': x,
            'y': y
        }

    def plot_inter_rover_distances(self):
        """
        Synchronize the three odometry logs over their common time interval.

        common_start = max(first timestamp of each rover)
        common_end   = min(last timestamp of each rover)

        The common time axis uses the maximum number of valid samples found
        among the three rover logs inside that overlap interval.
        """

        log1 = self.load_odom_log('rover_1_cbf_odom_log.npz')
        log2 = self.load_odom_log('rover_2_cbf_odom_log.npz')
        log3 = self.load_odom_log('rover_3_cbf_odom_log.npz')

        if log1 is None or log2 is None or log3 is None:
            self.get_logger().error(
                'CBF inter-rover distance plot was not generated.'
            )
            return

        logs = [log1, log2, log3]

        common_start = max(log['t'][0] for log in logs)
        common_end = min(log['t'][-1] for log in logs)

        if common_end <= common_start:
            self.get_logger().error(
                'No common time interval exists across the CBF odometry logs.'
            )
            return

        samples_in_common_interval = []

        for log in logs:
            count = np.count_nonzero(
                (log['t'] >= common_start) &
                (log['t'] <= common_end)
            )
            samples_in_common_interval.append(count)

        max_common_samples = max(samples_in_common_interval)

        if max_common_samples < 2:
            self.get_logger().error(
                'Not enough synchronized odometry samples to compute distances.'
            )
            return

        # Shared time vector over the largest common valid interval
        t_common = np.linspace(
            common_start,
            common_end,
            max_common_samples
        )

        # Interpolate each rover onto the exact same time axis
        x1 = np.interp(t_common, log1['t'], log1['x'])
        y1 = np.interp(t_common, log1['t'], log1['y'])

        x2 = np.interp(t_common, log2['t'], log2['x'])
        y2 = np.interp(t_common, log2['t'], log2['y'])

        x3 = np.interp(t_common, log3['t'], log3['x'])
        y3 = np.interp(t_common, log3['t'], log3['y'])

        d_12 = np.hypot(x1 - x2, y1 - y2)
        d_13 = np.hypot(x1 - x3, y1 - y3)
        d_23 = np.hypot(x2 - x3, y2 - y3)

        # Plot time starts at zero for readability
        t_relative = t_common - t_common[0]

        fig, ax = plt.subplots(figsize=(15, 9))

        ax.plot(
            t_relative,
            d_12,
            linewidth=2.6,
            label='Rover 1 - Rover 2'
        )

        ax.plot(
            t_relative,
            d_13,
            linewidth=2.6,
            label='Rover 1 - Rover 3'
        )

        ax.plot(
            t_relative,
            d_23,
            linewidth=2.6,
            label='Rover 2 - Rover 3'
        )

        # Trigger and release thresholds from the CBF tracker
        # CBF safety, activation, and release distances
        ax.axhline(
            D_SAFE,
            linestyle=':',
            linewidth=2.5,
            label=f'Safety distance $D_{{safe}}$ = {D_SAFE:.2f} m'
        )

        ax.axhline(
            R_TRIGGER,
            linestyle='--',
            linewidth=2.3,
            label=f'CBF trigger distance = {R_TRIGGER:.2f} m'
        )

        ax.axhline(
            R_RELEASE,
            linestyle='-.',
            linewidth=2.3,
            label=f'CBF release distance = {R_RELEASE:.2f} m'
        )

        ax.set_xlabel('Common time [s]', fontsize=16)
        ax.set_ylabel('Inter-rover distance [m]', fontsize=16)

        ax.set_title(
            'CBF Inter-rover Distances from Gazebo Odometry',
            fontsize=20,
            fontweight='bold'
        )

        ax.tick_params(axis='both', labelsize=14)

        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))

        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

        ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.55)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.35)

        ax.set_xlim(t_relative[0], t_relative[-1])

        ax.legend(
            fontsize=16,
            title_fontsize=18,
            loc='best',
            framealpha=0.95
        )

        plt.tight_layout()

        plt.savefig(
            'cbf_inter_rover_distances.jpg',
            dpi=600,
            bbox_inches='tight',
            format='jpeg',
            pil_kwargs={'quality': 95}
        )

        self.get_logger().info(
            'Inter-rover distance plot saved as cbf_inter_rover_distances.jpg'
        )

        plt.show()


def main(args=None):
    rclpy.init(args=args)

    node = CbfTrajectoryPlotter()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info(
            'Keyboard interrupt received. Generating plots with available data...'
        )

        node.plot_trajectories()
        node.plot_inter_rover_distances()

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
