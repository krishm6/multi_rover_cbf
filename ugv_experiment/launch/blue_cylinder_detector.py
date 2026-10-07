#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_geometry_msgs

import cv2
import numpy as np

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import (
    PoseStamped,
    PointStamped,
    TransformStamped
)

from tf2_ros import (
    Buffer,
    TransformListener
)

from tf2_ros.static_transform_broadcaster import (
    StaticTransformBroadcaster
)

from tf2_geometry_msgs import do_transform_point



class BlueCylinderDetector(Node):

    def __init__(self):

        super().__init__('blue_cylinder_detector')

        # ==========================
        # Camera Intrinsics
        # ==========================

        self.fx = 565.6008952774197
        self.fy = 565.6008952774197

        self.cx = 320.5
        self.cy = 240.5

        # ==========================
        # CvBridge
        # ==========================

        self.bridge = CvBridge()

        # ==========================
        # Image Storage
        # ==========================

        self.rgb_frame = None
        self.depth_frame = None

        # ==========================
        # TF
        # ==========================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ==========================
        # Static TF Broadcaster
        # ==========================

        self.static_broadcaster = (
            StaticTransformBroadcaster(self)
        )

        self.publish_static_transforms()

        # ==========================
        # Subscribers
        # ==========================

        self.rgb_sub = self.create_subscription(
            Image,
            '/rover_1/camera/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/rover_1/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        # ==========================
        # Goal Publisher
        # ==========================

        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        # ==========================
        # Timer
        # ==========================

        self.timer = self.create_timer(
            0.1,
            self.process_images
        )

        self.get_logger().info(
            "Blue Cylinder Detector Started"
        )

    # ====================================
    # Static TFs
    # ====================================

    def publish_static_transforms(self):

        transforms = []

        # --------------------------
        # world -> rover_1/odom
        # --------------------------

        t1 = TransformStamped()

        t1.header.stamp = (
            self.get_clock().now().to_msg()
        )

        t1.header.frame_id = "world"
        t1.child_frame_id = "rover_1/odom"

        t1.transform.translation.x = 0.0
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = 0.0

        t1.transform.rotation.x = 0.0
        t1.transform.rotation.y = 0.0
        t1.transform.rotation.z = 0.0
        t1.transform.rotation.w = 1.0

        transforms.append(t1)

        # --------------------------
        # world -> rover_2/odom
        # --------------------------

        t2 = TransformStamped()

        t2.header.stamp = (
            self.get_clock().now().to_msg()
        )

        t2.header.frame_id = "world"
        t2.child_frame_id = "rover_2/odom"

        t2.transform.translation.x = 3.0
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.0

        t2.transform.rotation.x = 0.0
        t2.transform.rotation.y = 0.0
        t2.transform.rotation.z = 0.0
        t2.transform.rotation.w = 1.0

        transforms.append(t2)

        # --------------------------
        # world -> rover_3/odom
        # --------------------------

        t3 = TransformStamped()

        t3.header.stamp = (
            self.get_clock().now().to_msg()
        )

        t3.header.frame_id = "world"
        t3.child_frame_id = "rover_3/odom"

        t3.transform.translation.x = 0.0
        t3.transform.translation.y = 4.0
        t3.transform.translation.z = 0.0

        t3.transform.rotation.x = 0.0
        t3.transform.rotation.y = 0.0
        t3.transform.rotation.z = 0.0
        t3.transform.rotation.w = 1.0

        transforms.append(t3)

        self.static_broadcaster.sendTransform(
            transforms
        )

    # ====================================
    # RGB Callback
    # ====================================

    def rgb_callback(self, msg):
        try:
            self.rgb_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
            self.get_logger().info("RGB converted")

        except Exception as e:
            self.get_logger().error(
                f"RGB error: {e}"
            )

    # ====================================
    # Depth Callback
    # ====================================

    def depth_callback(self, msg):
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )

            self.get_logger().info("Depth converted")
            
        except Exception as e:
            self.get_logger().error(
                f"Depth error: {e}"
            )
    # ====================================
    # Main Processing Loop
    # ====================================

    def process_images(self):
        self.get_logger().info(
            f"clock={self.get_clock().now().nanoseconds}"
        )
        print(self.get_clock().now().nanoseconds)
        if self.rgb_frame is None:
            self.get_logger().info("Waiting for RGB")
            return

        if self.depth_frame is None:
            self.get_logger().info("Waiting for Depth")
            return

        frame = self.rgb_frame.copy()

        # ==========================
        # BGR -> HSV
        # ==========================

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        # ==========================
        # Blue Mask
        # ==========================

        lower_blue = np.array(
            [100, 100, 50],
            dtype=np.uint8
        )

        upper_blue = np.array(
            [140, 255, 255],
            dtype=np.uint8
        )

        mask = cv2.inRange(
            hsv,
            lower_blue,
            upper_blue
        )
        cv2.imshow("Blue Mask", mask)
        cv2.waitKey(1)

        # ==========================
        # Find Contours
        # ==========================

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return

        # ==========================
        # Largest Contour
        # ==========================

        c = max(
            contours,
            key=cv2.contourArea
        )
        x, y, w, h = cv2.boundingRect(c)
        
        area = cv2.contourArea(c)
        hsv_display = cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR
        )

        cv2.rectangle(
            hsv_display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "HSV Detection",
            hsv_display
        )
        if area < 0.000005:
            return

        # ==========================
        # Moments
        # ==========================

        M = cv2.moments(c)

        if M["m00"] == 0:
            return

        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])

        # ==========================
        # Bounds Check
        # ==========================

        h, w = self.depth_frame.shape

        if u < 0 or u >= w:
            return

        if v < 0 or v >= h:
            return

        # ==========================
        # Depth Lookup
        # ==========================

        depth = float(
            self.depth_frame[v, u]
        )

        if np.isnan(depth):
            return

        if np.isinf(depth):
            return

        if depth <= 0.0:
            return

        # ==========================
        # Pixel -> Camera Coordinates
        # ==========================

        X = (
            (u - self.cx)
            * depth
            / self.fx
        )

        Y = (
            (v - self.cy)
            * depth
            / self.fy
        )

        Z = depth

        # ==========================
        # Camera Point
        # ==========================

        point_camera = PointStamped()

        point_camera.header.stamp = (self.get_clock().now()-Duration(seconds=0.01)).to_msg()
        point_camera.header.frame_id = "rover_1/3d_camera_link"

        point_camera.point.x = float(X)
        point_camera.point.y = float(Y)
        point_camera.point.z = float(Z)

        # ==========================
        # Camera -> World Transform
        # ==========================

        try:
            self.get_logger().info("USING LATEST TF LOOKUP")
            print('USING LATEST TF LOOKUP')
            transform = self.tf_buffer.lookup_transform(
                "world",
                point_camera.header.frame_id,
                rclpy.time.Time(seconds=0)
            )

            point_world = tf2_geometry_msgs.do_transform_point(
                point_camera,
                transform
            )

        except Exception as e:
            self.get_logger().warn(
                f"TF Error: {str(e)}"
            )
            return

        # ==========================
        # Goal Pose
        # ==========================

        goal = PoseStamped()

        goal.header.stamp = (
            self.get_clock().now().to_msg()
        )

        goal.header.frame_id = "world"

        goal.pose.position.x = (
            point_world.point.x
        )

        goal.pose.position.y = (
            point_world.point.y
        )

        goal.pose.position.z = (
            point_world.point.z
        )

        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = 0.0
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)

        # ==========================
        # Visualization
        # ==========================

        cv2.circle(
            frame,
            (u, v),
            5,
            (0, 0, 255),
            -1
        )

        cv2.imshow(
            "Blue Cylinder Detection",
            frame
        )

        cv2.imshow(
            "Blue Mask",
            mask
        )

        cv2.waitKey(1)

        self.get_logger().info(
            f"World Goal: "
            f"{point_world.point.x:.2f}, "
            f"{point_world.point.y:.2f}, "
            f"{point_world.point.z:.2f}"
        )
# ====================================
# Main
# ====================================

def main(args=None):

    rclpy.init(args=args)

    node = BlueCylinderDetector()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
