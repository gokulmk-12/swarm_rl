#!/usr/bin/python3

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from std_msgs.msg import String
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from ament_index_python.packages import get_package_share_directory
import os

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        self.laser_data = {}
        self.odom_data = {}
        self.global_obstacles = []
        self.obstacle_clusters = []  # Initialize obstacle_clusters
        self.obstacle_lifetimes = {}
        self.static_obstacles = []
        self.dbscan_eps = 0.5
        self.dbscan_min_samples = 5

        self.marker_publisher = self.create_publisher(MarkerArray, 'obstacle_markers', 10)
        self.static_obstacle_publisher = self.create_publisher(String, 'static_obstacles', 10)
        self.obstacle_coords_publisher = self.create_publisher(String, 'obstacle_coords', 10)  # New publisher for obstacle coordinates

        self.load_robot_configurations()
        self.create_timer(0.5, self.update_and_publish_markers)
        self.create_timer(2.0, self.decay_obstacles)
    
    def euler_from_quaternion(self, quat):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quat = [x, y, z, w]
        """
        x = quat[0]
        y = quat[1]
        z = quat[2]
        w = quat[3]

        sinr_cosp = 2 * (w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w*y - z*x)
        if sinp < -1:
            sinp = -1
        if sinp > 1:
            sinp = 1
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def load_robot_configurations(self):
        yaml_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'src', 'params.yaml')
        with open(yaml_file, 'r') as file:
            robots_config = yaml.safe_load(file)

        for robot_list in robots_config.values():
            if isinstance(robot_list, list):
                for robot in robot_list:
                    robot_name = robot['name']
                    self.create_subscription(LaserScan, f'/{robot_name}/scan', lambda msg, topic=robot_name: self.laser_callback(msg, topic), 10)
                    self.create_subscription(Odometry, f'/{robot_name}/odom', lambda msg, topic=robot_name: self.odom_callback(msg, topic), 10)

    def odom_callback(self, msg, topic):
        self.odom_data[topic] = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.euler_from_quaternion([msg.pose.pose.orientation.x,
                                            msg.pose.pose.orientation.y,
                                            msg.pose.pose.orientation.z,
                                            msg.pose.pose.orientation.w])[2]
        }

    def laser_callback(self, msg, topic):
        self.laser_data[topic] = {
            'ranges': torch.tensor(msg.ranges),
            'angle_min': msg.angle_min,
            'angle_increment': msg.angle_increment
        }
        self.process_data(topic)

    def process_data(self, robot_name):
        if robot_name not in self.laser_data or robot_name not in self.odom_data:
            return

        laser_data = self.laser_data[robot_name]
        odom_data = self.odom_data[robot_name]
        ranges = laser_data['ranges'][~torch.isinf(laser_data['ranges'])]
        angles = laser_data['angle_min'] + torch.arange(len(laser_data['ranges'])) * laser_data['angle_increment']
        angles = angles[~torch.isinf(laser_data['ranges'])]

        x_lidar = ranges * torch.cos(angles)
        y_lidar = ranges * torch.sin(angles)
        cos_theta = torch.cos(torch.tensor(odom_data['theta'], dtype=torch.float32))
        sin_theta = torch.sin(torch.tensor(odom_data['theta'], dtype=torch.float32))

        x_global = odom_data['x'] + x_lidar * cos_theta - y_lidar * sin_theta
        y_global = odom_data['y'] + x_lidar * sin_theta + y_lidar * cos_theta
        new_obstacles = [tuple(point) for point in np.vstack((x_global.cpu().numpy(), y_global.cpu().numpy())).T]

        for point in new_obstacles:
            if point not in self.obstacle_lifetimes:
                self.obstacle_lifetimes[point] = 0
                self.global_obstacles.append(point)

    def update_and_publish_markers(self):
        self.update_global_obstacles()

        # Log the number of distinct obstacles
        self.get_logger().info(f"Number of distinct obstacles: {len(self.obstacle_clusters)}")

        self.publish_markers()
        self.publish_static_obstacles()
        self.publish_obstacle_coords()  # Publish obstacle coordinates

    def decay_obstacles(self):
        to_remove = []
        for obstacle in self.global_obstacles:
            if obstacle in self.obstacle_lifetimes:
                self.obstacle_lifetimes[obstacle] += 1  # Incrementing by 1 every 2 seconds
                if self.obstacle_lifetimes[obstacle] > 5:  # Decay threshold set to 5 cycles (10 seconds)
                    if obstacle not in self.static_obstacles:
                        self.static_obstacles.append(obstacle)
                    to_remove.append(obstacle)
            else:
                to_remove.append(obstacle)  # Remove noise immediately

        for obstacle in to_remove:
            self.global_obstacles.remove(obstacle)
            if obstacle in self.obstacle_lifetimes:
                del self.obstacle_lifetimes[obstacle]

    def update_global_obstacles(self):
        all_obstacles = np.array(self.global_obstacles)
        if len(all_obstacles) > 0:
            all_obstacles = all_obstacles[~np.isnan(all_obstacles).any(axis=1)]
            clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(all_obstacles)  # Adjusted min_samples
            unique_labels = set(clustering.labels_)
            self.obstacle_clusters = []  # Reset obstacle_clusters
            merged_obstacles = []

            for label in unique_labels:
                if label != -1:
                    xy = all_obstacles[clustering.labels_ == label]
                    distances = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
                    total_length = np.sum(distances)
                    num_segments = int(total_length // self.dbscan_eps) + (1 if total_length % self.dbscan_eps > 0 else 0)
                    segment_lengths = np.cumsum(distances)

                    for i in range(num_segments):
                        start_index = 0 if i == 0 else np.searchsorted(segment_lengths, i * self.dbscan_eps)
                        end_index = np.searchsorted(segment_lengths, (i + 1) * self.dbscan_eps)
                        segment = xy[start_index:end_index] if start_index < end_index else []
                        if len(segment) > 0:
                            segment_center = np.mean(segment, axis=0)
                            merged_obstacles.append(tuple(segment_center))

                    self.obstacle_clusters.append(xy)

            self.global_obstacles = merged_obstacles

    def publish_markers(self):
        marker_array = MarkerArray()
        for i, (x, y) in enumerate(self.global_obstacles):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.lifetime = Duration(sec=0, nanosec=0)
            marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)

    def publish_static_obstacles(self):
        static_obstacle_data = {"count": len(self.static_obstacles), "obstacles": self.static_obstacles}
        static_obstacle_msg = String()
        static_obstacle_msg.data = str(static_obstacle_data)
        self.static_obstacle_publisher.publish(static_obstacle_msg)

    def publish_obstacle_coords(self):  # New method to publish obstacle coordinates
        obstacle_coords = [{x,y} for (x, y) in self.global_obstacles]
        obstacle_coords_msg = String()
        obstacle_coords_msg.data = str(obstacle_coords)
        self.obstacle_coords_publisher.publish(obstacle_coords_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
