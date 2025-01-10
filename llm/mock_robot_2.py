import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from path_interface.msg import Paths, Obstacles
from geometry_msgs.msg import Point
import random
import time


class MockRobot2(Node):
    def __init__(self):
        super().__init__('mock_robot_2')
        self.odom_publisher = self.create_publisher(Odometry, '/robot_4/odom', 10)
        self.obstacles_publisher = self.create_publisher(Obstacles, '/robot_3/obstacles', 10)
        self.paths_publisher = self.create_publisher(Paths, '/robot_4/paths', 10)

    def publish_odom(self):
        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = random.uniform(0, 10)
        odom_msg.pose.pose.position.y = random.uniform(0, 10)
        odom_msg.pose.pose.position.z = 0.0
        self.odom_publisher.publish(odom_msg)
        self.get_logger().info(f'Published odom for robot_2: {odom_msg.pose.pose.position.x}, {odom_msg.pose.pose.position.y}')

    def publish_obstacles(self):
        obs_msg = Obstacles()
        for _ in range(random.randint(1, 5)):
            point = Point()
            point.x = random.uniform(0, 10)
            point.y = random.uniform(0, 10)
            point.z = 0.0
            obs_msg.points.append(point)
        self.obstacles_publisher.publish(obs_msg)
        self.get_logger().info(f'Published obstacles for robot_2')

    def publish_paths(self):
        paths_msg = Paths()
        paths_msg.target_id = random.randint(1, 5)
        for _ in range(random.randint(2, 5)):
            point = Point()
            point.x = random.uniform(0, 10)
            point.y = random.uniform(0, 10)
            point.z = 0.0
            paths_msg.points.append(point)
        self.paths_publisher.publish(paths_msg)
        self.get_logger().info(f'Published paths for robot_2 with target {paths_msg.target_id}')


def main(args=None):
    rclpy.init(args=args)
    node = MockRobot2()

    try:
        while rclpy.ok():
            node.publish_odom()
            node.publish_obstacles()
            node.publish_paths()
            time.sleep(1)  # Adjust the delay if needed
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
