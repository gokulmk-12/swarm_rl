import os
import sys
import math
import copy
import yaml
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from custom_msgs.srv import DrlStep, Goal, RingGoal
from geometry_msgs.msg import Pose, Twist
from rclpy.qos import QoSProfile, qos_profile_sensor_data, ReliabilityPolicy

# If small environment, use this below line
# params_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "params.yaml")
# If large environment, use this below line, and comment the previous
params_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "params_large.yaml")

with open(params_file, 'r') as file:
    data = yaml.safe_load(file)

ARENA_LENGTH            = 10
ARENA_WIDTH             = 10
LIDAR_DISTANCE_CAP      = data.get('LIDAR_DISTANCE_CAP', None)
MAX_GOAL_DISTANCE       = math.sqrt(ARENA_LENGTH**2 + ARENA_WIDTH**2)
NUM_SCAN_SAMPLES        = data.get('NUM_SCAN_SAMPLES', None)
THRESHOLD_COLLISION     = data.get('THRESHOLD_COLLISION', None)
THRESHOLD_GOAL          = data.get('THRESHOLD_GOAL', None)

MAX_NUMBER_OBSTACLES    = data.get('MAX_NUMBER_OBSTACLES', None)
OBSTACLE_RADIUS         = data.get('OBSTACLE_RADIUS', None)

UNKNOWN                 = data.get('UNKNOWN', None)
SUCCESS                 = data.get('SUCCESS', None)
COLLISION_WALL          = data.get('COLLISION_WALL', None)
COLLISION_OBSTACLE      = data.get('COLLISION_OBSTACLE', None)
TIMEOUT                 = data.get('TIMEOUT', None)
TUMBLE                  = data.get('TUMBLE', None)

LINEAR                  = data.get('LINEAR', None)
ANGULAR                 = data.get('ANGULAR', None)

SPEED_LINEAR_MAX        = data.get('SPEED_LINEAR_MAX', None)
SPEED_ANGULAR_MAX       = data.get('SPEED_ANGULAR_MAX', None)

EPISODE_TIMEOUT_SECONDS = data.get('EPISODE_TIMEOUT_SECONDS', None)
MAX_EPISODE_DURATION    = data.get('MAX_EPISODE_DURATION', None)

class RLEnv(Node):
    def __init__(self, namespace=""):
        node_name = f"{namespace}_rl_env" if namespace else "rl_env"
        super().__init__(node_name)

        prefix = f"{namespace}/" if namespace else ""
        self.scan_topic = f"{prefix}scan"
        self.cmd_topic = f"{prefix}cmd_vel"
        self.odom_topic = f"{prefix}odom"
        self.goal_topic = f"{prefix}goal_pose"
        self.namespace = namespace

        self.task_succeed_service = f"{prefix}task_succeed"
        self.task_fail_service = f"{prefix}task_fail"
        self.step_comm_service = f"{prefix}step_comm"
        self.goal_comm_service = f"{prefix}goal_comm"

        self.robot_x, self.robot_y = 0.0, 0.0
        self.robot_x_prev, self.robot_y_prev = 0.0, 0.0
        self.robot_heading = 0.0
        self.total_distance = 0.0
        self.robot_tilt = 0.0
        self.goal_x, self.goal_y = 0.0, 0.0 

        self.goal_angle = 0.0
        self.episode_timeout = EPISODE_TIMEOUT_SECONDS
        self.goal_distance = MAX_GOAL_DISTANCE
        self.obstacle_distance = LIDAR_DISTANCE_CAP
        self.obstacle_distances = [np.inf] * MAX_NUMBER_OBSTACLES
        self.scan_ranges = [LIDAR_DISTANCE_CAP] * NUM_SCAN_SAMPLES

        self.succeed = UNKNOWN
        self.episode_deadline = np.inf
        self.reset_deadline = False
        self.done = False
        self.local_step = 0
        self.time_sec = 0
        self.new_goal = False

        self.goal_dist_initial = MAX_GOAL_DISTANCE
        self.difficulty_radius = 1
        self.clock_msgs_skipped = 0
        
        qos = QoSProfile(depth=10)
        qos_clock = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        ### Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_topic, qos)

        ### Subscriber
        self.goal_sub = self.create_subscription(Pose, self.goal_topic, self.goal_pose_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_profile=qos_profile_sensor_data)
        self.clock_sub = self.create_subscription(Clock, '/clock', self.clock_callback, qos_profile=qos_clock)
        self.obstacle_odom_sub = self.create_subscription(Odometry, 'obstacle/odom', self.obstacle_odom_callback, qos)

        ### Clients
        self.task_succeed_client = self.create_client(RingGoal, self.task_succeed_service)
        self.task_fail_client = self.create_client(RingGoal, self.task_fail_service)

        ### Service
        self.step_comm_server = self.create_service(DrlStep, self.step_comm_service, self.step_comm_callback)
        self.goal_comm_server = self.create_service(Goal, self.goal_comm_service, self.goal_comm_callback)
    
    def euler_from_quaternion(self, quat):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quat = [x, y, z, w]
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

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
    
    def goal_pose_callback(self, msg):
        self.goal_x = msg.position.x
        self.goal_y = msg.position.y
        self.new_goal = True
        print(f"Recieved Goal for {self.namespace} x: {self.goal_x} y: {self.goal_y}")
    
    def goal_comm_callback(self, request, response):
        response.new_goal = self.new_goal
        return response
    
    def obstacle_odom_callback(self, msg):
        if 'obstacle' in msg.child_frame_id:
            robot_pos = msg.pose.pose.position
            obstacle_id = int(msg.child_frame_id[-1]) - 4 #3
            diff_x = self.robot_x - robot_pos.x
            diff_y = self.robot_y - robot_pos.y
            self.obstacle_distances[obstacle_id] = math.sqrt(diff_y**2 + diff_x**2)
        else:
            print("ERROR: received odom was not from obstacle!")
    
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        _, _, self.robot_heading = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.robot_tilt = msg.pose.pose.orientation.y

        # calculate traveled distance for logging
        if self.local_step % 32 == 0:
            self.total_distance += math.sqrt(
                (self.robot_x_prev - self.robot_x)**2 +
                (self.robot_y_prev - self.robot_y)**2)
            self.robot_x_prev = self.robot_x
            self.robot_y_prev = self.robot_y

        diff_y = self.goal_y - self.robot_y
        diff_x = self.goal_x - self.robot_x
        distance_to_goal = math.sqrt(diff_x**2 + diff_y**2)
        heading_to_goal = math.atan2(diff_y, diff_x)
        goal_angle = heading_to_goal - self.robot_heading

        while goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        while goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = distance_to_goal
        self.goal_angle = goal_angle
    
    def scan_callback(self, msg):
        if len(msg.ranges) != NUM_SCAN_SAMPLES:
            print(f"more or less scans than expected! check model.sdf, got: {len(msg.ranges)}, expected: {NUM_SCAN_SAMPLES}")
        # normalize laser values
        self.obstacle_distance = 1
        for i in range(NUM_SCAN_SAMPLES):
                self.scan_ranges[i] = np.clip(float(msg.ranges[i]) / LIDAR_DISTANCE_CAP, 0, 1)
                if self.scan_ranges[i] < self.obstacle_distance:
                    self.obstacle_distance = self.scan_ranges[i]
        self.obstacle_distance *= LIDAR_DISTANCE_CAP

    def clock_callback(self, msg):
        self.time_sec = msg.clock.sec
        if not self.reset_deadline:
            return
        self.clock_msgs_skipped += 1
        if self.clock_msgs_skipped <= 10: # Wait a few message for simulation to reset clock
            return
        episode_time = min(self.episode_timeout, MAX_EPISODE_DURATION)
        self.episode_deadline = self.time_sec + episode_time
        self.reset_deadline = False
        self.clock_msgs_skipped = 0
    
    def stop_reset_robot(self, success):
        self.cmd_vel_pub.publish(Twist()) # stop robot
        self.episode_deadline = np.inf
        self.done = True
        req = RingGoal.Request()
        req.robot_pose_x = self.robot_x
        req.robot_pose_y = self.robot_y
        req.radius = np.clip(self.difficulty_radius, 0.5, 4)
        if success:
            self.difficulty_radius *= 1.01
            while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('success service not available, waiting again...')
            self.task_succeed_client.call_async(req)
        else:
            self.difficulty_radius *= 0.99
            while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('fail service not available, waiting again...')
            self.task_fail_client.call_async(req)
    
    def initalize_episode(self, response):
        self.initial_distance_to_goal = self.goal_distance
        response.state = self.get_state(0, 0)
        response.reward = 0.0
        response.done = False
        response.distance_traveled = 0.0
        return response

    def get_state(self, action_linear_previous, action_angular_previous):
        state = copy.deepcopy(self.scan_ranges)                                             # range: [ 0, 1]
        state.append(float(np.clip((self.goal_distance / MAX_GOAL_DISTANCE), 0, 1)))     # range: [ 0, 1]
        state.append(float(self.goal_angle) / math.pi)                                      # range: [-1, 1]
        state.append(float(action_linear_previous))                                         # range: [-1, 1]
        state.append(float(action_angular_previous))                                        # range: [-1, 1]
        self.local_step += 1

        if self.local_step <= 30:
            return state
        # Success
        safe = 0.16 if self.namespace == "robot4" else 0
        if self.goal_distance < THRESHOLD_GOAL + safe:
            self.succeed = SUCCESS
            self.episode_timeout += EPISODE_TIMEOUT_SECONDS
            self.local_step = 0
        # Collision
        elif self.obstacle_distance < THRESHOLD_COLLISION:
            dynamic_collision = False
            for obstacle_distance in self.obstacle_distances:
                if obstacle_distance < (THRESHOLD_COLLISION + OBSTACLE_RADIUS + 0.05):
                    dynamic_collision = True
            if dynamic_collision:
                self.succeed = COLLISION_OBSTACLE
            else:
                self.succeed = COLLISION_WALL
        # Timeout
        elif self.time_sec >= self.episode_deadline:
            self.succeed = TIMEOUT
        # Tumble
        elif self.robot_tilt > 0.06 or self.robot_tilt < -0.06:
            self.succeed = TUMBLE

        if self.succeed is not UNKNOWN:
            self.stop_reset_robot(self.succeed == SUCCESS)
        return state
    
    def step_comm_callback(self, request, response):
        if len(request.action) == 0:
            return self.initalize_episode(response)

        action_linear = (request.action[LINEAR] + 1) / 2 * SPEED_LINEAR_MAX
        action_angular = request.action[ANGULAR] * SPEED_ANGULAR_MAX

        # Publish action cmd
        twist = Twist()
        twist.linear.x = action_linear
        twist.angular.z = action_angular
        self.cmd_vel_pub.publish(twist)

        # Prepare repsonse
        response.state = self.get_state(request.previous_action[LINEAR], request.previous_action[ANGULAR])
        response.reward = self.get_reward(self.succeed, action_linear, action_angular, self.goal_distance,
                                            self.goal_angle, self.obstacle_distance)
        response.done = self.done
        response.success = self.succeed
        response.distance_traveled = 0.0
        if self.done:
            response.distance_traveled = self.total_distance
            # Reset variables
            self.succeed = UNKNOWN
            self.total_distance = 0.0
            self.local_step = 0
            self.done = False
            self.reset_deadline = True
        if self.local_step % 200 == 0:
            print(f"Rtot: {response.reward:<8.2f}GD: {self.goal_distance:<8.2f}GA: {math.degrees(self.goal_angle):.1f}°\t", end='')
            print(f"MinD: {self.obstacle_distance:<8.2f}Alin: {request.action[LINEAR]:<7.1f}Aturn: {request.action[ANGULAR]:<7.1f}")
        return response
    
    def get_reward(self, succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # [-3.14, 0]
        r_yaw = -2 * abs(goal_angle) * (1 / (goal_dist + 0.1))

        # [-4, 0]
        r_vangular = -1 * (action_angular**2)

        # [-1, 1]
        r_distance = 1 * (2 * self.initial_distance_to_goal) / (self.initial_distance_to_goal + goal_dist) - 1
        if goal_dist < 0.3:
            r_distance += 1.0 * (0.3 - goal_dist)

        # [-20, 0]
        if min_obstacle_dist < 0.22:
            r_obstacle = -30
        else:
            r_obstacle = 0

        # [-2 * (2.2^2), 0]
        r_vlinear = -1 * (((0.22 - action_linear) * 10) ** 2)

        reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular - 1

        if succeed == SUCCESS:
            reward += 3000
        elif succeed == COLLISION_OBSTACLE:
            reward -= 2000
        elif succeed == COLLISION_WALL:
            reward -= 2500

        return float(reward)

def main(args=sys.argv[1:]):
    rclpy.init(args=args)

    namespace = args[0]

    rl_env = RLEnv(namespace=namespace)
    rclpy.spin(rl_env)
    rclpy.shutdown()

if __name__ == "__main__":
    main()