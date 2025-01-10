import os
import time
import json
import rclpy
import random
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
from custom_msgs.srv import RingGoal
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

class Gazebo(Node):
    def __init__(self, namespace=""):
        node_name = f"{namespace}_rl_gazebo" if namespace else "rl_gazebo"
        super().__init__(node_name)

        # Declare parameters for the node
        # If small environment, use this below line
        # json_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "llm", "datafinal.json")
        # If large environment, use this below line, and comment the previous
        json_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "llm", "datafinallarge.json")

        self.declare_parameter("mode", "training")
        self.declare_parameter("goal_file", json_file)
        self.declare_parameter("namespace", namespace)

        # Retrieve parameters
        self.mode = self.get_parameter("mode").value
        self.goal_file = self.get_parameter("goal_file").value
        self.namespace = self.get_parameter("namespace").value

        # Default goal positions
        self.goal_x, self.goal_y = -4.15, 8.86

        ### Publishers
        self.goal_pub = self.create_publisher(Pose, f'{self.namespace}/goal_pose', QoSProfile(depth=10))

        ### Clients
        self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')
        self.gazebo_pause = self.create_client(Empty, '/pause_physics')
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')

        ### Services
        self.task_succeed_server = self.create_service(RingGoal, f'{self.namespace}/task_succeed', self.task_succeed_callback)
        self.task_fail_server = self.create_service(RingGoal, f'{self.namespace}/task_fail', self.task_fail_callback)

        self.goal_pose_list = []
        if self.mode == "testing" and self.goal_file:
            self.load_goals_from_file()
        else:
            self.generate_default_goals()

        self.init_callback()

    def load_goals_from_file(self):
        """Load goals from a JSON file for testing mode."""
        if os.path.exists(self.goal_file):
            try:
                with open(self.goal_file, 'r') as file:
                    data = json.load(file)
                
                # Find assigned_object_coords for robot_1
                for robot in data.get("robot", []):
                    if robot.get("robot_id") == "robot_"+ str(self.namespace[-1]) and robot.get("assigned_object_coords"):
                        self.goal_pose_list = [robot["assigned_object_coords"]]
                        self.goal_x = robot["assigned_object_coords"][0]
                        self.goal_y = robot["assigned_object_coords"][1]
                        print(f"Loaded goal for {self.namespace}: {self.goal_x, self.goal_y}")
                        return
                
                print("Robot or assigned_object_coords not found. Using default goals.")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading the JSON file: {e}. Using default goals.")
        else:
            print(f"Goal file {self.goal_file} not found. Using default goals.")
    
    def generate_default_goals(self):
        """Set default goal poses for training mode."""
        self.goal_pose_list = [[4.55, 8.85], [0.96, 6.4], [7.0, 0.8], [1.57, 1.06], [8.5, 8.5],
                              [4.38, 2.71]]
        # self.goal_pose_list = [[-4.45, -2.04], [-4.0, -4.82], [-3.83, -2.07], 
                            #    [-3.448, -4.788], [-3.64, 8.64], [9.605, -4.84],
                            #    [0.45, -1.45], [-2.52, 5.89]]
        goal_choice = random.choice(self.goal_pose_list)
        self.goal_x = goal_choice[0]
        self.goal_y = goal_choice[1]
        print("Init, goal pose:", self.goal_x, self.goal_y)

    def init_callback(self):
        self.reset_simulation()
        self.publish_callback()
        time.sleep(1)

    def publish_callback(self):
        goal_pose = Pose()
        goal_pose.position.x = self.goal_x
        goal_pose.position.y = self.goal_y
        self.goal_pub.publish(goal_pose)
    
    def task_succeed_callback(self, request, response):
        if self.mode == "training":
            self.generate_goal_pose()
            print(f"Training success: new goal pose: {self.goal_x:.2f}, {self.goal_y:.2f}")
        elif self.mode == "testing":
            self.update_robot_status("success")
            print(f"Testing success: reached goal pose: {self.goal_x:.2f}, {self.goal_y:.2f}")
        return response

    def update_robot_status(self, status):
        if not self.goal_file:
            print("Goal file not specified, cannot update status.")
            return

        try:
            with open(self.goal_file, "r") as file:
                data = json.load(file)
            
            for robot in data["robot"]:
                if (
                    "assigned_object_coords" in robot
                    and len(robot["assigned_object_coords"]) == 2
                    and robot["assigned_object_coords"][0] == self.goal_x
                    and robot["assigned_object_coords"][1] == self.goal_y
                ):
                    robot["status"] = status
            
            with open(self.goal_file, "w") as file:
                json.dump(data, file, indent=4)
            print(f"Updated status to {status} for goal at ({self.goal_x}, {self.goal_y}).")
        except Exception as e:
            print(f"Error updating robot status: {e}")
    
    def task_fail_callback(self, request, response):
        self.reset_simulation()
        print(f"fail: reset the environment, goal pose: {self.goal_x:.2f}, {self.goal_y:.2f}")
        return response
    
    def reset_simulation(self):
        req = Empty.Request()
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset service not available, waiting again...')
        self.reset_simulation_client.call_async(req)
    
    def generate_goal_pose(self):
        if self.mode == "training":
            goal_pose_list = [[4.55, 8.85], [0.96, 6.4], [7.0, 0.8], [1.57, 1.06], [8.5, 8.5],
                              [4.38, 2.71]]
            goal_choice = random.choice(goal_pose_list)
            self.goal_x, self.goal_y = goal_choice
            self.publish_callback()
            print(f"Training mode: New goal generated: ({self.goal_x}, {self.goal_y}).")
        elif self.mode == "testing":
            try:
                with open(self.goal_file, "r") as file:
                    data = json.load(file)
                for robot in data["robot"]:
                    if robot["status"] == "unknown" and robot["assigned_object_coords"]:
                        self.goal_x, self.goal_y = robot["assigned_object_coords"]
                        self.publish_callback()
                        print(f"Testing mode: Goal loaded: ({self.goal_x}, {self.goal_y}).")
                        return
                print("Testing mode: No more unknown goals available.")
            except Exception as e:
                print(f"Error reading goal file: {e}")

def main(args=None):
    rclpy.init(args=args)
    gazebo_node = Gazebo()
    rclpy.spin(gazebo_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
