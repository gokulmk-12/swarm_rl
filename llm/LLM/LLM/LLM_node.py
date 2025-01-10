import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import yaml
from ament_index_python.packages import get_package_share_directory
import os
import json
import numpy as np
import requests

# url = "http://127.0.0.1:9000/blog"
# headers = {
#     "Authorization": "Bearer your_token_here",  # if authentication is needed
#     "Content-Type": "application/json"          # if sending JSON data
# }

class LLMNode(Node):
    def __init__(self, yaml_file):
        super().__init__('LLM_node')
        self.yaml_file = yaml_file
        
        # Load initial robot and object data from YAML file
        self.robot_data = []
        self.object_data = []
        self.load_initial_data()

        # Create subscriptions for each robot's odometry updates
        for i in range(1, len(self.robot_data) + 1):
            pos_topic = f'/robot_{i}/odom'
            self.create_subscription(Odometry, pos_topic, lambda msg, id=i: self.pos_callback(id, msg), 10)
            

    def load_initial_data(self):
        if os.path.exists(self.yaml_file):
            with open(self.yaml_file, 'r') as file:
                data = yaml.safe_load(file)
                print(data)

            # Extract robot1 data and format it as required
            for i, robot in enumerate(data.get('robot1', []), start=1):
                coords = (robot['x'], robot['y'])
                self.robot_data.append({
                    "robot_id": f'robot_{i}',
                    "position": list(coords),
                    "idle": True  # Assuming all robots are idle initially
                })

            # Extract target data and format it as required
            for i, target in enumerate(data.get('target', []), start=1):
                coords = (target['x'], target['y'])
                name = target['name']
                obj_type = '_'.join(name.split('_')[:-1]) if '_' in name else name
                self.object_data.append({
                    "object_id": f"Object_{i}",
                    "coords": coords,
                    "type": obj_type,  # You may want to differentiate types based on your logic
                    "is_assigned": False
                })

            # Log initial data for debugging purposes
            self.get_logger().info(f'Initial Robot Data: {self.robot_data}')
            self.get_logger().info(f'Initial Object Data: {self.object_data}')

            '''response = requests.post(url, json={"robot_id":"robot_1","reached":False}) 
            print(response)'''

            print(type(self.robot_data))
        else:
            self.get_logger().error(f'YAML file {self.yaml_file} does not exist.')

    def pos_callback(self, robot_id, msg):
        robot_no = f'robot_{robot_id}'
        pos = msg.pose.pose.position
        x = pos.x
        y = pos.y



        # Update the robot's position in the dictionary
        #This is the error.
        '''
        [{'robot_id': 'robot_1', 'position': [0.5, 2.5], 'idle': True},
         {'robot_id': 'robot_2', 'position': [8.5, 8.5], 'idle': True}, 
         {'robot_id': 'robot_3', 'position': [6.0, 4.0], 'idle': True}, 
         {'robot_id': 'robot_4', 'position': [8.3, 4], 'idle': True}]

         robot_1

        '''
        '''if robot_no in self.robot_data:
            self.robot_data[robot_no]["position"] = [x, y]

            # Log updated robot data for debugging purposes
            self.get_logger().info(f'Updated {robot_no} Position: {self.robot_data[robot_no]}')'''
        

        for i in self.robot_data:
            if i["robot_id"]==robot_no:
                i["position"]=[x,y]
                self.get_logger().info(f'updated {robot_no} position = {i["position"]}')
                print(self.robot_data)
        with open('robot_data.json', 'w') as json_file:
            json.dump(self.robot_data, json_file, indent=4)
        with open('object_data.json','w') as json_file1:
            json.dump(self.object_data,json_file1, indent = 4)
            

    

    






def main(args=None):
    rclpy.init(args=args)
    yaml_file_path = "config/robot_details.yaml"
    llm_node = LLMNode(yaml_file_path)

    rclpy.spin(llm_node)

    # Destroy the node explicitly after spinning is done.
    llm_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
