import os
import yaml
import time
import rclpy
from std_srvs.srv import Empty
from custom_msgs.srv import Goal, DrlStep

# If small environment, use this below line
# params_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "params.yaml")
#If large environment, use this below line, and comment the previous
params_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "params_large.yaml")

with open(params_file, 'r') as file:
    data = yaml.safe_load(file)

UNKNOWN                 = data.get('UNKNOWN', None)
SUCCESS                 = data.get('SUCCESS', None)
COLLISION_WALL          = data.get('COLLISION_WALL', None)
COLLISION_OBSTACLE      = data.get('COLLISION_OBSTACLE', None)
TIMEOUT                 = data.get('TIMEOUT', None)
TUMBLE                  = data.get('TUMBLE', None)

def step(agent_self, action, previous_action):
    req = DrlStep.Request()
    req.action = action
    req.previous_action = previous_action

    while not agent_self.step_comm_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('env step service not available, waiting again...')
    future = agent_self.step_comm_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                return res.state, res.reward, res.done, res.success, res.distance_traveled
            else:
                agent_self.get_logger().error(
                    'Exception while calling service: {0}'.format(future.exception()))
                print("ERROR getting step service response!")

def init_episode(agent_self):
    state, _, _, _, _ = step(agent_self, [], [0.0, 0.0])
    return state

def get_goal_status(agent_self):
    req = Goal.Request()
    while not agent_self.goal_comm_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('new goal service not available, waiting again...')
    future = agent_self.goal_comm_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                return res.new_goal
            else:
                agent_self.get_logger().error(
                    'Exception while calling service: {0}'.format(future.exception()))
                print("ERROR getting service response!")

def wait_new_goal(agent_self):
    while(get_goal_status(agent_self) == False):
        print("Waiting for new goal... (if persists: reset gazebo node)")
        time.sleep(1.0)

def pause_simulation(agent_self):
    while not agent_self.gazebo_pause.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('pause gazebo service not available, waiting again...')
    future = agent_self.gazebo_pause.call_async(Empty.Request())
    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            return

def unpause_simulation(agent_self):
    while not agent_self.gazebo_unpause.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('unpause gazebo service not available, waiting again...')
    future = agent_self.gazebo_unpause.call_async(Empty.Request())
    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            return

def translate_outcome(outcome):
    if outcome == SUCCESS:
        return "SUCCESS"
    elif outcome == COLLISION_WALL:
        return "COLL_WALL"
    elif outcome == COLLISION_OBSTACLE:
        return "COLL_OBST"
    elif outcome == TIMEOUT:
        return "TIMEOUT"
    elif outcome == TUMBLE:
        return "TUMBLE"
    else:
        return f"UNKNOWN: {outcome}"