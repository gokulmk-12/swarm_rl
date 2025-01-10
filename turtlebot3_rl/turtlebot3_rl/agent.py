import os
import sys
import csv
import copy
import time
import json
import rclpy
import torch
import yaml
import requests
from rclpy.node import Node

from std_srvs.srv import Empty
from custom_msgs.srv import DrlStep, Goal
from geometry_msgs.msg import Twist

from turtlebot3_rl.util import *
from turtlebot3_rl.td3 import TD3
from turtlebot3_rl.ddpg import DDPG
from turtlebot3_rl.replaybuffer import ReplayBuffer


from torch.utils.tensorboard import SummaryWriter

# If small environment, use this below line
# params_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "params.yaml")
# If large environment, use this below line, and comment the previous
params_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "params_large.yaml")

with open(params_file, 'r') as file:
    data = yaml.safe_load(file)

OBSERVE_STEPS = data.get('OBSERVE_STEPS', None)

class Agent(Node):
    def __init__(self, training, load_episode=4800, namespace=""):
        node_name = f"{namespace}_td3_agent" if namespace else "td3_agent"
        super().__init__(node_name)
        self.training = int(training)
        self.episode = int(load_episode)

        prefix = f"{namespace}/" if namespace else ""
        self.step_comm_service = f"{prefix}step_comm"
        self.goal_comm_service = f"{prefix}goal_comm"
        self.cmd_topic = f"{prefix}cmd_vel"
        self.namespace = namespace
        self.offset_episode = load_episode
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sim_speed = 1
        self.total_steps = 0
        print(f"{'** Start Training **' if (self.training) else '** Start Testing **' }")
        self.observe_steps = OBSERVE_STEPS if self.training else 0

        # self.model = DDPG(self.device, self.sim_speed)
        self.model = TD3(self.device, self.sim_speed)
        self.replay_buffer = ReplayBuffer(self.model.buffer_size) if self.training else None

        self.model_save_path = "models"
        os.makedirs(self.model_save_path, exist_ok=True)

        if load_episode > 0:
            self.load_model(self.episode)
            if self.training:
                self.observe_steps = 0
        
        self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(self.model_save_path, "tensorboard_logs"))
        
        ## Clients
        self.step_comm_client = self.create_client(DrlStep, self.step_comm_service)
        self.goal_comm_client = self.create_client(Goal, self.goal_comm_service)
        self.gazebo_pause = self.create_client(Empty, '/pause_physics')
        self.gazebo_unpause = self.create_client(Empty, '/unpause_physics')
        
        ## Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.request = True
        self.process()
    
    def load_model(self, load_episode):
        """Load the actor and critic models."""
        actor_path = os.path.join(self.model_save_path, f"actor_{load_episode}.pth")
        critic_path = os.path.join(self.model_save_path, f"critic_{load_episode}.pth")

        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.model.actor.load_state_dict(torch.load(actor_path, weights_only=True))
            self.model.critic.load_state_dict(torch.load(critic_path, weights_only=True))
            print(f"Loaded models: {actor_path} and {critic_path}")
        else:
            print(f"Pre-trained weights not found for episode {load_episode}. Starting fresh.")
    
    def save_model(self):
        """Save the actor and critic models."""
        actor_path = os.path.join(self.model_save_path, f"td3_actor_{self.episode}.pth")
        critic_path = os.path.join(self.model_save_path, f"td3_critic_{self.episode}.pth")

        torch.save(self.model.actor.state_dict(), actor_path)
        torch.save(self.model.critic.state_dict(), critic_path)

        print(f"Models saved: {actor_path} and {critic_path}")
    
    def save_replay_buffer(self):
        buffer_path = os.path.join(self.model_save_path, f"replay_buffer_{self.episode}.pt")
        torch.save(self.replay_buffer, buffer_path)
        print(f"Replay buffer saved: {buffer_path}")
    
    def load_replay_buffer(self, load_episode):
        buffer_path = os.path.join(self.model_save_path, f"replay_buffer_{load_episode}.pt")
        if os.path.exists(buffer_path):
            self.replay_buffer = torch.load(buffer_path)
            print(f"Replay buffer loaded: {buffer_path}")
        else:
            print(f"No replay buffer found for episode {load_episode}. Starting with an empty buffer.")
    
    def update_goal_status_from_file(self):
        """Check and update the robot's goal status from the JSON file."""
        # json_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "llm", "datafinal.json")
        json_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "llm", "datafinallarge.json")
        try:
            with open(json_file, "r") as file:
                data = json.load(file)
            
            # Assuming that each goal contains the coordinates and status
            for robot in data["robot"]:
                assigned_coords = robot.get("assigned_object_coords", None)
                if assigned_coords and robot["robot_id"] == "robot_"+str(self.namespace[-1]):
                    self.goal_x, self.goal_y = assigned_coords 
                    self.status = robot.get("status", "unknown")
                    break
        except Exception as e:
            print(f"Error reading goal status from file: {e}")
            self.status = "unknown"
            self.goal_x, self.goal_y = None, None
    
    def stop_robot(self):
        """Stop the robot by publishing zero velocity."""
        twist_msg = Twist()
        self.cmd_vel_pub.publish(twist_msg)
    
    def process(self):
        pause_simulation(self)

        while True:
            wait_new_goal(self)
            episode_done = False
            steps, reward_sum, loss_critic, loss_actor = 0, 0, 0, 0
            action_past = [0.0, 0.0]
            state = init_episode(self)

            unpause_simulation(self)
            time.sleep(0.5)
            episode_start = time.perf_counter()

            while not episode_done:
                self.update_goal_status_from_file()

                if self.status == "unknown":
                    if self.training and self.total_steps < self.observe_steps:
                        action = self.model.get_action_random()
                    else:
                        action = self.model.get_action(state, self.training, steps, False)
                    
                    action_current = action
                    
                    next_state, reward, episode_done, outcome, distance_travelled = step(self, action_current, action_past)
                    action_past = copy.deepcopy(action_current)
                    reward_sum += reward
                
                elif self.status == "success":
                    # print("Hi")
                    # response_dict = {}
                    # response_dict["robot_id"] = "robot_"+str(self.namespace[-1])
                    # response_dict["object_coord"] = [self.goal_x, self.goal_y]
                    # if self.request:
                    #     try:
                    #         response = requests.post(url="http://127.0.0.1:9000/set", json=response_dict)
                    #         if response.status_code == 200:
                    #             self.request = False
                    #     except Exception as e:
                    #         print("Error: {e}")
                    self.stop_robot()
                    continue

                if self.training == True:
                    self.replay_buffer.add_sample(state, action, [reward], next_state, [episode_done])
                    if self.replay_buffer.get_length() >= self.model.batch_size:
                        loss_c, loss_a = self.model._train(self.replay_buffer)
                        loss_critic += loss_c
                        loss_actor += loss_a
                
                state = copy.deepcopy(next_state)
                steps += 1
                time.sleep(self.model.step_time)
            
            pause_simulation(self)
            self.total_steps += steps
            duration = time.perf_counter() - episode_start

            self.finish_episode(steps, duration, outcome, distance_travelled, reward_sum, loss_critic, loss_actor)
    
    def finish_episode(self, step, eps_duration, outcome, dist_traveled, reward_sum, loss_critic, loss_actor):
        if self.total_steps <= self.observe_steps:
            print(f"Observation Phase: {self.total_steps}/{self.observe_steps} steps")
        
        if not self.training:
            print(f"Outcome: {translate_outcome(outcome):<13} Steps: {step:<6} Time: {eps_duration:<6.2f}")
            return
        
        self.episode += 1
        log_episode = self.episode - self.offset_episode

        print(f"Epi: {self.episode:<5} R: {reward_sum:<8.0f} outcome: {translate_outcome(outcome):<13}", end='')
        print(f"steps: {step:<6} steps_total: {self.total_steps:<7} time: {eps_duration:<6.2f}")

        # Initialize event counts if not already done
        if not hasattr(self, 'event_counts'):
            self.event_counts = {
                "SUCCESS": 0,
                "COLL_OBS": 0,
                "COLL_WALL": 0,
                "TUMBLE": 0,
                "TIMEOUT": 0
            }

        # Update event counts based on outcome
        if outcome == SUCCESS:
            self.event_counts["SUCCESS"] += 1
        elif outcome == COLLISION_OBSTACLE:
            self.event_counts["COLL_OBS"] += 1
        elif outcome == COLLISION_WALL:
            self.event_counts["COLL_WALL"] += 1
        elif outcome == TUMBLE:
            self.event_counts["TUMBLE"] += 1
        elif outcome == TIMEOUT:
            self.event_counts["TIMEOUT"] += 1

        self.tensorboard_writer.add_scalar("Loss/Actor", loss_actor, log_episode)
        self.tensorboard_writer.add_scalar("Loss/Critc", loss_critic, log_episode)
        self.tensorboard_writer.add_scalar("Reward/Sum", reward_sum, log_episode)

        self.tensorboard_writer.add_scalars("Events", {
            "SUCCESS": self.event_counts["SUCCESS"],
            "COLL_OBS": self.event_counts["COLL_OBS"],
            "COLL_WALL": self.event_counts["COLL_WALL"],
            "TUMBLE": self.event_counts["TUMBLE"],
            "TIMEOUT": self.event_counts["TIMEOUT"]
        }, log_episode)

        # Save models and replay buffer periodically during training
        if self.training and self.episode % 200 == 0:
            self.save_model()

def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    rlagent = Agent(training=args[0], namespace=args[1])
    rclpy.spin(rlagent)
    rclpy.shutdown()

def main_train(args=sys.argv[1:]):
    args=['1']+args
    main(args)

def main_test(args=sys.argv[1:]):
    args = ['0'] + args
    main(args)

if __name__ == "__main__":
    main()
