import networkx as nx
import matplotlib.pyplot as plt
import os
import mujoco
import gym
from os import path
import numpy as np
from gymnasium import spaces
import xml.etree.ElementTree as ET
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import re
import torch
import dgl
import torch.nn as nn
import argparse
from torch_geometric.nn import GATConv  # Using Graph Attention Convolution
from torch_geometric.utils import add_self_loops
from tensorflow.keras.callbacks import TensorBoard
import torch.nn.functional as F
import itertools
import torch.optim as optim
import tensorflow as tf
import shutil
torch.autograd.set_detect_anomaly(True)


def resize_tensor(tensor, target_shape):
    target_size = target_shape[0]
    current_shape = tensor.shape[0]
    if (current_shape < target_size):
        padding = target_size - current_shape
        padded_tensor = torch.cat((tensor, torch.zeros(padding, tensor.shape[1])), dim=0)
        padded_tensor = torch.flatten(padded_tensor)
        return padded_tensor

    else:
        tensor = tensor[:target_size]
        tensor = torch.flatten(tensor)
        return tensor


class MAPointMazeEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(self, xml_file: str = None, render_mode: str = "human", num_agents = 3, num_goals = 3, width = 1000, height = 700, **kwargs):
        if xml_file is None:
            xml_file = path.join(
                path.dirname(path.realpath(__file__)), "../assets/custom_robot.xml"
            )

        self.num_agents = num_agents
        self.num_goals = num_goals
        self.goal_positions, self.goal_names = self._parse_goal_from_xml(xml_file)
        self.obstacle_positions, self.obstacle_names = self._parse_obstacles_from_xml(xml_file)

        self.num_obstacles = len(self.obstacle_names)
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), xml_file)
        default_camera_config={"distance": 12}

        self.agent_names = [f"agent_{i+1}" for i in range(self.num_agents)]
        self.goal_names = [f"goal_{i+1}" for i in range(self.num_agents)]

        self.collision_threshold = 0.6
        self.obstacle_threshold = 1.1
        self.obs_dim = 2
        self.action_dim = 2

        self.max_distance = 2

        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64) for _ in range(self.num_agents)]  # (x_1, y_1, x_2, y_2)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim, self.num_agents), dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32),  spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)))

        self.previous_positions = [0] * self.num_agents

        super().__init__(
            model_path=xml_path,
            frame_skip=1,
            observation_space=self.observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
            default_camera_config=default_camera_config,
            **kwargs,
        )

    def _parse_goal_from_xml(self, xml_file: str):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        goal_positions = []
        goal_names = []
        for body in root.findall(".//body"):
            if 'goal_1' in body.attrib.get('name', ''):
                goal_positions.append(np.array([float(body.attrib['pos'].split()[0]),
                                                float(body.attrib['pos'].split()[1])]))
                goal_names.append(body.attrib.get('name', ''))
                
            elif 'goal_2' in body.attrib.get('name', ''):
                goal_positions.append(np.array([float(body.attrib['pos'].split()[0]),
                                                float(body.attrib['pos'].split()[1])]))
                goal_names.append(body.attrib.get('name', ''))

            elif 'goal_3' in body.attrib.get('name', ''):
                goal_positions.append(np.array([float(body.attrib['pos'].split()[0]),
                                                float(body.attrib['pos'].split()[1])]))
                goal_names.append(body.attrib.get('name', ''))
                
        if len(goal_positions) != self.num_agents:
            raise ValueError("The XML file must contain exactly three goal geoms: 'g1', 'g2' and 'g3'.")

        return np.array(goal_positions), goal_names
    
    def _parse_obstacles_from_xml(self, xml_file: str):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        obstacle_positions = []
        obstacle_names = []
        for geom in root.findall(".//geom"):
            if 'obstacle1' in geom.attrib.get('name', ''):
                obstacle_positions.append(np.array([float(geom.attrib['pos'].split()[0]),
                                                float(geom.attrib['pos'].split()[1])]))
                obstacle_names.append(geom.attrib.get('name', ''))
            
            elif 'obstacle2' in geom.attrib.get('name', ''):
                obstacle_positions.append(np.array([float(geom.attrib['pos'].split()[0]),
                                                float(geom.attrib['pos'].split()[1])]))
                obstacle_names.append(geom.attrib.get('name', ''))
                
            elif 'obstacle3' in geom.attrib.get('name', ''):
                obstacle_positions.append(np.array([float(geom.attrib['pos'].split()[0]),
                                                float(geom.attrib['pos'].split()[1])]))
                obstacle_names.append(geom.attrib.get('name', ''))
                
            elif 'obstacle4' in geom.attrib.get('name', ''):
                obstacle_positions.append(np.array([float(geom.attrib['pos'].split()[0]),
                                                float(geom.attrib['pos'].split()[1])]))
                obstacle_names.append(geom.attrib.get('name', ''))
                
        return np.array(obstacle_positions), obstacle_names

    def reset_model(self) -> np.ndarray:
        self.set_state(self.init_qpos, self.init_qvel)
        obs, _ = self._get_obs()
        return obs

    def step(self, action_taken):
        
        for i in range(self.num_agents):
            self.previous_positions[i] = self.data.xpos[i+4][:2]
        
        action = action_taken.reshape((6,))
        self.do_simulation(action, self.frame_skip)
        
        obs, info = self._get_obs()
        reward = self._compute_rewards()
        terminated = self._check_done()
        truncated = False
        
        for i in range(self.num_agents):
            self.previous_positions[i] = self.data.xpos[i+4][:2]

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        """Get the current observations for both agents and the goal positions."""
        """Blue is the first agent"""
        agent_pos_1 = np.array(self.data.xpos[4][:2])  # Assuming qpos stores (x, y) position for agent 1
        agent_pos_2 = np.array(self.data.xpos[5][:2])  # Assuming qpos stores (x, y) position for agent 2
        agent_pos_3 = np.array(self.data.xpos[6][:2])  # Assuming qpos stores (x, y) position for agent 3

        observation = np.array([agent_pos_1, agent_pos_2, agent_pos_3])
        
        return observation, {}

    def _compute_rewards(self):
        rewards = [0] * self.num_agents

        for i, agent_name in enumerate(self.agent_names):
            agent_pos = self.data.xpos[i + 4][:2]
            goal_pos = self.goal_positions[i]
            
            agent_reward = 0

            # Goal proximity reward (encourages reaching the goal)
            goal_reward = max(0, 10 - np.linalg.norm(agent_pos - goal_pos))  # Reward based on goal proximity
            agent_reward += goal_reward

            # Movement reward (encourages moving, discourages staying still)
            movement_reward = np.linalg.norm(agent_pos - self.previous_positions[i])
            if movement_reward < 0.1:  # Penalize for staying still
                agent_reward -= 0.5  # Small penalty for inaction
            agent_reward += movement_reward  # Reward for moving toward goal

            # Collision penalties
            for j in range(self.num_agents):
                if j != i:
                    agent_pos = self.data.xpos[i + 4]
                    pos2 = self.data.xpos[j + 4]
                    if self.check_agent_collision(agent_pos, pos2):
                        agent_reward -= 2  # Strong penalty for agent-agent collision
            
            for obstacle in self.obstacle_positions:
                if self.check_collision_with_obstacles(obstacle, self.data.xpos[i+4][:2]):
                    agent_reward -= 10  # Strong penalty for agent-obstacle collision

            # Encourage spacing between agents (penalize when too close)
            for j in range(self.num_agents):
                if j != i:
                    agent_pos = self.data.xpos[i + 4][:2]
                    pos2 = self.data.xpos[j + 4][:2]
                    distance = np.linalg.norm(agent_pos - pos2)
                    if distance < 1.0:  # Threshold for too close
                        agent_reward -= 5  # Small penalty for crowding

            rewards[i] = agent_reward

        return rewards

    
    def _check_done(self):
        """
        Check if the episode should terminate.
        """
        agent1_pos = self.data.xpos[4][:2]
        agent2_pos = self.data.xpos[5][:2]
        agent3_pos = self.data.xpos[6][:2]
        
        goal1_reached = np.linalg.norm(agent1_pos - self.goal_positions[0]) < self.collision_threshold
        goal2_reached = np.linalg.norm(agent2_pos - self.goal_positions[1]) < self.collision_threshold
        goal3_reached = np.linalg.norm(agent3_pos - self.goal_positions[2]) < self.collision_threshold

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                pos1 = self.data.xpos[i+4]
                pos2 = self.data.xpos[j+4]
                if self.check_agent_collision(pos1, pos2):
                    return True
        
        for i in range(self.num_agents):
            for obstacle_pos in self.obstacle_positions:
                if self.check_collision_with_obstacles(obstacle_pos, self.data.xpos[i+4][:2]):
                    return True
        
        if goal1_reached and goal2_reached and goal3_reached:
            return -1
        
        return False
    
    def check_agent_collision(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2) < self.collision_threshold

    def check_collision_with_obstacles(self, obstacle_pos, agent_pos):
        return np.linalg.norm(agent_pos - obstacle_pos) < self.obstacle_threshold

    def agents_local_observation(self, velocity):
        agent_pos_1 = np.array(self.data.xpos[4][:2])  # Assuming qpos stores (x, y) position for agent 1
        
        velocity = velocity.reshape((6, ))
        
        agents_local_observation = []
        
        
        for i in range(self.num_goals):
            agent_pos = self.data.xpos[i + 4][:2]
            goal_pos = self.goal_positions[i]
            agent_x, agent_y = agent_pos[0], agent_pos[1]
            goal_x, goal_y = goal_pos[0], goal_pos[1]
            relative_pos_x, relative_pos_y = goal_x - agent_x, goal_y - agent_y            
            velocity_x, velocity_y = velocity[2 * i], velocity[2 * i + 1]
            temp = np.array([[agent_x, agent_y], [velocity_x, velocity_y], [relative_pos_x, relative_pos_y]])
            agents_local_observation.append(temp)
            
        agents_local_observation = np.array(agents_local_observation)
        
        
        return agents_local_observation
    
    def graph_of_agent_i(self, temp):
        rows = len(temp)
        cols = rows
        adj_matrix = np.zeros((rows + 1, cols + 1))
        node_observation = []
        
        node_observation.append(list(itertools.chain.from_iterable([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], ['agent', 'self']])))
        entity_name_list = []
        entity_name_list.append('agent_self')
        
        for i in range(len(temp)):
            entity_type, entity_name = temp[i][3][0], temp[i][3][1]
            relative_pos_x, relative_pos_y = float(temp[i][0][0]), float(temp[i][0][1])
            eucld_dist = np.sqrt(relative_pos_x * relative_pos_x + relative_pos_y * relative_pos_y) 
            number = int(re.findall(r'\d+', entity_name)[0])
            if(entity_type == 'agent'):
                adj_matrix[0][i + 1] = eucld_dist
                adj_matrix[i + 1][0] = eucld_dist
            
            elif(entity_type == 'goal'):
                adj_matrix[0][i + 1] = eucld_dist
            
            elif(entity_type == 'obstacle'):
                adj_matrix[0][i + 1] = eucld_dist
                
            node_observation.append(list(itertools.chain.from_iterable(temp[i])))
            entity_name_list.append(f'{entity_type}_{entity_name}')
        
        for i in range(len(entity_name_list)):
            node_observation[i].pop(7)
            node_observation[i][6] = i
        
        node_observation = np.array(node_observation)
        adj_matrix = np.array(adj_matrix)

        return adj_matrix, node_observation
    
    def make_graph(self, temp_node):
        rows = self.num_agents + self.num_goals + self.num_obstacles
        cols = rows
        adj_matrix = np.zeros((rows, cols))

        for agent_index in range(int(self.num_agents)):
            for j in range(len(temp_node[agent_index])):
                entity_type, entity_name = temp_node[agent_index][j][3][0], temp_node[agent_index][j][3][1]
                relative_pos_x, relative_pos_y = float(temp_node[agent_index][j][0][0]), float(temp_node[agent_index][j][0][1])
                eucld_dist = np.sqrt(relative_pos_x * relative_pos_x + relative_pos_y * relative_pos_y) 
                number = int(re.findall(r'\d+', entity_name)[0])
                
                if(entity_type == 'agent'):
                    adj_matrix[agent_index][number - 1] = eucld_dist
                
                elif(entity_type == 'goal'):
                    adj_matrix[agent_index][self.num_agents + number - 1] = eucld_dist
                
                elif(entity_type == 'obstacle'):
                    adj_matrix[agent_index][self.num_agents + self.num_goals + number - 1] = eucld_dist
        
        
        row, col = np.where(adj_matrix != 0)

        # Create a DGL graph
        graph = dgl.graph((row, col), num_nodes=adj_matrix.shape[0])
        edge_weights = adj_matrix[row, col]
        graph.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)
        
        new_node_ids = {}
        
        for i in range(self.num_agents):
            new_node_ids[i] = f'agent_{i + 1}'
        
        for i in range(self.num_goals):
            new_node_ids[self.num_agents + i] = f'goal_{i + 1}'
        
        for i in range(self.num_obstacles):
            new_node_ids[self.num_agents + self.num_goals + i] = f'obstacle_{i + 1}'
        
        node_label_tensor = torch.tensor([i for i in range(graph.num_nodes())], dtype=torch.int64)
        edge_weights = adj_matrix[row, col]
        graph.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)
        graph.ndata['new_labels'] = node_label_tensor
        # print("New Node Labels:", graph.ndata['new_labels'])       
        # Print the graph
        # print(graph)
        
        # print("Edge weights:", graph.edata['weight'])
        # print("Edge indices:", graph.edges())

        
        # Convert DGL graph to NetworkX graph for visualization
        # nx_graph = graph.to_networkx()
        # # Extract edge weights and create a dictionary for edge labels
        # edge_labels = {(i, j):  f'{graph.edata["weight"][k].item():.2f}' for k, (i, j) in enumerate(zip(row, col))}
        # # Draw the graph with node labels and edge labels (weights)
        # plt.figure(figsize=(8, 6))
        # pos = nx.spring_layout(nx_graph)  # Generate layout for node positions
        # nx.draw(nx_graph, pos, labels={i: new_node_ids[i] for i in range(graph.num_nodes())}, with_labels=True, font_weight='bold')
        # # Draw edge labels (weights)
        # nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=10, font_weight='bold')
        # plt.show()        

        # print("adjacency matrix from graph = ", graph.adj())
        
        return graph, adj_matrix
        
    # entity_type = {agent, obstacle, goal}
    def node_observation(self, velocity, agent_index, sensing_radius = 3):
        entity_type = {"agent", "obstacle", "goal"}
        temp = []
        velocity = velocity.reshape((6, ))
        original_agent_pos = self.data.xpos[agent_index + 4][:2]
        original_agent_vel = [velocity[2 * agent_index], velocity[2 * agent_index + 1]]
        original_agent_goal = self.goal_positions[agent_index]
        
        for i in range(self.num_agents):
            agent_name = self.agent_names[i]
            agent_pos = self.data.xpos[i + 4][:2]
            if(i != agent_index and np.linalg.norm(original_agent_pos - agent_pos) <= sensing_radius):
                agent_vel = [velocity[2 * i], velocity[2 * i + 1]]
                agent_goal = self.goal_positions[i]
                relative_pos_x, relative_pos_y = agent_pos[0] - original_agent_pos[0], agent_pos[1] - original_agent_pos[1]
                relative_vel_x, relative_vel_y = agent_vel[0] - original_agent_vel[0], agent_vel[1] - original_agent_vel[1]
                relative_goal_x, relative_goal_y = agent_goal[0] - original_agent_goal[0], agent_goal[1] - original_agent_goal[1]
                entity_type = ["agent", agent_name]

                temp.append(np.array([[relative_pos_x, relative_pos_y], [relative_vel_x, relative_vel_y], [relative_goal_x, relative_goal_y], entity_type]))
                    
         
        for i in range(self.num_obstacles):       
            obstacle_pos = self.obstacle_positions[i]
            obstacle_name = self.obstacle_names[i]
            if(np.linalg.norm(agent_pos - obstacle_pos) <= sensing_radius):
                relative_pos_x, relative_pos_y = obstacle_pos[0] - original_agent_pos[0], obstacle_pos[1] - original_agent_pos[1]
                relative_vel_x, relative_vel_y = (-1) * original_agent_vel[0], (-1) * original_agent_vel[1]
                entity_type = ["obstacle", obstacle_name]
                
                temp.append(np.array([[relative_pos_x, relative_pos_y], [relative_vel_x, relative_vel_y], [relative_pos_x, relative_pos_y], entity_type]))

        for i in range(self.num_goals):
            goal_pos = self.goal_positions[i]
            goal_name = self.goal_names[i]
            if(np.linalg.norm(goal_pos - original_agent_pos) <= sensing_radius):
                relative_pos_x, relative_pos_y = goal_pos[0] - original_agent_pos[0], goal_pos[1] - original_agent_pos[1]
                relative_vel_x, relative_vel_y = (-1) * original_agent_vel[0], (-1) * original_agent_vel[1]
                entity_type = ["goal", goal_name]
                
                temp.append(np.array([[relative_pos_x, relative_pos_y], [relative_vel_x, relative_vel_y], [relative_pos_x, relative_pos_y], entity_type])) 
        
        temp = np.array(temp)
        return temp                   
                    

class GNN(nn.Module):
    def __init__(self, num_features, hidden_size, num_heads, concat_heads, layer_N, use_relu):
            super(GNN, self).__init__()
            
            self.layer_N = layer_N
            self.concat_heads = concat_heads
            
            # Ensure hidden_size is divisible by num_heads
            assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
            
            ## Define GNN layers
            self.convs = nn.ModuleList()
            for i in range(layer_N):
                in_channels = num_features if i == 0 else hidden_size
                self.convs.append(GATConv(in_channels, hidden_size // num_heads, heads=num_heads))

            # Activation function
            self.activation = nn.ReLU() if use_relu else nn.Identity()

    def adjacency_to_edge_index(self, adj):
        # Get the indices of non-zero entries (edges)
        edge_index = torch.nonzero(adj, as_tuple=False).t().contiguous()
        return edge_index

    def forward(self, x, adj):
        edge_index = self.adjacency_to_edge_index(adj)

        # Add self-loops to the edge index
        edge_index, _ = add_self_loops(edge_index)

        
        for conv in self.convs:
            x = conv(x.clone(), edge_index.clone())
            x = self.activation(x.clone())

            # If concatenating heads
            if self.concat_heads:
                x = x.view(-1, x.size(1) * x.size(2))  # Concatenate heads
            
            
        return x
            

class ActorNetwork(nn.Module):
    def __init__(self, step_size = 10, input_dim = 20 * 2, output_dim = 2, stacked_frames=1, hidden_size=64, layer_N=1):
        super(ActorNetwork, self).__init__()
        
        # Parameters
        self.stacked_frames = stacked_frames
        self.hidden_size = hidden_size
        self.layer_N = layer_N
        self.step_size = step_size
        # Define layers
        self.layers = nn.ModuleList()
        for _ in range(layer_N):
            if _ == 0:
                # Adjusting input dimension for stacked frames
                self.layers.append(nn.Linear(input_dim * stacked_frames, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer for movement in x and y directions
        self.output_layer = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # Forward pass through the layers
        for layer in self.layers:
            x = F.relu(layer(x.clone()))
        
        # Output movement in x and y directions
        movement = self.output_layer(x) * self.step_size
        # movement = F.softmax(movement, dim=-1)
        return movement


class CriticNetwork(nn.Module):
    def __init__(self, input_dim = 20 * 2, stacked_frames=1, hidden_size=64, layer_N=1):
        super(CriticNetwork, self).__init__()
        
        self.stacked_frames = stacked_frames
        self.hidden_size = hidden_size
        self.layer_N = layer_N
        
        # Define layers
        self.layers = nn.ModuleList()
        for _ in range(layer_N):
            in_dim = input_dim if _ == 0 else hidden_size
            self.layers.append(nn.Linear(in_dim, hidden_size))
        
        # Final output layer for value estimation
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, batch=None):
        # Process through hidden layers
        for layer in self.layers:
            x = torch.relu(layer(x.clone()))  # Activation function
        
        # Output value estimate
        value = self.output_layer(x.clone())
        return value


def actor_step(actor_network, info_aggr_list):
    movement_output_list = []    
    for info_aggr in info_aggr_list:
        movement_output = actor_network(info_aggr.float())
        movement_output_list.append(movement_output.detach().numpy())
    
    action = np.array(movement_output_list)        
    # action = torch.stack(movement_output_list)
    
    return action

def critic_step(env, model, critic_network):
    temp_node = []
    for i in range(env.num_agents):
        temp_node.append(env.node_observation(velocity=np.random.normal(loc=0, scale=1, size=(3,2)), agent_index=i))

    graph, adj_matrix = env.make_graph(temp_node=temp_node)
    
    adj_matrix_each_node = []
    node_observation_each_node = []
    x_agg = []

    agents_local_observation = env.agents_local_observation(velocity=np.random.normal(loc=0, scale=1, size=(3,2)))
    agents_local_observation = torch.from_numpy(agents_local_observation)
    
    info_aggr_list = []
    movement_output_list = []
    for i in range(len(temp_node)):
        x, y = env.graph_of_agent_i(temp_node[i])
        adj_matrix_each_node.append(x)
        node_observation_each_node.append(y)
        adj_matrix = torch.from_numpy(x).float()
        node_features = torch.from_numpy(y.astype(float)).float()

        # Perform a forward pass
        output = model.forward(node_features, adj_matrix)
        
        x_agg.append(output)

        info_aggr = torch.concat((agents_local_observation[i], output), dim = 0)
        
        info_aggr = resize_tensor(info_aggr, (20, 2))
        info_aggr_list.append(info_aggr)


    # Stack tensors into a single tensor of shape (3, 40, 1)
    stacked_tensors = torch.stack(info_aggr_list)  

    # Perform global average pooling
    graph_info_aggr = stacked_tensors.mean(dim=0)

    value_estimate = critic_network.forward(graph_info_aggr.float().clone())
    
    return value_estimate, info_aggr_list

if __name__ == "__main__":
    
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn_hidden_size", type=int, default=2, help="Hidden layer dimension in the GNN")
    parser.add_argument("--gnn_num_heads", type=int, default=1, help="Number of heads in the transformer conv layer (GNN)")
    parser.add_argument("--gnn_concat_heads", action="store_true", default=False, help="Whether to concatenate the head output or average")
    parser.add_argument("--gnn_layer_N", type=int, default=4, help="Number of GNN conv layers")
    parser.add_argument("--gnn_use_ReLU", action="store_false", default=True, help="Whether to use ReLU in GNN conv layers")
    args = parser.parse_args()
    num_features = 7
    
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Save the weights after every episode (or at some intervals)
    actor_path = os.path.join(weights_dir, "actor_weights.pth")
    critic_path = os.path.join(weights_dir, "critic_weights.pth")
    model_path = os.path.join(weights_dir, "model_weights.pth")

    
    log_dir = "logs/metrics"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)

    
    # Initialize the model with parsed arguments
    model = GNN(
        num_features=num_features,
        hidden_size=args.gnn_hidden_size,
        num_heads=args.gnn_num_heads,
        concat_heads=args.gnn_concat_heads,
        layer_N=args.gnn_layer_N,
        use_relu=args.gnn_use_ReLU
    )

    input_dim = 20 * 2
    output_dim = 2
    step_size = 10
    actor_network = ActorNetwork(
        step_size=step_size,
        input_dim=input_dim,
        output_dim=output_dim,
        stacked_frames=1,
        hidden_size=64,
        layer_N=1
    )
    
    critic_network = CriticNetwork(
        input_dim=input_dim,
        stacked_frames=1,
        hidden_size=64,
        layer_N=2
    )
    
    # Hyperparameters
    learning_rate_actor = 0.001
    learning_rate_critic = 0.001
    learning_rate_gnn = 0.001
    gamma = 0.99  # Discount factor
    num_episodes = 10000  # Number of episodes to train

    actor_optimizer = optim.Adam(actor_network.parameters(), lr=learning_rate_actor)
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate_gnn)
    critic_optimizer = optim.Adam(critic_network.parameters(), lr=learning_rate_critic)




    total_reward = 0
    step = 0
    
    previous_loss = float('inf')  # Start with a very high value
    
    for episode_num in range(num_episodes):
        print("episode = ", episode_num)
        env = MAPointMazeEnv(xml_file="pointmaze_final.xml", render_mode="human", num_agents=3)
        observation, _ = env.reset()
        terminated = False   
        while not terminated:
            value_estimate, info_aggr_list = critic_step(env=env, model=model, critic_network=critic_network)
        
            action = actor_step(actor_network=actor_network, info_aggr_list=info_aggr_list)

            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            next_state_value_estimate, _ = critic_step(env=env, model=model, critic_network=critic_network) 
            
            td_error = sum(reward) + (1 - terminated) * gamma * next_state_value_estimate - value_estimate
            critic_loss = (td_error) ** 2

            action = torch.from_numpy(action)
            action = F.softmax(action, dim=-1)
            negative_log_action = (-1) * np.log(action)
            
            actor_loss = negative_log_action.sum().item() * td_error.clone()

            model_loss = actor_loss + critic_loss
                
            actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)

            critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)

            model_optimizer.zero_grad()
            model_loss.backward(retain_graph=True)

            model_optimizer.step()
            actor_optimizer.step()
            critic_optimizer.step()


            total_reward = sum(reward)
            step = step + 1

            with summary_writer.as_default():
                tf.summary.scalar('Total Reward', total_reward.item(), step=step)
                tf.summary.scalar('Actor loss', actor_loss[0].detach().numpy().item(), step=step)
                tf.summary.scalar('Critic loss', critic_loss[0].detach().numpy().item(), step=step)
                tf.summary.scalar('Model loss', model_loss[0].detach().numpy().item(), step=step)
                tf.summary.scalar(f'Total Reward, episode number {episode_num}', total_reward.item(), step=step)
                tf.summary.scalar(f'Actor loss, episode number {episode_num}', actor_loss[0].detach().numpy().item(), step=step)
                tf.summary.scalar(f'Critic loss, episode number {episode_num}', critic_loss[0].detach().numpy().item(), step=step)
                tf.summary.scalar(f'Model loss, episode number {episode_num} ', model_loss[0].detach().numpy().item(), step=step)


            if(model_loss[0].detach().numpy() < previous_loss):
                torch.save(actor_network.state_dict(), actor_path)
                torch.save(critic_network.state_dict(), critic_path)
                torch.save(model.state_dict(), model_path)
                previous_loss = model_loss[0].detach().numpy().item()
                print("Saving model at step ", step)


            if terminated != -1 and terminated != True:
                continue

            if terminated == -1:
                break
            else:
                break
            # print("Observation:", observation, type(observation), observation.shape)
            # print("Reward:", reward, type(reward))
            # print("Dones:", terminated, type(terminated))
        if(terminated == -1):
            break
