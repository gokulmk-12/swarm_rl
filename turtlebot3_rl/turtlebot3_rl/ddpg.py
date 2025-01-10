import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from turtlebot3_rl.env import NUM_SCAN_SAMPLES

ACTION_SIZE = 2
HIDDEN_SIZE = 512
BATCH_SIZE = 128
BUFFER_SIZE = 1000000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.003
TAU = 0.003
STEP_TIME = 0.01
EPSILON_DECAY = 0.9995
EPSILON_MINIMUM = 0.05
STACK_DEPTH =  3
FRAME_SKIP = 4

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.99, min_sigma=0.01, decay_period=600000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_noise(self, t=0):
        ou_state = self.evolve_state()
        decaying = float(float(t) / self.decay_period)
        self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
        return ou_state


class Network(nn.Module, ABC):
    def __init__(self, name, visual=None):
        super(Network, self).__init__()
        self.name = name
        self.visual = visual
        self.iteration = 0
    
    @abstractmethod
    def forward():
        pass

    def init_weights(n, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class OffPolicyAgent(ABC):
    def __init__(self, device, simulation_speed):
        self.device = device
        self.simulation_speed = simulation_speed

        self.state_size = NUM_SCAN_SAMPLES + 4
        self.action_size = ACTION_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.input_size = self.state_size

        self.batch_size = BATCH_SIZE
        self.buffer_size = BUFFER_SIZE
        self.discount_factor = DISCOUNT_FACTOR
        self.learning_rate = LEARNING_RATE
        self.tau = TAU

        self.step_time = STEP_TIME
        self.loss_function = F.smooth_l1_loss
        self.epsilon = 1.0
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_minimum = EPSILON_MINIMUM
        self.stack_depth = STACK_DEPTH
        self.frame_skip  = FRAME_SKIP

        self.networks = []
        self.iteration = 0
    
    @abstractmethod
    def train():
        pass

    @abstractmethod
    def get_action():
        pass

    @abstractmethod
    def get_action_random():
        pass

    def _train(self, replaybuffer):
        batch = replaybuffer.sample(self.batch_size)
        sample_s, sample_a, sample_r, sample_ns, sample_d = batch

        sample_s = torch.from_numpy(sample_s).to(self.device)
        sample_a = torch.from_numpy(sample_a).to(self.device)
        sample_r = torch.from_numpy(sample_r).to(self.device)
        sample_ns = torch.from_numpy(sample_ns).to(self.device)
        sample_d = torch.from_numpy(sample_d).to(self.device)

        result = self.train(sample_s, sample_a, sample_r, sample_ns, sample_d)
        self.iteration += 1

        if self.epsilon and self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay
        
        return result
    
    def create_network(self, type, name):
        network = type(name, self.input_size, self.action_size, self.hidden_size).to(self.device)
        self.networks.append(network)
        return network
    
    def create_optimizer(self, network):
        return torch.optim.AdamW(network.parameters(), self.learning_rate)
    
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        self.apply(super().init_weights)
    
    def forward(self, states, visualize=False):
        x1 = torch.relu(self.fc1(states))
        x2 = torch.relu(self.fc2(x1))
        action = torch.tanh(self.fc3(x2))

        if visualize and self.visual:
            self.visual.update_layers(states, action, [x1, x2], [self.fc1.bias, self.fc2.bias])
        
        return action

class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        self.l1 = nn.Linear(state_size, int(hidden_size / 2))
        self.l2 = nn.Linear(action_size, int(hidden_size / 2))
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)

        self.apply(super().init_weights)

    def forward(self, states, actions):
        xs = torch.relu(self.l1(states))
        xa = torch.relu(self.l2(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.l3(x))
        x = self.l4(x)
        return x

class DDPG(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)
        self.noise = OUNoise(self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)

        self.actor = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.critic = self.create_network(Critic, 'critic')
        self.critic_target = self.create_network(Critic, 'target_critic')
        self.critic_optimizer = self.create_optimizer(self.critic)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        print("Welcome folks, DDPG has Started !!!")
    
    def get_action(self, state, is_training, step, visualize=False):
        state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
        action = self.actor(state, visualize)
        if is_training:
            noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(step))).to(self.device)
            action = torch.clamp(torch.add(action, noise), -1.0, 1.0)
        return action.detach().cpu().data.numpy().tolist()

    def get_action_random(self):
        return [np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0)] * self.action_size

    def train(self, state, action, reward, state_next, done):
        action_next = self.actor_target(state_next)
        Q_next = self.critic_target(state_next, action_next)
        Q_target = reward + (1-done)*self.discount_factor*Q_next
        Q = self.critic(state, action) 

        loss_critic = self.loss_function(Q, Q_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        pred_a_sample = self.actor(state)
        loss_actor = -1*(self.critic(state, pred_a_sample)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau) 

        return [loss_critic.mean().detach().cpu(), loss_actor.mean().detach().cpu()]   