import os 
import random 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple

learning_rate = 5e - 4
minibatch = 150 
gamma = 0.99
replay_buffer_size = 100,000
interpolation_parameter = 1e -3
number_episodes = 5000
max_time_steps = 1000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_value = 0.995
scores_100_episodes = deque(maxlen = 100)

class ANN(nn.module):
    def __init(self, state_size, action_size, seed=42):
        super(ANN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def foward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
    
class ReplayMemory(self, state):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones

class Agent():
    def __init__(self, state_size, action_size):
        self.state_size =state_size
        self.action_size = action_size
        self.local_qnetwork = ANN(state_size, action_size)
        self.target_qnetwork = ANN(state_size, action_size)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr= learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch:
                experiences = self.memory.memory(minibatch)
                self.learn(experiences, gamma)
    
    def get_action(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0)