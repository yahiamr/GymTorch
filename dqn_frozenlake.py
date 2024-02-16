import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=50):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.network(x)
env = gym.make('FrozenLake-v1', is_slippery=True)
replay_buffer = deque(maxlen=10000)