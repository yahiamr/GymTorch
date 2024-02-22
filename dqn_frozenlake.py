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

#select action 
    def select_action(state, policy_net, epsilon, n_actions):
        if np.random.rand() > epsilon:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.choice(n_actions)]], dtype=torch.long)
# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
n_actions = env.action_space.n

    def extract_tensors(transitions):
        # Convert batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        return states, actions, rewards
        
    def compute_q_values(states, actions):
    return policy_net(states).gather(1, actions.unsqueeze(-1))



