import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

# Environment setup
size = 8
random_map = generate_random_map(size=size)
env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=True, render_mode=None)

# Q-table initialization
state_size = env.observation_space.n
action_size = env.action_space.n
Q_table = np.zeros((state_size, action_size))


# Hyperparameters
total_episodes = 10000       # Total episodes for training
learning_rate = 0.8          # Learning rate
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate
epsilon = 1.0                # Exploration rate
max_epsilon = 1.0            # Exploration probability at start
min_epsilon = 0.01           # Minimum exploration probability
decay_rate = 0.005           # Exponential decay rate for exploration probability