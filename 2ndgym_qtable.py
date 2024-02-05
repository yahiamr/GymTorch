import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Set seeds
np.random.seed(42)  # Set numpy seed for reproducibility
env_seed = 42  # Define a seed for the environment

# Environment setup with seed
size = 4
random_map = generate_random_map(size=size, seed=env_seed)  # Generate a reproducible random map
env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=True, render_mode=None)

# Q-table initialization
state_size = env.observation_space.n
action_size = env.action_space.n
Q_table = np.zeros((state_size, action_size))


# Hyperparameters
total_episodes = 100000      # Total episodes for training
learning_rate = 0.8          # Learning rate
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate
epsilon = 1.0                # Exploration rate
max_epsilon = 1.0            # Exploration probability at start
min_epsilon = 0.01           # Minimum exploration probability
decay_rate = 0.005           # Exponential decay rate for exploration probability

for episode in range(total_episodes):
    state , info= env.reset()

    done = False
    #inner loop for single episode
    for step in range(max_steps):
         # Exploration-exploitation trade-off
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(Q_table[state, :])  # Exploit learned values


       # Take action and observe the outcome
        new_state, reward, terminated, truncated, info = env.step(action)
        # Determine if the episode is done (either terminated or truncated)
        done = terminated or truncated
        # Update Q-table
        Q_table[state, action] = Q_table[state, action] + learning_rate * (reward + gamma * np.max(Q_table[new_state, :]) - Q_table[state, action])

        state = new_state

        if done:
            break
 # Reduce epsilon (less exploration over time)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

# Print the trained Q-table
print(Q_table)
np.save('q_table.npy', Q_table)  # Saves the Q-table to 'q_table.npy'
