import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Set numpy seed for reproducibility (ensure it matches the seed used during training)
np.random.seed(42)

# Environment setup with the same seed as training to generate the same map
size = 4
random_map = generate_random_map(size=size, seed=42)  # Ensure the seed matches the one used during training
env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=True, render_mode="human")
# Load the trained Q-table
Q_table = np.load('q_table.npy')

# Run the environment with the loaded Q-table
num_episodes = 10  # Number of episodes to run
max_steps = 99  # Maximum number of steps per episode
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    for step in range(max_steps):
        # Select action using the loaded Q-table
        action = np.argmax(Q_table[state, :])

        # Take action and observe the outcome
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Accumulate rewards
        total_reward += reward

        # Transition to new state
        state = new_state

        # Break the loop if the episode is terminated or truncated
        if done:
            break

    print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()