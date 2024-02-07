import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Set numpy seed for reproducibility (ensure it matches the seed used during training)
np.random.seed(42)

# Environment setup with the same seed as training to generate the same map
size = 4
random_map = generate_random_map(size=size, seed=42)  # Ensure the seed matches the one used during training
env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=True, render_mode=None)
