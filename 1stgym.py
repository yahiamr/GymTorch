import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Generate a random map for the FrozenLake environment
size = 8  # Define the size of the map
random_map = generate_random_map(size=size)

# Create the environment with the generated map
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human', desc=random_map)

for episode in range(5):
    state = env.reset()
    done = False
    step = 0

    while not done and step < 100:
        action = env.action_space.sample()  # Select a random action
        state, reward, done, _, info = env.step(action)  # Adapted for potential extra value
        env.render()
        step += 1

        if done:
            print(f"Episode ended with reward: {reward}")
            break

env.close()