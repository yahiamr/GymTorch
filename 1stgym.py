# Import the Gymnasium library and a specific function for generating random maps.
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Generate a random map for the Frozen Lake environment.
size = 8  # Set the size of the lake map to be 8x8.
random_map = generate_random_map(size=size)  # Generate the map using the specified size.

# Create the Frozen Lake environment using the generated map.
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human', desc=random_map)
# 'FrozenLake-v1' specifies the version of the environment.
# 'is_slippery=True' makes the lake's ice slippery, affecting the agent's movement.
# 'render_mode='human'' sets the rendering mode to human-friendly output.
# 'desc=random_map' uses the generated map for the environment layout.

# Run the agent for 5 episodes.
for episode in range(5):
    state = env.reset()  # Reset the environment to a new, random state.
    done = False  # Initialize the 'done' flag to False indicating the episode is not finished.
    step = 0  # Step counter to prevent infinitely long episodes.

    # Execute steps until the episode ends or reaches 100 steps.
    while not done and step < 100:
        action = env.action_space.sample()  # Randomly sample an action from the action space.
        state, reward, done, _, info = env.step(action)  # Perform the action in the environment.
        env.render()  # Render the current state of the environment.
        step += 1  # Increment the step counter.

        if done:
            # If the episode has ended, print the reward.
            print(f"Episode ended with reward: {reward}")
            break  # Exit the loop since the episode is finished.

env.close()  # Close the environment to clean up resources.