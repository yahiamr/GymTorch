import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human',desc=generate_random_map(size=8))

for _ in range(5):
    state = env.reset()
    for _ in range(100):
        action = env.action_space.sample()  # Select a random action
        state, reward, done,_, info = env.step(action)
        env.render()
        if done:
            break

env.close()


