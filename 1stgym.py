import gymnasium as gym

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')

state = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Select a random action
    state, reward, done,_, info = env.step(action)
    env.render()
    if done:
        break

env.close()


