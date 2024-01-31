import gym

env = gym.make('FrozenLake-v1', is_slippery=False)  # Non-slippery version for simplicity
state_size = env.observation_space.n
action_size = env.action_space.n