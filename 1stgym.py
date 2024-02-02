import gym

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')
state_size = env.observation_space.n
action_size = env.action_space.n

for episode in range(5):  # Run 5 episodes
    state = env.reset()  # Reset the environment to the starting state
    done = False
    print(f"*** Episode {episode+1} ***\n\n\n\n")
    step = 0
    total_reward = 0  # Track total reward in the episode

    while not done:
        env.render()  # Display the current environment state
        action = env.action_space.sample()  # Take a random action
        state, reward, done,_extra, info = env.step(action)  # Apply the action
        total_reward += reward
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
        
        step += 1
        if done:
            if total_reward > 0:
                print("\n\nReached the goal!")
            else:
                print("\n\nFell into a hole.")
            break

    print(f"Episode finished after {step} steps with total reward: {total_reward}\n")
