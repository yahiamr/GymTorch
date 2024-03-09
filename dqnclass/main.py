import gym
import os

if __name__ == "__main__":
    # Initialize the Gym environment
    env = gym.make('CartPole-v1')
    
    # Get the size of state space and action space
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Define hyperparameters
    hidden_size = 128  # Size of the hidden layer in DQN
    batch_size = 64    # Batch size for experience replay
    gamma = 0.99       # Discount factor for future rewards
    eps_start = 0.9    # Starting value of epsilon
    eps_end = 0.05     # Minimum value of epsilon
    eps_decay = 200    # Decay rate of epsilon
    target_update = 10 # How often to update the target network
    num_episodes = 500 # Number of episodes to train on
    lr = 1e-3          # Learning rate
    
    # Initialize the DQN Agent
    agent = Agent(state_size=state_size, action_size=action_size, hidden_size=hidden_size,
                  batch_size=batch_size, gamma=gamma, eps_start=eps_start, eps_end=eps_end,
                  eps_decay=eps_decay, target_update=target_update, lr=lr)
    
    # Train the Agent
    agent.train(num_episodes)
    
    # Close the environment
    env.close()