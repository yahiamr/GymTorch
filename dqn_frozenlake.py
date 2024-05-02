import numpy as np  # Importing the NumPy library for numerical operations.
import torch  # Importing PyTorch for tensor operations and deep learning.
import torch.nn as nn  # Importing the neural network blocks from PyTorch.
import torch.optim as optim  # Importing PyTorch's optimization algorithms.
import gymnasium as gym  # Importing the Gymnasium library to create and manage various environments.
from collections import deque  # Importing deque, a list-like container with fast appends and pops on either end.

# Definition of the DQN class, inheriting from nn.Module for neural network functionality.
class DQN(nn.Module):
    # Constructor for the DQN class, with parameters for state and action dimensions, and hidden layer size.
    def __init__(self, state_size, action_size, hidden_size=50):
        super(DQN, self).__init__()  # Initialize the parent class (nn.Module).
        self.network = nn.Sequential(  # Defining a sequential container of layers.
            nn.Linear(state_size, hidden_size),  # Linear layer from state_size to hidden_size.
            nn.ReLU(),  # ReLU activation function for non-linearity.
            nn.Linear(hidden_size, action_size)  # Linear layer from hidden_size to action_size.
        )

    # Forward pass definition for the DQN network.
    def forward(self, x):
        return self.network(x)  # Pass the input tensor x through the network.

    # Method to select an action based on the current state, policy network, epsilon, and number of actions.
    def select_action(state, policy_net, epsilon, n_actions):
        if np.random.rand() > epsilon:  # With probability 1 - epsilon, choose the best action.
            with torch.no_grad():  # Temporarily set all the required_grad flag to false.
                return policy_net(state).max(1)[1].view(1, 1)  # Get the action with the max Q-value.
        else:  # With probability epsilon, choose a random action.
            return torch.tensor([[np.random.choice(n_actions)]], dtype=torch.long)  # Return a random action.
        
    # Method to extract tensors from a batch of transitions.
    def extract_tensors(transitions):
        # Convert batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))  # Unzip transitions into separate arrays.
        states = torch.cat(batch.state)  # Concatenate state tensors.
        actions = torch.cat(batch.action)  # Concatenate action tensors.
        rewards = torch.cat(batch.reward)  # Concatenate reward tensors.
        return states, actions, rewards  # Return the separate tensors.
        
    # Compute Q values for current states using the policy network.
    def compute_q_values(states, actions):
        return policy_net(states).gather(1, actions.unsqueeze(-1))  # Gather Q-values corresponding to taken actions.
    
    # Compute Q values for next states using the target network.
    def compute_next_state_values(non_final_next_states):
        return target_net(non_final_next_states).max(1)[0].detach()  # Get max Q-value and detach from graph.

    # Calculate the expected Q values from next state values and rewards.
    def compute_expected_q_values(next_state_values, rewards):
        return (next_state_values * GAMMA) + rewards  # Apply the Bellman equation.

    # Compute the loss using Smooth L1 loss between the expected Q values and the computed Q values.
    def compute_loss(state_action_values, expected_state_action_values):
        return F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(-1))

# Setting hyperparameters for training.
BATCH_SIZE = 128
GAMMA = 0.999  # Discount factor for future rewards.
EPS_START = 0.9  # Starting value of epsilon for the epsilon-greedy strategy.
EPS_END = 0.05  # Minimum value of epsilon after decay.
EPS_DECAY = 250  # Rate of decay for epsilon.
TARGET_UPDATE = 8  # How frequently to update the target network.

# Initializing the environment.
env = gym.make('FrozenLake-v1', is_slippery=True)  # Create the FrozenLake environment, slippery version.
replay_buffer = deque(maxlen=10000)  # Initialize a replay buffer for experience replay.
n_actions = env.action_space.n  # Get the number of actions from the environment's action space.