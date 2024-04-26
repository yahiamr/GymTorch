import random  # Import the random module for generating random numbers.
import torch  # Import PyTorch, a library for tensor computations and building neural networks.
import torch.nn as nn  # Import the neural network component from PyTorch.
import torch.optim as optim  # Import the optimization algorithms from PyTorch.
import numpy as np  # Import NumPy for numerical operations.
from collections import deque, namedtuple  # Import deque for a double-ended queue and namedtuple for creating tuple subclasses with named fields.

# Definition of the Deep Q-Network class, which extends the PyTorch Module class.
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=50):
        """Initialize the DQN with specified input size, output size, and hidden layer size."""
        super(DQN, self).__init__()  # Initialize the superclass (nn.Module).
        # The neural network consists of two linear layers separated by a ReLU activation.
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),  # Linear layer from state to hidden layer.
            nn.ReLU(),  # Activation function to introduce non-linearity.
            nn.Linear(hidden_size, action_size)  # Linear layer from hidden layer to output actions.
        )

    def forward(self, x):
        """Define the forward pass through the network."""
        return self.network(x)  # Pass input x through the network and return the output.

# Agent class handles the interaction with the environment and the learning process.
class Agent:
    def __init__(self, state_size, action_size, 
        hidden_size=50, 
        batch_size=128, 
        gamma=0.99, 
        eps_start=0.9, 
        eps_end=0.1, 
        eps_decay=200, 
        target_update=10, 
        lr=1e-3):
        """Initialize the agent with the environment information and hyperparameters."""
        self.state_size = state_size  # Number of state inputs.
        self.action_size = action_size  # Number of possible actions.
        self.hidden_size = hidden_size  # Number of units in the hidden layer.
        self.batch_size = batch_size  # Number of experiences to sample from memory.
        self.gamma = gamma  # Discount factor for future rewards.
        self.eps_start = eps_start  # Initial value for epsilon in the epsilon-greedy policy.
        self.eps_end = eps_end  # Minimum value of epsilon after decay.
        self.eps_decay = eps_decay  # Rate at which epsilon decays.
        self.target_update = target_update  # Frequency of updates to the target network.
        self.lr = lr  # Learning rate for the optimizer.

        self.policy_net = DQN(state_size, action_size, hidden_size)  # Neural network for selecting actions.
        self.target_net = DQN(state_size, action_size, hidden_size)  # Clone of the policy network that lags behind it (for stable learning).
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Copy weights and biases from policy_net to target_net.
        self.target_net.eval()  # Set the target network to evaluation mode (no training).

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)  # Optimizer for the policy network.
        self.memory = deque(maxlen=10000)  # A buffer for storing transitions.
        self.steps_done = 0  # Counter for the number of steps completed.
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))  # A tuple for storing experience data.

    def select_action(self, state):
        """Selects an action using epsilon-greedy policy."""
        sample = np.random.rand()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
    
        if sample > eps_threshold:
            with torch.no_grad():
                # Here, the network is in evaluation mode and torch.no_grad() is used to
                # turn off gradients computation to speed up this forward pass.
                state = torch.tensor([state], dtype=torch.float32)  # Convert state to tensor
                self.policy_net.eval()  # Set the network to evaluation mode
                action_values = self.policy_net(state)
                self.policy_net.train()  # Set the network back to training mode
                return torch.argmax(action_values, dim=1).view(1, 1)  # Return the action with the highest value
        else:
            # Return a random action
            return torch.tensor([[np.random.choice(self.action_size)]], dtype=torch.long)
    
    def store_transition(self, state, action, next_state, reward):
        """Stores a transition in memory."""
        # Convert state, action, next_state, and reward to appropriate PyTorch tensors
        # Note: This conversion is necessary if your environment's outputs are not already tensors.
        # You may need to adjust the data types based on your specific environment and model requirements.
        state = torch.tensor([state], dtype=torch.float)
        action = torch.tensor([action], dtype=torch.long)
        next_state = torch.tensor([next_state], dtype=torch.float)
        reward = torch.tensor([reward], dtype=torch.float)
    
        # Store the transition in the replay buffer
        self.memory.append(self.Transition(state, action, next_state, reward))
    
    def sample_batch(self):
        """Randomly samples a batch of transitions from memory."""
        # Ensure that we have enough samples in the memory for a batch
        if len(self.memory) < self.batch_size:
            return None
        
        # Randomly sample a batch of transitions from the memory
        transitions = random.sample(self.memory, self.batch_size)
        
        # The following line transposes the batch (a batch of transitions)
        # to a transition of batches. This is a common idiom to convert a list
        # of tuples into a tuple of lists in a very efficient way.
        batch = self.Transition(*zip(*transitions))
        
        # Convert batch of transitions to separate batches of states, actions,
        # next states, and rewards.
        # Note: torch.cat is used to concatenate a list of tensors into a single tensor.
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        next_states = torch.cat(batch.next_state)
        rewards = torch.cat(batch.reward)
        
        return states, actions, next_states, rewards

    import torch.nn.functional as F

    def learn(self):
        """Update the model by learning from a batch of transitions."""
        # Sample a batch of transitions
        transitions = self.sample_batch()
        if transitions is None:
            return  # Not enough samples to learn
        
        states, actions, next_states, rewards = transitions
    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(states).gather(1, actions)
    
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non-final states are computed based
        # on the older target_net; selecting their best reward with max(1)[0].
        # We don't want to compute the gradient for this operation, so we use detach().
        next_state_values = torch.zeros(self.batch_size)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_states if s is not None])
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + rewards
    
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def update_target_net(self):
        """Update the target network with the current policy network's weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize_model(self):
        """Perform a single step of the optimization (update) for the policy network."""
        # Check if we have enough samples in the memory to sample a batch
        if len(self.memory) < self.batch_size:
            return  # Skip if not enough samples
    
        # Sample a batch of transitions
        transitions = self.sample_batch()
        if transitions is None:
            return  # Not enough samples to learn from
    
        states, actions, next_states, rewards = transitions
    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(states).gather(1, actions)
    
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non-final states are computed based
        # on the older target_net; selecting their best reward with max(1)[0].
        # We don't want to compute the gradient for this operation, so we use detach().
        next_state_values = torch.zeros(self.batch_size)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_states if s is not None])
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + rewards
    
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, num_episodes):
        """Train the agent over a specified number of episodes."""
        for episode in range(num_episodes):
            # Reset the environment and state
            state = env.reset()
            state = torch.tensor([state], dtype=torch.float)
            total_reward = 0

            for timestep in count():
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action.item())
                total_reward += reward
                reward = torch.tensor([reward], dtype=torch.float)

                if not done:
                    next_state = torch.tensor([next_state], dtype=torch.float)
                else:
                    next_state = None  # Next state is None if the episode is done

                # Store the transition in memory
                self.store_transition(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization
                self.optimize_model()

                if done:
                    break

            # Update the target network, copying all weights and biases in DQN
            if episode % self.target_update == 0:
                self.update_target_net()

            # Decrement epsilon
            self.epsilon = max(self.eps_end, self.eps_decay * self.epsilon)  # Adjust epsilon according to your decay strategy

            print(f'Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}')
