# Q-Learning Training Loop Explanation

This document provides a detailed explanation of the Q-learning training loop used for training an agent in the FrozenLake environment from Gymnasium. The training loop is designed to enable the agent to learn an optimal policy for navigating the environment based on the rewards it receives for its actions.

## Hyperparameters

Before diving into the training loop, several key hyperparameters are defined, which control the learning process:

- `total_episodes`: The total number of episodes for training. Each episode is a sequence of steps starting from a random state and ending when a terminal state is reached or the maximum number of steps is exceeded.
- `learning_rate` (α): Determines how much new information overrides old information. A higher learning rate means the agent quickly adopts new information while discarding old information.
- `max_steps`: The maximum number of steps the agent is allowed to take in a single episode.
- `gamma` (γ): The discount factor, which balances immediate and future rewards. A higher gamma makes the agent prioritize future rewards more strongly.
- `epsilon`: Used for the ε-greedy strategy to balance exploration and exploitation. Initially set to `max_epsilon` and decays over time to `min_epsilon`.
- `max_epsilon`: The initial value of epsilon, favoring exploration at the beginning of training.
- `min_epsilon`: The minimum value epsilon can decay to, ensuring some level of exploration throughout training.
- `decay_rate`: The rate at which epsilon decays from `max_epsilon` to `min_epsilon`.

## Training Loop

```python
for episode in range(total_episodes):
    state, info = env.reset()
    done = False

	•	The loop starts by iterating over each episode. For each episode, the environment is reset to an initial state, and done is set to False to indicate the episode is ongoing.

    for step in range(max_steps):

	•	For each episode, an inner loop iterates up to max_steps times or until the episode ends. Each iteration represents a step the agent takes in the environment.

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q_table[state, :])  # Exploit

	•	At each step, the agent decides whether to explore or exploit based on ε-greedy policy. If a randomly chosen value is less than epsilon, it explores by taking a random action. Otherwise, it exploits its current knowledge by choosing the action with the highest Q-value for the current state.

        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

	•	The agent takes the chosen action, and the environment returns the new state, reward, and whether the episode has terminated or been truncated. done is set to True if the episode is over (either terminated or truncated).

        Q_table[state, action] = Q_table[state, action] + learning_rate * (reward + gamma * np.max(Q_table[new_state, :]) - Q_table[state, action])

	•	The Q-table is updated using the Q-learning formula. This update adjusts the Q-value for the taken action at the current state towards the reward received plus the discounted maximum future reward expected from the new state.

        state = new_state

	•	The agent’s current state is updated to the new state, preparing it for the next step in the episode.

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

	•	After each episode, epsilon is decayed according to the specified decay_rate, gradually shifting the policy from exploration towards exploitation.

The training loop continues until all episodes have been completed. The resulting Q-table contains the learned Q-values, representing the estimated maximum expected future rewards for each state-action pair, which the agent can use to navigate the environment.

## Conclusion

By following this training loop, the agent incrementally learns an optimal policy for the given environment, balancing exploration of new actions with exploitation of known rewards. The hyperparameters play a crucial role in guiding the learning process and must be carefully tuned to achieve the best performance.