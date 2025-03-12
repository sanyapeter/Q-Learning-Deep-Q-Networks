import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Environment Setup
n_channels = 5
n_episodes = 10000
true_probs = [0.1, 0.4, 0.6, 0.3, 0.9]

# Q-Learning Parameters
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999

# Initialize Q-table for Q-learning
Q_table = np.zeros(n_channels)

# DQN Hyperparameters
alpha_dqn = 0.0005
batch_size = 128
memory_size = 50000
target_update_freq = 20

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Sanya TODO: Define the neural network layers
        pass

    def forward(self, x):
        # Sanya TODO: Define forward propagation
        pass

# Initialize Neural Networks and optimizer for DQN
# Sanya TODO: Complete network initialization and optimizer setup

# Experience Replay Memory for DQN
# Sanya TODO: Define the replay memory

# Lists to store rewards
rewards_qlearning = []
rewards_dqn = []

# Training Loop
for episode in range(n_episodes):
    # Q-Learning (QN)
    # Action selection (epsilon-greedy) for Q-Learning
    random_float = random.uniform(0, 1) # a number in range [0,1)
    chosen_channel = 0 # Default channel is 0
    if random_float <= epsilon:
        # Act randomnly
        chosen_channel = random.randint(0, n_channels-1) # a number in range [0, n-1]
    else:
        # Act on current policy
        current_biggest = Q_table[0] # Default channel is 0
        for i in range(1, n_channels):
            if (Q_table[i] > current_biggest):
                current_biggest = Q_table[i]
                chosen_channel = i
    
    # Calculate the reward on the chosen channel
    random_float2 = random.uniform(0, 1) # a number in range [0,1)
    reward = 0
    if (random_float2 <= true_probs[chosen_channel]):
        reward = 1
    print("Reward on channel", chosen_channel, "is", reward, "for episode #", episode) # Debug
    
    # Update Q-table
    Q_table[chosen_channel] = Q_table[chosen_channel] + alpha * (reward - Q_table[chosen_channel])
   
    # Append reward to rewards_qlearning
    rewards_qlearning.append(reward)

    # DQN
    # Sanya TODO: Implement action selection (epsilon-greedy) for DQN
    # Sanya TODO: Store experiences in replay memory

    # DQN training using replay memory
    # if len(memory) >= batch_size:
        # Sanya TODO: Sample a batch from replay memory
        # Sanya TODO: Compute Q-values and target Q-values
        # Sanya TODO: Compute loss and update DQN
        # pass

    # Sanya TODO: Update target network periodically (based on target_update_freq)

    # Decay epsilon
    epsilon = epsilon * epsilon_decay
    
    # Ensure epsilon doesn't go below the min (want to retain a small amount of randomness)
    if epsilon < epsilon_min:
        epsilon = epsilon_min

    # Sanya TODO: Append reward to rewards_dqn
    # Currently just adding a number between 0-1 so I can make a graph
    rand_int2 = random.randint(0, 1)
    rewards_dqn.append(rand_int2)
    
    
print("After ", n_episodes, " the Qtable has generated the probabilities: ", Q_table) # Debug

# Moving Average Calculation
# Sanya TODO: Complete the moving average function
def moving_average(data, window_size):
    pass

# Plot Learning Curve for Q-Learning vs. DQN
# Compute cumulative win rates
q_learning_cumulative = np.cumsum(rewards_qlearning) / (np.arange(n_episodes) + 1)
random_cumulative = np.cumsum(rewards_dqn) / (np.arange(n_episodes) + 1)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(q_learning_cumulative, label="Q-learning", linewidth=2)
plt.plot(random_cumulative, label="Deep Q-Network", linestyle="dashed", linewidth=2)
plt.xlabel("Episodes")
plt.ylabel("Cumulative Win Rate")
plt.title("Performance of Q-learning vs Deep Q-Network")
plt.legend()
plt.grid(True)
plt.show()
