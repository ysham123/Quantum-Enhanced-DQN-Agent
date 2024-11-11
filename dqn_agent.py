import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # Initialization of the hyperparameters
        self.state_dim = state_dim  # This is the number of features in the dim
        self.action_dim = action_dim  # Number of possible actions
        self.hidden_dim = hidden_dim  # This is the size of the hidden layer in the neural network

        # Exploration parameters
        self.epsilon = epsilon  # This is the initial exploration rate
        self.epsilon_decay = epsilon_decay  # Rate at which the exploration decreases
        self.epsilon_min = epsilon_min  # Minimum exploration rate

        # Learning parameters
        self.lr = lr  # Learning rate for the optimizer
        self.gamma = gamma  # Discount factor for future rewards

        # Experience replay memory
        self.memory = []  # List to store experiences (state, action, reward, next_state, done)
        self.batch_size = 64  # This is the number of experiences to sample during the training

        # Neural Network
        self.q_network = self.build_network()  # Main Q-network
        self.target_network = self.build_network()  # This is the target Q-Network
        self.update_target_network()  # Target network weights

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    def build_network(self):
        # Build the network

        # So here I'm going to define the simple feed-forward neural network
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),  # Input layer
            nn.ReLU(),  # Activation function
            nn.Linear(self.hidden_dim, self.hidden_dim),  # Hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(self.hidden_dim, self.action_dim)  # Output layer
        )

    def update_target_network(self):
        # Copy the weights from the main Q-network to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())
