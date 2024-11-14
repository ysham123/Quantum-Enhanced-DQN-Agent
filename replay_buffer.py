import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, max_capacity=10000):
        # Here we are going to initialize the replay buffer with a maximum capacity
        self.max_capacity = max_capacity
        self.buffer = deque(maxlen=self.max_capacity)

    def add(self, state, action, reward, next_state, done):
        # Adding a new experience to the buffer
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size=64):
        # Check if the buffer has enough experiences to sample
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in the buffer to draw the requested batch size.")

        # Sampling a random batch of experiences from the buffer
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Converting to NumPy arrays for efficient processing
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        return states, actions, rewards, next_states, dones

    def size(self):
        # Return the current size of the buffer
        return len(self.buffer)
