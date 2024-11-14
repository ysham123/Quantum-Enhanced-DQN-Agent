import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with actual episode rewards from your testing)
episode_rewards = [58.13, 39.54, 38.92, 78.93, 81.92]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, marker='o', linestyle='-', color='b', label='Total Reward per Episode')
plt.title('Quantum DQN Agent Performance')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid()
plt.legend()
plt.show()
