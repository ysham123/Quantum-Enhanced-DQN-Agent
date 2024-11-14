import numpy as np
import matplotlib.pyplot as plt

def smooth(data, window_size=10):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

try:
    rewards = np.load("rewards.npy")
    losses = np.load("losses.npy")
    epsilons = np.load("epsilons.npy")
    print("Data loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

smoothed_rewards = smooth(rewards)
smoothed_losses = smooth(losses) if len(losses) > 1 else losses
smoothed_epsilons = smooth(epsilons)

plt.figure(figsize=(15, 12))

plt.subplot(3, 1, 1)
plt.plot(smoothed_rewards, marker='o', color='b', label='Smoothed Total Reward per Episode')
plt.title("Enhanced Quantum DQN Agent Visualization")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.xticks(range(len(smoothed_rewards)), rotation=45, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(3, 1, 2)
if len(smoothed_losses) > 0:
    plt.plot(smoothed_losses, marker='x', color='r', label='Smoothed Loss per Step')
    plt.title("Training Loss over Time")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
else:
    plt.text(0.5, 0.5, 'Minimal loss data available', fontsize=14, ha='center', va='center')
    plt.title("Training Loss over Time")

plt.subplot(3, 1, 3)
plt.plot(smoothed_epsilons, marker='s', color='g', label='Smoothed Epsilon Decay')
plt.title("Exploration Rate Decay")
plt.xlabel("Training Step")
plt.ylabel("Epsilon (Exploration Rate)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout(pad=2.0)
plt.suptitle("Enhanced Quantum DQN Agent Visualization", fontsize=16)
plt.subplots_adjust(top=0.93)
plt.savefig("enhanced_quantum_dqn_performance.png")
plt.show()
