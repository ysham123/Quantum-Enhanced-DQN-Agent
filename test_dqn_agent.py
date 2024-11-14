import numpy as np
import torch
from dqn_agent import DQNAgent
from quantum_environment import QuantumEnvironment

# Initialize Quantum Environment and DQN Agent
env = QuantumEnvironment(num_qubits=2)
state_dim = len(env.reset())
action_dim = 3  # We have 3 possible actions (X gate, H gate, RZ gate)
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

# Test the DQN Agent
print("Starting testing of DQN Agent with Quantum Environment and Replay Buffer...")

num_episodes = 5
for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Select action using the DQN agent
        action = agent.select_action(state)

        # Execute action in the environment
        next_state, reward, done = env.step(action)

        # Store the experience in the replay buffer
        agent.add_experience(state, action, reward, next_state, done)

        # Train the agent
        agent.train()

        # Update state and accumulate reward
        state = next_state
        total_reward += reward

    print(f"Episode {episode} - Total Reward: {total_reward}")

print("Testing completed successfully!")
