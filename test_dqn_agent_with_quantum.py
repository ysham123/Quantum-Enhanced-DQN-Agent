import numpy as np
from dqn_agent import DQNAgent
from quantum_environment import QuantumEnvironment

# Initialize the quantum environment and DQN agent
env = QuantumEnvironment(num_qubits=2)
agent = DQNAgent(state_dim=4, action_dim=3)

# Training parameters
num_episodes = 5
total_rewards = []
losses = []
epsilons = []

print("Starting testing of DQN Agent with Quantum Layer and Quantum Environment...")

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.add_experience(state, action, reward, next_state, done)

        # Store experience and update the total reward
        total_reward += reward
        state = next_state

        # Train the agent
        loss = agent.train()

        # Record the loss and epsilon values
        if loss is not None and not np.isnan(loss):
            losses.append(loss)
        epsilons.append(agent.epsilon)

        if done:
            break

    total_rewards.append(total_reward)
    print(f"Episode {episode} - Total Reward: {total_reward}")

# Save training data for visualization
np.save("rewards.npy", total_rewards)
np.save("losses.npy", losses)
np.save("epsilons.npy", epsilons)

print("Training completed successfully and data saved!")
