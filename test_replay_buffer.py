from replay_buffer import ReplayBuffer
import numpy as np

# Initialize the replay buffer
buffer = ReplayBuffer(max_capacity=100)

# Create some dummy data for testing
state = np.array([1, 0, 0, 0])
next_state = np.array([0, 1, 0, 0])
action = 0
reward = 1.0
done = False

# Add experiences to the buffer
print("Adding experiences to the replay buffer...")
for i in range(10):
    buffer.add(state, action, reward, next_state, done)
    print(f"Added experience {i + 1}, current buffer size: {buffer.size()}")

# Check the size of the buffer
print(f"Current buffer size: {buffer.size()}")

# Try sampling a batch
try:
    batch_size = 5
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    print(f"Sampled a batch of size {batch_size}:")
    print("States:", states)
    print("Actions:", actions)
    print("Rewards:", rewards)
    print("Next States:", next_states)
    print("Dones:", dones)
except ValueError as e:
    print("Error:", e)

# Test sampling when the buffer is not large enough
try:
    print("Testing sampling with a larger batch size than available...")
    buffer.sample(50)
except ValueError as e:
    print("Expected Error:", e)
