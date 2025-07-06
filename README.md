# Quantum-Enhanced DQN Agent

A Deep Q-Network (DQN) reinforcement learning agent enhanced with quantum computing capabilities for quantum circuit optimization tasks.

## Overview

This project implements a hybrid classical-quantum reinforcement learning system that combines traditional Deep Q-Networks with quantum neural networks to solve quantum circuit optimization problems. The agent learns to apply quantum gates (X, H, RZ) to optimize quantum circuit states towards target configurations.

## Features

- **Hybrid Classical-Quantum Architecture**: Combines classical neural networks with quantum layers
- **Quantum Environment**: Simulates quantum circuits using Qiskit
- **Experience Replay**: Implements replay buffer for stable training
- **Target Network**: Uses separate target network for stable Q-value estimation
- **Epsilon-Greedy Exploration**: Balanced exploration vs exploitation strategy
- **Visualization Tools**: Built-in plotting and performance analysis

## Project Structure

```
Quantum-Enhanced-DQN-Agent/
├── main.py                          # Main entry point
├── dqn_agent.py                     # DQN agent implementation
├── quantum_environment.py           # Quantum circuit environment
├── quantum_layer.py                 # Quantum neural network layer
├── replay_buffer.py                 # Experience replay buffer
├── test_dqn_agent_with_quantum.py  # Training script
├── plot_results.py                  # Visualization utilities
├── enhanced_visualization.py        # Enhanced plotting features
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ysham123/Quantum-Enhanced-DQN-Agent
   cd Quantum-Enhanced-DQN-Agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- **qiskit**: Quantum computing framework
- **torch**: PyTorch for deep learning
- **gym**: Reinforcement learning environments
- **matplotlib**: Data visualization
- **numpy**: Numerical computations

## Usage

### Basic Training

Run the main training script:

```bash
python test_dqn_agent_with_quantum.py
```

This will:
- Initialize a 2-qubit quantum environment
- Train the DQN agent for 5 episodes
- Save training data (rewards, losses, epsilon values)
- Display episode results

### Custom Training

You can modify the training parameters in `test_dqn_agent_with_quantum.py`:

```python
# Training parameters
num_episodes = 100  # Increase for longer training
env = QuantumEnvironment(num_qubits=3)  # Use more qubits
agent = DQNAgent(state_dim=8, action_dim=3)  # Adjust dimensions
```

### Visualization

After training, visualize the results:

```bash
python plot_results.py
```

Or use the enhanced visualization:

```bash
python enhanced_visualization.py
```

## Architecture

### DQN Agent (`dqn_agent.py`)

The DQN agent implements:
- **Neural Network**: Classical layers + quantum layer
- **Experience Replay**: Buffer for storing transitions
- **Target Network**: Separate network for stable learning
- **Epsilon-Greedy**: Exploration strategy

### Quantum Environment (`quantum_environment.py`)

Simulates a quantum circuit environment:
- **State Representation**: Quantum statevector
- **Actions**: Apply X, H, or RZ gates
- **Reward Function**: Fidelity to target state
- **Episode Management**: Step limits and termination

### Quantum Layer (`quantum_layer.py`)

Implements quantum neural network components:
- **Variational Quantum Circuit**: Parameterized quantum gates
- **Feature Encoding**: Classical to quantum data mapping
- **Expectation Values**: Quantum measurement simulation

## Key Components

### State Space
- Quantum statevector representation
- Real-valued components from quantum circuit simulation

### Action Space
- **Action 0**: Apply X gate to first qubit
- **Action 1**: Apply H gate to first qubit  
- **Action 2**: Apply RZ gate (π/4 rotation) to first qubit

### Reward Function
- Fidelity between current state and target state |00⟩
- Higher fidelity = higher reward

## Training Process

1. **Environment Reset**: Initialize quantum circuit
2. **State Observation**: Get current quantum state
3. **Action Selection**: Epsilon-greedy policy
4. **Environment Step**: Apply quantum gate
5. **Reward Calculation**: Compute fidelity-based reward
6. **Experience Storage**: Add to replay buffer
7. **Network Training**: Update Q-network with batch of experiences
8. **Target Update**: Periodically update target network

## Performance Metrics

The training process tracks:
- **Episode Rewards**: Total reward per episode
- **Training Loss**: Q-value prediction error
- **Epsilon Decay**: Exploration rate over time
- **Convergence**: Learning stability indicators

## Customization

### Environment Modifications

- **Number of Qubits**: Modify `num_qubits` parameter
- **Target State**: Change target state in `calculate_reward()`
- **Available Gates**: Add new quantum gates in `apply_action()`

### Agent Hyperparameters

- **Learning Rate**: Adjust `lr` in DQNAgent initialization
- **Discount Factor**: Modify `gamma` for future reward weighting
- **Exploration**: Tune `epsilon_decay` and `epsilon_min`
- **Network Architecture**: Modify `build_network()` method

## Testing

Run the test suite to verify functionality:

```bash
python test_dqn_agent.py
python test_quantum_environment.py
python test_replay_buffer.py
```

## Results

The agent learns to:
- Apply appropriate quantum gates to reach target states
- Optimize circuit configurations for maximum fidelity
- Balance exploration and exploitation during training
- Converge to stable policy over multiple episodes

## Future Enhancements

- **Multi-qubit Operations**: Support for multi-qubit gates
- **Advanced Quantum Circuits**: More complex circuit architectures
- **Quantum Advantage**: Leverage quantum hardware backends
- **Multi-objective Optimization**: Multiple target states
- **Transfer Learning**: Pre-trained models for new tasks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{quantum_enhanced_dqn,
  title={Quantum-Enhanced DQN Agent},
  author={Yosef Shammout},
  year={2024},
  url={https://github.com/ysham123/Quantum-Enhanced-DQN-Agent}
}
```

## Contact

For questions or contributions, please open an issue on the repository or contact Yosefshammout123@gmail.com.
