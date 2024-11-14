from dqn_agent import DQNAgent
from quantum_environment import QuantumCircuitEnv

def main():
    # Initialize the quantum circuit optimization environment
    env = QuantumCircuitEnv()

    # Initialize the DQN Agent
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    

if __name__ == "__main__":
    main()
