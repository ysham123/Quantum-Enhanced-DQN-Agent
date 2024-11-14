from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

class QuantumEnvironment:
    def __init__(self, num_qubits=2):
        # Initialize the quantum environment with the given number of qubits
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(self.num_qubits)
        self.state = None  # This holds the state representation
        self.max_steps = 100  # This represents the maximum steps per episode
        self.current_step = 0

        # Backend for the simulation using AerSimulator
        self.simulator = AerSimulator()

    def reset(self):
        # Reset the quantum circuit and the step counter
        self.circuit = QuantumCircuit(self.num_qubits)
        self.circuit.save_statevector()  # Save the statevector only once
        self.current_step = 0

        # Get the initial state (statevector or simple representation)
        self.state = self.get_state()
        return self.state

    def get_state(self):
        # Transpile the circuit using the qiskit.transpile function
        transpiled_circuit = transpile(self.circuit, self.simulator)

        # Run the simulation and get the result
        job = self.simulator.run(transpiled_circuit)
        result = job.result()

        # Extract the statevector from the result
        try:
            statevector = result.get_statevector()
        except Exception as e:
            print(f"Error in retrieving statevector: {e}")
            statevector = np.zeros(2 ** self.num_qubits)

        # Return the real part of the statevector as the state representation
        return np.real(statevector)

    def step(self, action):
        # Increment the step counter
        self.current_step += 1

        # Apply the action to the quantum circuit
        self.apply_action(action)

        # Get the new state
        new_state = self.get_state()

        # Calculate the reward
        reward = self.calculate_reward()

        # Check if the episode is done
        done = self.current_step >= self.max_steps

        return new_state, reward, done

    def apply_action(self, action):
        # Define possible actions: 0 -> Apply X gate, 1 -> Apply H gate, 2 -> Apply RZ gate
        if action == 0:
            self.circuit.x(0)  # Apply X gate to the first qubit
        elif action == 1:
            self.circuit.h(0)  # Apply H gate to the first qubit
        elif action == 2:
            self.circuit.rz(np.pi / 4, 0)  # Apply RZ gate with π/4 rotation to the first qubit

    def calculate_reward(self):
        # For simple measures, use the fidelity of the state as the reward
        target_state = np.array([1, 0, 0, 0])  # Example target state (|00> state)
        current_state = self.get_state()

        # Calculate the fidelity as the reward
        try:
            fidelity = np.dot(target_state, current_state) ** 2
        except Exception as e:
            print(f"Error in reward calculation: {e}")
            fidelity = 0.0

        return fidelity
