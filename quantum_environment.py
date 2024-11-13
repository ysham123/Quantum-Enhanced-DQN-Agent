import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import Aer
import numpy as np

class QuantumEnvironment:
    def __init__(self, num_qubits=2):
        # Here we are going to initialize the quantum environment with the given number of qubits
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(self.num_qubits)
        self.state = None  # This holds the state representation
        self.max_steps = 100  # This represents the maximum steps per episode
        self.current_step = 0

        # Backend for the simulation
        self.backend = Aer.get_backend('statevector_simulator')

    def reset(self):
        # This is going to reset the quantum circuit and the step counter
        self.circuit = QuantumCircuit(self.num_qubits)
        self.current_step = 0

        # Then we need to return the initial state (which can be the statevector or a simple representation)
        self.state = self.get_state()
        return self.state

    def get_state(self):
        # Execute the circuit and get the state vector
        job = execute(self.circuit, self.backend)
        result = job.result()
        statevector = result.get_statevector()

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
            self.circuit.rz(np.pi / 4, 0)  # Apply RZ gate with pi/4 rotation to the first qubit

    def calculate_reward(self):
        # For simple measures, I'm going to use the fidelity of the state as the reward
        target_state = np.array([1, 0, 0, 0])  # Example target state (|00> state)
        current_state = self.get_state()

        # Calculate the fidelity as the reward using the complex conjugate
        fidelity = np.abs(np.dot(np.conj(target_state), current_state)) ** 2
        return fidelity
