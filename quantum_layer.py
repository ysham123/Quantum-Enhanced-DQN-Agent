import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator

class QuantumLayer(nn.Module):
    def __init__(self, num_qubits, num_features):
        super(QuantumLayer, self).__init__()
        # Initialize the number of qubits and input features
        self.num_qubits = num_qubits
        self.num_features = num_features

        # Initialize the quantum circuit
        self.qc = self.build_quantum_circuit()

        # Estimator for calculating the expectation value
        self.estimator = Estimator()

    def build_quantum_circuit(self):
        # Building a simple variational quantum circuit
        qc = QuantumCircuit(self.num_qubits)
        
        # Applying Hadamard gates to all qubits to create superposition
        for qubit in range(self.num_qubits):
            qc.h(qubit)
        
        # Parameterized RX and RZ gates for each qubit
        for qubit in range(self.num_qubits):
            qc.rx(np.pi / 4, qubit)
            qc.rz(np.pi / 4, qubit)

        # Adding CNOT gates for entanglement
        for qubit in range(self.num_qubits - 1):
            qc.cx(qubit, qubit + 1)

        return qc

    def forward(self, x):
        # x: Input features, we will use a subset of the input features
        batch_size = x.size(0)

        # Placeholder for quantum circuit outputs
        quantum_outputs = []

        # Process each sample in the batch
        for i in range(batch_size):
            input_features = x[i, :self.num_qubits].detach().numpy()

            # Encode classical features into quantum circuit
            self.encode_features(input_features)

            # Get the expectation value from the quantum circuit
            try:
                expectation_value = self.estimator.run(
                    circuits=self.qc,
                    observables=["Z" * self.num_qubits]
                ).result().values[0]
            except Exception as e:
                print("Quantum Estimator Error:", e)
                expectation_value = 0

            # Append the result
            quantum_outputs.append(expectation_value)

        # Convert to tensor and return
        return torch.tensor(quantum_outputs).float().view(batch_size, 1)

    def encode_features(self, features):
        # Reset the quantum circuit with encoded features
        self.qc = self.build_quantum_circuit()
        for i, feature in enumerate(features):
            self.qc.rx(feature, i)
