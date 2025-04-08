import numpy as np
import qiskit
import qutest
from qiskit_aer import AerSimulator


def quantum_subprogram1(circuit):
    """Tested quantum subroutine -
       random number generator with uniform distribution"""

    circuit.rx(np.pi / 2, 0)
    return circuit


def quantum_subprogram2(circuit):
    """Tested quantum subroutine -
       random number generator with uniform distribution"""

    circuit.rx(np.pi / 4, 0)
    return circuit


def quantum_subprogram3(circuit):
    """Tested quantum subroutine -
       random number generator with uniform distribution"""

    circuit.rx(np.pi, 0)
    return circuit


class MyTests(qutest.QUT_PROJ):
    """Class prepares environment for a quantum unit test
       based on the protocol QUT_PROJ"""

    def pre(self):
        """Prepare state for the input quantum register"""
        circuit = qiskit.QuantumCircuit(1)
        return circuit

    def post(self):
        """Prepare expected output"""
        return np.array([0.5, 0.5]) * self.shots


# run tests
MyTests(backend=AerSimulator(), shots=2000).run(quantum_subprogram1)
MyTests(backend=AerSimulator(), shots=2000).run(quantum_subprogram2)
MyTests(backend=AerSimulator(), shots=2000).run(quantum_subprogram3)