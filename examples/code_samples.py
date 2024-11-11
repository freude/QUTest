import matplotlib.pyplot as plt
import qiskit
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePerth


def quantum_subprogram(circuit):
    # quantum unit

    nq = 3

    circuit.x(0)  # apply Hadamard gate to the first qubit
    circuit.h(0)  # apply phase shift gate to the first qubit

    for i in range(1, nq):  # introduce entanglement via controlled-X gates
        circuit.cx(0, i)

    return circuit


def quantum_subprogram_mut1(circuit):
    # quantum unit

    nq = 3

    circuit.x(0)  # apply Hadamard gate to the first qubit
    circuit.h(1)  # apply phase shift gate to the first qubit

    for i in range(1, nq):  # introduce entanglement via controlled-X gates
        circuit.cx(0, i)

    return circuit


def quantum_subprogram_mut2(circuit):
    # quantum unit

    nq = 3

    circuit.s(0)  # apply Hadamard gate to the first qubit
    circuit.h(0)  # apply phase shift gate to the first qubit

    for i in range(1, nq):  # introduce entanglement via controlled-X gates
        circuit.cx(0, i)

    return circuit


if __name__ == '__main__':
    # choose backend
    backend = AerSimulator.from_backend(FakePerth())

    nq = 3  # number of qubits
    qc = qiskit.QuantumCircuit(nq)  # define the quantum circuit object
    qc = quantum_subprogram(qc)
    qc_meas = qc.measure_all(inplace=False)
    qc_meas.draw('mpl')
    plt.show()

    circ = qiskit.transpile(qc_meas, backend)
    results = backend.run(circ, shots=2000).result()
    counts = results.get_counts(qc_meas)
    print(counts)

    plot_histogram(counts)
    plt.show()
