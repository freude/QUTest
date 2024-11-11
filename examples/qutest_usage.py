import qutest
import numpy as np
from code_samples import *
import warnings
warnings.filterwarnings("ignore")

class MyTests(qutest.QUT_ST):

    def pre(self):

        nq = 3  # number of qubits
        precondition = qiskit.QuantumCircuit(nq)

        return precondition

    def post(self):

        expected = np.zeros((8, 8))
        expected[0, 0] = 0.5
        expected[7, 7] = 0.5
        expected[0, 7] = -0.5
        expected[7, 0] = -0.5

        return expected


if __name__ == '__main__':

    aaa = MyTests(backend=AerSimulator())
    aaa.run(quantum_subprogram)

    MyTests(backend=AerSimulator.from_backend(FakePerth())).run(quantum_subprogram)
    MyTests(backend=AerSimulator.from_backend(FakePerth())).run(quantum_subprogram_mut1)
    MyTests(backend=AerSimulator.from_backend(FakePerth())).run(quantum_subprogram_mut2)

