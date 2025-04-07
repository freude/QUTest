# qutest

The `qutest` framework facilitates writing small and readable unit tests for quantum software. 
The `qutest`  framework does not require additional quantum resources beyond those already needed by the application and utilizes quantum tomography or statistical tests on the data obtained from measurements.


## Get started

### Install from source

```bash
git clone https://gitlab.com/freude1/qutest.git
cd qutest
pip install -r requirements.txt
pip install .
```

### Create your first test


```python
import numpy as np
import qiskit
import qutest
from qiskit_aer import AerSimulator


def quantum_subprogram(circuit):
    """Tested quantum subroutine -
       random number generator with uniform distribution"""
    circuit.rx(np.pi / 2, 0)
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
MyTests(backend=AerSimulator(), shots=2000).run(quantum_subprogram)
```

This code should produce an output similar to the following:

```
[PASSED]: R^2 score s is 0.9999730007829774
[FAILED]: R^2 score is 0.4034846478435792
[FAILED]: R^2 score is 0.25000037500000005
```