from abc import ABC, abstractmethod
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePerth
from qiskit_experiments.library import StateTomography, ProcessTomography
from qiskit.quantum_info import state_fidelity


class QUT(ABC):

    def __init__(self, **kwargs):

        default_backend = AerSimulator.from_backend(FakePerth())
        self.backend = kwargs.get('backend', default_backend)


    @abstractmethod
    def pre(self):
        pass

    @abstractmethod
    def post(self):
        pass

    @abstractmethod
    def assertEqual(self, arg1, arg2):
        pass

    @abstractmethod
    def run(self, unit):

        qc = self.pre()
        qc = unit(qc)

        res = self.post()

        return self.assertEqual(qc, res)


class QUT_PT(QUT, ABC):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.experiment = ProcessTomography

    @abstractmethod
    def pre(self):
        pass

    @abstractmethod
    def post(self):
        pass

    def assertEqual(self, arg1, arg2):
        pass

    def run(self, prog):
        pass


class QUT_ST(QUT, ABC):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.experiment = StateTomography

    @abstractmethod
    def pre(self):
        pass

    @abstractmethod
    def post(self):
        pass

    def assertEqual(self, arg1, arg2):

        return state_fidelity(arg1, arg2)

    def run(self, prog):

        qc = self.pre()
        qc = prog(qc)

        qstexp = StateTomography(qc)
        qstdata = qstexp.run(self.backend, seed_simulation=100, shots=5000).block_for_results()
        res = qstdata.analysis_results("state").value

        exp = self.post()

        return self.assertEqual(exp, res)


if __name__ == '__main__':

    aaa = QUT_PT()
    aaa.run()
