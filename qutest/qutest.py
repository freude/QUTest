from abc import ABC, abstractmethod
import time
from colorama import Fore
import numpy as np
import qiskit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePerth, FakeMelbourneV2, FakeSydneyV2
from qiskit_experiments.library import StateTomography, ProcessTomography
from qiskit.quantum_info import state_fidelity, process_fidelity
from qutest.aux_function import make_keys
from scipy.stats import chisquare
from sklearn.metrics import r2_score


SEED = 100
# SEED = 1200


class QUT(ABC):

    def __init__(self, **kwargs):

        default_backend = AerSimulator.from_backend(FakeSydneyV2())
        self.backend = kwargs.get('backend', default_backend)
        self.params = []

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
    def protocol(self, init_circuit, unit):
        pass

    def run(self, unit):

        qc = self.pre()

        try:
            qc = unit(qc, *self.params)
        except TypeError:
            qc = unit(qc)

        res = self.protocol(qc, unit)
        res_expected = self.post()
        self.assertEqual(res, res_expected)


class QUT_PT(QUT, ABC):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.experiment = ProcessTomography
        self.shots = kwargs.get('shots', 2000)
        self.data = {}

    @abstractmethod
    def pre(self):
        pass

    @abstractmethod
    def post(self):
        pass

    def assertEqual(self, arg1, arg2):

        fid = process_fidelity(arg1, arg2, require_tp=False)
        # fid = np.linalg.norm(np.abs(arg1 - arg2.data))
        # fid = np.exp(-fid)
        self.data['fid'] = fid

        if fid > 0.5:
            print(Fore.GREEN + '[PASSED]: The fidelity of two states is {}'.format(fid) + Fore.RESET)
        else:
            print(Fore.RED + '[FAILED]: The fidelity of two states is {}'.format(fid) + Fore.RESET)

    def protocol(self, qc, prog):

        qstexp = self.experiment(qc)
        start = time.time()
        qstdata = qstexp.run(self.backend, seed_simulation=SEED, shots=self.shots).block_for_results()
        res = qstdata.analysis_results("state").value
        end = time.time()
        self.data['time'] = end - start
        self.data['rho'] = res

        return res


class QUT_ST(QUT, ABC):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.experiment = StateTomography
        self.shots = kwargs.get('shots', 2000)
        self.basis_indices = kwargs.get('basis_indices', None)
        self.measurement_indices = kwargs.get('measurement_indices', None)
        self.data = {}
        self.params = []


    @abstractmethod
    def pre(self):
        pass

    @abstractmethod
    def post(self):
        pass

    def assertEqual(self, arg1, arg2):

        fid = state_fidelity(arg1, arg2)

        # import matplotlib.pyplot as plt
        # plt.imshow(np.abs(arg1.data))
        # plt.show()
        # plt.imshow(np.abs(arg2.data))
        # plt.show()

        # fid = np.max(np.abs(arg1 - arg2.data))
        # fid = np.sqrt(np.sum(np.abs(arg1 - arg2.data)**2))
        # fid = np.linalg.norm(np.abs(arg1 - arg2.data))
        # T = 1
        # fid = np.exp(-fid/T)
        # diff = np.matrix(arg1.data - arg2.data)
        # fid = 1.0 - 0.5 * np.trace(sqrtm(diff.H @ diff))
        self.data['fid'] = fid

        if fid > 0.5:
            print(Fore.GREEN + '[PASSED]: The fidelity of two states is {}'.format(fid) + Fore.RESET)
        else:
            print(Fore.RED + '[FAILED]: The fidelity of two states is {}'.format(fid) + Fore.RESET)

    def protocol(self, qc, prog):

        qstexp = self.experiment(qc,
                                 basis_indices=self.basis_indices,
                                 measurement_indices=self.measurement_indices)

        start = time.time()
        # qstdata = qstexp.run(self.backend, seed_simulation=SEED, shots=self.shots, fitter="cvxpy_gaussian_lstsq").block_for_results()
        qstdata = qstexp.run(self.backend, seed_simulation=SEED, shots=self.shots).block_for_results()
        res = qstdata.analysis_results("state").value
        end = time.time()
        self.data['time'] = end - start
        self.data['rho'] = res

        return res


class Proj(object):

    def __init__(self, qc):
        self.qc = qc

    def run(self, backend, shots=100, seed_simulation=1111):

        circ = qiskit.transpile(self.qc, backend)
        results = backend.run(circ, shots=shots).result()
        counts = results.get_counts()

        return counts

class QUT_PROJ(QUT, ABC):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.experiment = Proj
        self.shots = kwargs.get('shots', 2000)
        self.data = {}
        self.params = []


    @abstractmethod
    def pre(self):
        pass

    @abstractmethod
    def post(self):
        pass

    def assertEqual(self, arg1, arg2):

        # arg1 = np.copy(a1)
        # arg2 = np.copy(a2)

        arg1 = np.array(arg1) + 0.001
        arg2 = np.array(arg2) + 0.001

        ind = np.where(arg2 != 0)

        arg1 = arg1[ind]
        arg2 = arg2[ind]

        fid = r2_score(arg1, arg2, force_finite=True)

        if fid == 0.0:
            arg1 = np.append(arg1, 0)
            arg2 = np.append(arg2, 0)

            fid = r2_score(arg1, arg2, force_finite=True)

        if fid < 0:
            fid = 0

        # import matplotlib.pyplot as plt
        # plt.plot(arg1)
        # plt.plot(arg2)
        # plt.show()
        #
        # fid = chisquare(f_obs=arg1, f_exp=arg2, sum_check=False)
        # fid = fid.pvalue
        # fid = np.exp(-fid.statistic)
        # fid = np.max(np.abs(arg1 - arg2.data))
        # fid = np.sqrt(np.sum(np.abs(arg1 - arg2.data)**2))
        # fid = np.linalg.norm(np.abs(arg1 - arg2.data))
        # T = 1
        # fid = np.exp(-fid/T)
        # diff = np.matrix(arg1.data - arg2.data)
        # fid = 1.0 - 0.5 * np.trace(sqrtm(diff.H @ diff))
        self.data['fid'] = fid

        if fid > 0.5:
            print(Fore.GREEN + '[PASSED]: R^2 score is {}'.format(fid) + Fore.RESET)
        else:
            print(Fore.RED + '[FAILED]: R^2 score is {}'.format(fid) + Fore.RESET)

    def protocol(self, qc, prog):

        qc.measure_all()
        qstexp = self.experiment(qc)
        start = time.time()
        res = qstexp.run(self.backend, seed_simulation=SEED, shots=self.shots)
        end = time.time()
        self.data['time'] = end - start
        keys = set(make_keys(qc.num_qubits))
        diff_keys = keys.difference(res.keys())
        res.update(dict(zip(diff_keys, [0] * len(diff_keys))))
        res = dict(sorted(res.items()))
        self.data['rho'] = res

        return list(res.values())


if __name__ == '__main__':

    aaa = QUT_PT()
    aaa.run()
