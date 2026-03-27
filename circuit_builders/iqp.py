from qiskit import QuantumCircuit
import numpy as np

def randint_excluding(rng, a, b, x):
    # pick uniformly from [a,b] excluding x
    r = rng.integers(a, b)  # [a, b)
    if r >= x:
        r += 1
    return int(r)

def apply_hadamards(qc):
    qc.h(range(qc.num_qubits))
    return qc

def build_iqp_circuit(num_qubits, num_ops, seed=None):
    rng = np.random.default_rng(seed)

    qc = QuantumCircuit(num_qubits)
    apply_hadamards(qc)
    qc.barrier()

    gates = ["T", "RZ", "CZ"]

    for _ in range(num_ops):
        gate = rng.choice(gates)
        target = int(rng.integers(0, num_qubits))

        if gate == "T":
            qc.t(target)

        elif gate == "RZ":
            phi = float(rng.random() * 2 * np.pi)
            qc.rz(phi, target)

        else:  # "CZ"
            control = randint_excluding(rng, 0, num_qubits - 1, target)
            qc.cz(control, target)
    qc.barrier()
    apply_hadamards(qc)
    return qc
