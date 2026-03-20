from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit_ibm_runtime import SamplerV2 as Sampler
import numpy as np

# Rotation codes used by multi_basis and shadows
# 0 = Z basis (identity)  1 = X basis (H)  2 = Y basis (Sdg + H)
BASIS_Z, BASIS_X, BASIS_Y = 0, 1, 2

def build_and_transpile(circuit, simulator=None):
    """
    circuit: qiskit QuantumCircuit
    simulator: AerSimulator. If None, default simulator will be used.
    """
    circuit.remove_final_measurements()
    circuit.measure_all()

    if simulator is None:
        simulator = AerSimulator()

    return transpile(backend=simulator, circuits=circuit)


def run_sampler(transpiled_circuit, sim=None, shots=None):
    if sim is None:
        sim = AerSimulator()

    result = sim.run(transpiled_circuit, shots=shots).result()
    return result.get_counts()


def _append_basis_rotations(base_circuit, rotations):
    """
    Return a new circuit with per-qubit basis rotations appended before measurement.
    rotations: sequence of length n_qubits with values BASIS_Z / BASIS_X / BASIS_Y.
    """
    n = base_circuit.num_qubits
    circ = base_circuit.copy()
    circ.remove_final_measurements()
    for q, rot in enumerate(rotations):
        if rot == BASIS_X:
            circ.h(q)
        elif rot == BASIS_Y:
            circ.sdg(q)
            circ.h(q)
    circ.measure_all()
    return circ


def run_multi_basis(base_circuit, sim, shots_per_basis):
    """
    Run base_circuit in Z, X, and Y bases.
    Returns (z_counts, x_counts, y_counts).
    """
    n = base_circuit.num_qubits
    circuits = [
        _append_basis_rotations(base_circuit, [BASIS_Z] * n),
        _append_basis_rotations(base_circuit, [BASIS_X] * n),
        _append_basis_rotations(base_circuit, [BASIS_Y] * n),
    ]
    transpiled = transpile(circuits, backend=sim)
    results = sim.run(transpiled, shots=shots_per_basis).result()
    return (
        results.get_counts(0),
        results.get_counts(1),
        results.get_counts(2),
    )



# Per-qubit rotation matrices for Z(I), X(H), Y(Sdg·H) bases
_ROT = np.array([
    [[1, 0], [0, 1]],                                           # Z: identity
    [[1, 1], [1, -1]] / np.sqrt(2),                             # X: H
    [[1, 1], [-1j, 1j]] / np.sqrt(2),                          # Y: Sdg·H
], dtype=complex)


def run_shadows(base_circuit, sim, n_shadows, rng):
    """
    Execute the classical shadows protocol using a single GPU statevector
    simulation followed by numpy-based sampling — no per-shadow transpilation.

    Returns:
        rotations: (n_shadows, n_qubits) int array  (0/1/2 = Z/X/Y)
        outcomes:  (n_shadows, n_qubits) int array  (0/1)
    """
    n = base_circuit.num_qubits
    rotations = rng.integers(0, 3, size=(n_shadows, n))

    # ── One GPU simulation to get the statevector ─────────────────────────
    sv_circ = base_circuit.copy()
    sv_circ.remove_final_measurements()
    sv_circ.save_statevector()
    sv = np.array(sim.run(sv_circ).result().get_statevector())  # (2^n,) complex

    # Reshape to (2, 2, ..., 2) — axis q corresponds to qubit q
    sv_tensor = sv.reshape((2,) * n)

    outcomes = np.zeros((n_shadows, n), dtype=np.int8)

    for s in range(n_shadows):
        # Apply per-qubit rotations via tensor contraction along each qubit axis
        st = sv_tensor.copy()
        for q, rot in enumerate(rotations[s]):
            if rot != BASIS_Z:
                # contract _ROT[rot] (2×2) with axis q of st
                st = np.tensordot(_ROT[rot], st, axes=([1], [q]))
                # tensordot moves the contracted axis to position 0; move it back
                st = np.moveaxis(st, 0, q)

        probs = np.abs(st.ravel()) ** 2
        probs /= probs.sum()  # renormalise for floating-point safety
        outcome_idx = rng.choice(len(probs), p=probs)
        for q in range(n):
            outcomes[s, q] = (outcome_idx >> q) & 1  # qubit q is bit q

    return rotations, outcomes