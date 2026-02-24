from qiskit import QuantumCircuit
import numpy as np
def randint_exclude(n: int, exclude: int, rng: np.random.Generator) -> int:
    """
    Returns a random integer in [0, n-1], excluding 'exclude'.
    """

    res = rng.integers(0, n - 1)
    
    if res >= exclude:
        res += 1
        
    return int(res)

def generate_random_indices(num_qubits: int, rng: np.random.Generator, gate: str = None):
    if gate == 'cx':
        control_qubit = rng.integers(0, num_qubits)
        target_qubit = randint_exclude(num_qubits, control_qubit, rng)
        
        return int(control_qubit), int(target_qubit)
    else:
        return int(rng.integers(0, num_qubits))
        
    
def apply_gate(qc: QuantumCircuit, rng: np.random.Generator):
    """
    This function chooses the quantum gate to apply, and applies it
    """
    
    num_qubits = qc.num_qubits
    clifford = ["cx", "h", "s"]
    gate = np.random.choice(clifford)
    
    if gate == 'cx':
        control_qubit, target_qubit = generate_random_indices(num_qubits = num_qubits, gate = "cx", rng = rng)
        qc.cx(control_qubit, target_qubit)
        
    if gate == 'h':
        qubit = generate_random_indices(num_qubits, rng)
        qc.h(qubit)
        
    if gate == 's':
        qubit = generate_random_indices(num_qubits, rng)
        qc.s(qubit)
    
    return qc

def build_clifford_circuit(num_qubits, num_ops, seed = None):
    rng = np.random.default_rng(seed)

    qc = QuantumCircuit(num_qubits)
    
    for _ in range(num_ops):
        qc = apply_gate(qc, rng)
        
    return qc


