from enum import Enum
from typing import Dict, Callable
from qiskit import QuantumCircuit
from .clifford import build_clifford_circuit
from .iqp import build_iqp_circuit
from .clifford_T import build_clifford_T_circuit

class CircuitFamily(str, Enum):
    IQP = 'iqp'
    CLIFFORD = 'clifford'
    CLIFFORD_T = 'clifford_t'
    # Add new types here

BUILDERS: Dict[CircuitFamily, Callable[..., QuantumCircuit]] = {
    CircuitFamily.IQP: build_iqp_circuit,
    CircuitFamily.CLIFFORD: build_clifford_circuit,
    CircuitFamily.CLIFFORD_T: build_clifford_T_circuit
    # Add new types here

}

def get_builder(family: CircuitFamily) -> Callable[..., QuantumCircuit]:
    """Return the builder function for a given circuit family"""
    try:
        return BUILDERS[family]
    except KeyError:
        raise ValueError(f"Unknown circuit family: {family}. Options: {list(BUILDERS.keys())}.")