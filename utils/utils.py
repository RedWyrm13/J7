from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple
import numpy as np

ZZPair = Tuple[int, int]

@dataclass(frozen=True)
class cfgCircuit:
    n_qubits: int = 4
    n_ops: int = 1000
    n_circuits: int = 500
    resamples_per_circuit: int = 10
    shots_per_datapoint: int = 1024

    reverse_bits_for_marginals: bool = False
    zz_pairs: Optional[Sequence[ZZPair]] = field(default=None)

    master_seed: int = 12345
    label: str = "configuration not defined"

    # flattening options (keep here so you can standardize across families)
    include_zz_in_flatten: bool = True
    
    def __post_init__(self):
        n = self.n_qubits
        pairs = [(i,i+1) if i < n else None for i in range(n)]
        object.__setattr__(self, 'zz_pairs', pairs)

def make_rng(master_seed: int) -> np.random.Generator:
    return np.random.default_rng(master_seed)

def sample_circuit_seed(rng: np.random.Generator) -> int:
    # keep within signed 32-bit
    return int(rng.integers(0, 2**31 - 1))

def make_metadata(cfg, circuit_index, resample_index, circuit_seed):
    meta_data = {
    "family": cfg.label, # IQP, clifford, etc
    "n_qubits": cfg.n_qubits, #number of qubits
    "num_iqp_ops": cfg.n_ops, # Number of operations in the circuit
    "shots": cfg.shots_per_datapoint, # Number of shots for each circuit
    "circuit_index": circuit_index, # The position of the circuit in the list of circuit family datapoints
    "resample_index": resample_index, # Each circuit is resampled multiple times. This is that index
    "circuit_seed": circuit_seed, # Seed to recreate the circuit
    "reverse_bits_for_marginals": cfg.reverse_bits_for_marginals, # Reverse from little endien to big endien
    "zz_pairs": list(cfg.zz_pairs) if cfg.zz_pairs is not None else None, # zz pairs between different qubits
    "master_seed": cfg.master_seed # Master seed
}
    
    return meta_data

def make_filename(cfg: cfgCircuit, file_extension: str = ".npz") -> str:
    """Create a stable dataset filename based on config."""
    return (f"../data/quantum/"
        f"circuitFamily_{cfg.label}_qubits{cfg.n_qubits}_ops{cfg.n_ops}_shotsPerDatapoint{cfg.shots_per_datapoint}"
        f"_numCircuits{cfg.n_circuits}_resamplesPerCircuit{cfg.resamples_per_circuit}"
        f"_masterSeed{cfg.master_seed}{file_extension}")

def save_data(filename, X, y, meta, featdicts):
    np.savez(filename, X=X, y=y, meta=meta, featdicts=featdicts)

    