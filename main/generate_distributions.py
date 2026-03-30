import sys
sys.path.append('../')

from utils.run_circuit import build_and_transpile, run_sampler, run_multi_basis, run_shadows
from utils.circuit_statistics import (
    summarize_counts_dict, flatten_feature_dict,
    summarize_multi_basis, summarize_shadows,
)
from utils.utils import make_metadata, cfgCircuit, sample_circuit_seed, make_filename, save_data
import numpy as np
from typing import List, Dict, Any
from qiskit_aer import AerSimulator
from circuit_builders.builder import get_builder, CircuitFamily
import yaml
import sys


def load_cfg_from_yaml(path: str) -> List[cfgCircuit]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        qubit_min = data['qubit_range'][0]
        qubit_max = data['qubit_range'][-1]
        labels = data['labels']
        shots_base = data['shots_base']
        modes = data.get('measurement_modes', ['z_only'])

        cfgs = []
        i = 0
        for mode in modes:
            for num_qubits in range(qubit_min, qubit_max):
                shots = shots_base * num_qubits ** 2
                for label in labels:
                    cfg_dict = {
                        'n_qubits': num_qubits,
                        'label': label,
                        'n_circuits': data['n_circuits'],
                        'n_ops': data['n_ops'],
                        'resamples_per_circuit': data['resamples_per_circuit'],
                        'shots_per_datapoint': shots,
                        'master_seed': data['master_seed'] + i,
                        'measurement_mode': mode,
                    }
                    cfg = cfgCircuit(**cfg_dict)
                    cfgs.append(cfg)
                    i += 1

        return cfgs


def _make_sim():
    import os
    sim = AerSimulator()
    devices = sim.available_devices()
    device = 'GPU' if 'GPU' in devices else 'CPU'
    sim.set_options(method='statevector', device=device, max_memory_mb=0)
    print(f"Aer device = {device}")
    return sim


def generate_distributions(cfg: cfgCircuit = cfgCircuit()):
    import os
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

    num_qubits   = cfg.n_qubits
    num_ops      = cfg.n_ops
    num_samples  = cfg.n_circuits
    resamples    = cfg.resamples_per_circuit
    mode         = cfg.measurement_mode

    family        = cfg.label.lower()
    family_enum   = CircuitFamily(family)
    circuit_builder = get_builder(family_enum)

    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    meta_list: List[Dict[str, Any]] = []
    featdict_list: List[Dict[str, Any]] = []

    sim = _make_sim()
    rng = np.random.default_rng(cfg.master_seed)

    for circuit_index in range(num_samples):
        circuit_seed = sample_circuit_seed(rng)

        raw_circuit = circuit_builder(
            num_qubits=num_qubits,
            num_ops=num_ops,
            seed=circuit_seed,
        )
        # build_and_transpile adds measurements; for multi_basis/shadows we need
        # a measurement-free base to add our own rotations later.
        transpiled = build_and_transpile(circuit=raw_circuit, simulator=sim)

        for r_idx in range(resamples):

            if mode == "z_only":
                counts = run_sampler(transpiled, sim=sim, shots=cfg.shots_per_datapoint)
                featdict = summarize_counts_dict(counts, zz_pairs=cfg.zz_pairs)
                feature_vec = flatten_feature_dict(featdict)

            elif mode == "multi_basis":
                shots_per_basis = cfg.shots_per_datapoint // 3
                z_counts, x_counts, y_counts = run_multi_basis(
                    transpiled, sim, shots_per_basis=shots_per_basis
                )
                feature_vec = summarize_multi_basis(
                    z_counts, x_counts, y_counts,
                    n_qubits=num_qubits,
                    pairs=cfg.zz_pairs,
                )
                featdict = {}   # raw counts not stored for multi_basis

            elif mode == "shadows":
                rotations, outcomes = run_shadows(
                    transpiled, sim,
                    n_shadows=cfg.shots_per_datapoint,
                    rng=rng,
                )
                feature_vec = summarize_shadows(
                    rotations, outcomes,
                    n_qubits=num_qubits,
                    pairs=cfg.zz_pairs,
                )
                featdict = {}

            else:
                raise ValueError(f"Unknown measurement_mode: {mode!r}")

            meta_data = make_metadata(
                cfg=cfg,
                circuit_index=circuit_index,
                resample_index=r_idx,
                circuit_seed=circuit_seed,
            )

            X_list.append(feature_vec)
            y_list.append(cfg.label)
            meta_list.append(meta_data)
            featdict_list.append(featdict)

    filename = make_filename(cfg=cfg)

    X        = np.vstack(X_list)
    y        = np.array(y_list)
    meta     = np.array(meta_list)
    featdicts = np.array(featdict_list)

    print(filename)
    save_data(filename=filename, X=X, y=y, meta=meta, featdicts=featdicts)
    return filename


def run_one_cfg(cfg: cfgCircuit) -> str:
    generate_distributions(cfg=cfg)
    return make_filename(cfg)


if __name__ == "__main__":
    import time
    import os
    from dask.distributed import Client, as_completed

    start_time = time.time()
    config_path = sys.argv[1]
    cfgs = load_cfg_from_yaml(config_path)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    slurm_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", "0"))
    n_workers  = min(2, slurm_gpus) if slurm_gpus > 0 else 0

    cluster = None
    if n_workers > 0:
        try:
            from dask_cuda import LocalCUDACluster # type:ignore
            cluster = LocalCUDACluster(
                n_workers=n_workers,
                threads_per_worker=1,
            )
            print(f"Using LocalCUDACluster with {n_workers} GPU workers.")
        except Exception as e:
            print("Could not start LocalCUDACluster (falling back to CPU). Reason:", repr(e))

    if cluster is None:
        from dask.distributed import LocalCluster
        cpu_workers = min(4, os.cpu_count() or 1)
        cluster = LocalCluster(
            n_workers=cpu_workers,
            threads_per_worker=1,
            processes=True,
        )
        print(f"Using CPU LocalCluster with {cpu_workers} workers.")

    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    futures = client.map(run_one_cfg, cfgs)

    for fut in as_completed(futures):
        try:
            out = fut.result()
            print("DONE:", out)
        except Exception as e:
            print("FAILED:", repr(e))

    client.close()
    cluster.close(timeout=60)
    print(f"Total time: {(time.time() - start_time):.2f}s")
