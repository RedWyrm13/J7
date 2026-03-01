import sys
sys.path.append('../')

from utils.run_circuit import build_and_transpile, run_sampler
from utils.circuit_statistics import summarize_counts_dict, flatten_feature_dict
from utils.utils import make_metadata, cfgCircuit, sample_circuit_seed, make_filename, save_data
import numpy as np
from typing import List, Dict, Any
from qiskit_aer import AerSimulator
from circuit_builders.builder import get_builder, CircuitFamily
import yaml
import sys

def load_cfg_from_yaml(path:str) -> List[cfgCircuit]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        qubit_min = data['qubit_range'][0]
        qubit_max = data['qubit_range'][-1]
        labels = data['labels']
        shot_min = data['shot_range'][0]
        shot_max = data['shot_range'][-1]
        
        cfgs = []
        i = 0
        for num_qubits in range(qubit_min, qubit_max):
            for label in labels:
                for shot_exponent in range (shot_min, shot_max):
                    shots = 2**shot_exponent
                    cfg_dict = {
                        'n_qubits': num_qubits,
                        'label': label,
                        'n_circuits': data['n_circuits'],
                        'n_ops': data['n_ops'],
                        'resamples_per_circuit': data['resamples_per_circuit'],
                        'shots_per_datapoint': shots,
                        'master_seed': data['master_seed'] + i
                    }
                    cfg = cfgCircuit(**cfg_dict)
                    cfgs.append(cfg)
                    i += 1

        return cfgs

def generate_distributions(cfg: cfgCircuit = cfgCircuit()):  

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    # Circuit building and sampling parameters
    num_qubits = cfg.n_qubits
    num_ops = cfg.n_ops
    num_samples = cfg.n_circuits
    resamples = cfg.resamples_per_circuit
    
    # Next 3 lines select the correct circuit builder based on the label
    family = cfg.label.lower()
    family_enum = CircuitFamily(family)
    circuit_builder = get_builder(family_enum)
    

    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    meta_list: List[Dict[str, Any]] = []
    featdict_list: List[Dict[str, Any]] = []
    
    sim = AerSimulator()
    devices = sim.available_devices()
    device = 'GPU' if 'GPU' in devices else 'CPU'
    print(f"Aer Simulator is using {device}.")
    sim.set_options(method='density_matrix', device=device)
    print("Simulator Options:", sim.options)
    print(f"[{cfg.label} | q={cfg.n_qubits} | shots={cfg.shots_per_datapoint}] Aer device = {device}")
    
    rng = np.random.default_rng(cfg.master_seed)
    
    for circuit_index in range(num_samples):
        
        circuit_seed = sample_circuit_seed(rng)

        clifford_circuit = circuit_builder(num_qubits = num_qubits, 
                                           num_ops = num_ops,
                                           seed = circuit_seed)
        transpiled_clifford_circuit = build_and_transpile(circuit = clifford_circuit, 
                                                          simulator=sim)

        
        for r_idx in range (resamples):
            counts = run_sampler(transpiled_circuit=transpiled_clifford_circuit, 
                                 sim=sim, 
                                 shots = cfg.shots_per_datapoint)
            
            featdict = summarize_counts_dict(counts)
            
            
            flattened_feature_dict =flatten_feature_dict(feat=featdict)
            
            meta_data = make_metadata(cfg=cfg,
                                     circuit_index=circuit_index,
                                     resample_index=r_idx,
                                     circuit_seed=circuit_seed)
            
            X_list.append(flattened_feature_dict)
            y_list.append(cfg.label)
            meta_list.append(meta_data)
            featdict_list.append(featdict)
            
            
            
            
    filename = make_filename(cfg=cfg)
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    meta = np.array(meta_list)
    featdicts = np.array(featdict_list)
    
    print(filename)
    save_data(filename = filename,
                X = X, 
                y = y,
                meta=meta,
                featdicts=featdicts)
    
    return filename

def run_one_cfg(cfg: cfgCircuit) -> str:
    generate_distributions(cfg=cfg)
    return make_filename(cfg)
    

if __name__ == "__main__":
    import time
    start_time = time.time()
    import os
    from dask.distributed import Client, LocalCluster, as_completed
    
    config_path = sys.argv[1]
    cfgs = load_cfg_from_yaml(config_path)
    
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    
    # How many GPUs did slurm allocate for us?
    slurm_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", "0"))

    # Try GPU cluster
    cluster = None
    if slurm_gpus > 0:
        try:
            from dask_cuda import LocalCUDACluster # This requires pip install dask-cuda
            cluster = LocalCUDACluster(
                n_workers = slurm_gpus,
                threads_per_worker=1,
            )
            print(f"Using LocalCUDACluster with {slurm_gpus} GPU workers.")
        except Exception as e:
            print("Could not start LocalCUDACluster (falling back to CPU Loca Cluster). Reason:", repr(e))
            cluster = None
    
    # Fallback onto CPU cluster
    
    if cluster is None:
        from dask.distributed import LocalCluster
        n_workers = min(8, os.cpu_count() or 1) # Change 8 to number of cpus desired
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker = 1,
            processes = True,
        )        
        print(f"Using CPU LocalCluster with {n_workers} workers") 
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
    cluster.close()
    end_time = time.time()
    print(f"Total time is {(end_time - start_time):.2f}.")