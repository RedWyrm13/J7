"""
run_spectrum_experiment.py
==========================
Runs three classification experiments comparing quantum circuit data
against contrived spectrum data.

Directory structure expected:
    data/quantum/   -- circuitFamily_*.npz files
    data/spectrum/  -- spectrum_contrived_*.npz files

Experiments:
    1. Circuit-only   : train/test on quantum circuit data (baseline)
    2. Spectrum-only  : train/test on contrived spectrum data
    3. Cross          : train on circuit data, test on spectrum data
                        (key result for the paper)

Usage (from spectrum/ directory):
    python3 run_spectrum_experiment.py <config.yaml>
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split

from models.models import initialize_models, eval_model

# ── Configuration ──────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent

_MODE_DIRS = {
    "z_only":      (ROOT_DIR / "data" / "quantum",         ROOT_DIR / "data" / "spectrum"),
    "multi_basis": (ROOT_DIR / "data" / "quantum_mb",       ROOT_DIR / "data" / "spectrum_mb"),
    "shadows":     (ROOT_DIR / "data" / "quantum_shadows",  ROOT_DIR / "data" / "spectrum_shadows"),
}

# Set by main() based on CLI argument
QUANTUM_DIR  = ROOT_DIR / "data" / "quantum"
SPECTRUM_DIR = ROOT_DIR / "data" / "spectrum"
OUT_DIR      = Path(__file__).resolve().parent / "plots"


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    q_min, q_max = data["qubit_range"][0], data["qubit_range"][-1]
    return {
        "qubit_range": range(q_min, q_max),
        "seed":        data.get("master_seed", 42),
    }

# ── Data loading ───────────────────────────────────────────────────────────

def load_npz_files(pattern: str, folder: Path):
    """Load all npz files matching a glob pattern. Returns (X, y) arrays."""
    X_list, y_list = [], []
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No files found for pattern '{pattern}' in {folder}\n"
            f"Make sure generate_contrived_spectrum.py has been run first."
        )
    for f in matches:
        data = np.load(f, allow_pickle=True)
        X_list.append(data['X'])
        y_list.append(data['y'])
    print(f"  Loaded {len(matches)} file(s) from {folder.name}/ matching '{pattern}'")
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def load_quantum_data(n_qubits: int):
    return load_npz_files(f"circuitFamily_*qubits{n_qubits}_*.npz", QUANTUM_DIR)


def load_spectrum_data(n_qubits: int):
    return load_npz_files(f"spectrum_contrived_*qubits{n_qubits}*.npz", SPECTRUM_DIR)


# ── Experiment runner ──────────────────────────────────────────────────────

def run_experiment(X_train, y_train, X_test, y_test, label: str) -> dict:
    """Fit all four models and return {model_name: accuracy}."""
    print(f"\n  [{label}]")
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"  Classes in train: {sorted(np.unique(y_train))}")
    print(f"  Classes in test:  {sorted(np.unique(y_test))}")

    models  = initialize_models()
    results = {}
    for name, model in models.items():
        acc = eval_model(model, X_train, y_train, X_test, y_test)
        results[name] = acc
        print(f"    {name:<35} {acc:.4f}")
    return results


def experiment_circuit_only(n_qubits: int, seed: int) -> dict:
    """Baseline: train and test entirely on quantum circuit data."""
    X, y = load_quantum_data(n_qubits)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return run_experiment(X_train, y_train, X_test, y_test,
                          "Experiment 1 — Circuit-only baseline")


def experiment_spectrum_only(n_qubits: int, seed: int) -> dict:
    """Train and test entirely on contrived spectrum data."""
    X, y = load_spectrum_data(n_qubits)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return run_experiment(X_train, y_train, X_test, y_test,
                          "Experiment 2 — Spectrum-only")


def experiment_cross(n_qubits: int) -> dict:
    """
    Train on circuit data, test on spectrum data.
    Labels in spectrum files are prefixed 'spectrum_' — strip that prefix
    so they match the circuit training labels (iqp, clifford, clifford_t).
    """
    X_train, y_train = load_quantum_data(n_qubits)
    X_test,  y_test  = load_spectrum_data(n_qubits)
    y_test_mapped    = np.array([y.replace("spectrum_", "") for y in y_test])

    return run_experiment(X_train, y_train, X_test, y_test_mapped,
                          "Experiment 3 — Cross: train=circuit, test=spectrum  *** KEY RESULT ***")


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_comparison(results: dict, n_qubits: int):
    """Grouped bar chart: one group per model, one bar per experiment."""
    exp_names   = list(results.keys())
    model_names = list(next(iter(results.values())).keys())
    x           = np.arange(len(model_names))
    width       = 0.25
    colors      = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (exp_name, color) in enumerate(zip(exp_names, colors)):
        accs = [results[exp_name][m] for m in model_names]
        bars = ax.bar(x + i * width, accs, width, label=exp_name, color=color, alpha=0.85)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.2f}", ha='center', va='bottom', fontsize=8)

    ax.axhline(y=1/3, color='gray', linestyle='--', linewidth=1.0,
               label='Random chance (3 classes)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=12, ha='right')
    ax.set_ylabel("Classification Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Circuit vs Spectrum Classification — {n_qubits} Qubits / Frequency Bins")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"spectrum_experiment_qubits{n_qubits}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def run_for_qubits(n_qubits: int, seed: int):
    print(f"\n{'═'*62}")
    print(f"  n_qubits = {n_qubits}")
    print(f"{'═'*62}")

    print("── Experiment 1: Circuit-only baseline ──────────────────────────")
    circuit_results  = experiment_circuit_only(n_qubits, seed)

    print("\n── Experiment 2: Spectrum-only ──────────────────────────────────")
    spectrum_results = experiment_spectrum_only(n_qubits, seed)

    print("\n── Experiment 3: Cross (train=circuit, test=spectrum) ────────────")
    cross_results    = experiment_cross(n_qubits)

    all_results = {
        "Circuit": circuit_results,
        "Spectrum": spectrum_results,
        "Cross":   cross_results,
    }

    plot_comparison(all_results, n_qubits)

    print("\n── Summary ───────────────────────────────────────────────────────")
    for exp, res in all_results.items():
        best = max(res, key=res.get)
        print(f"  {exp:<10}  best: {best} → {res[best]:.4f}")
    print(f"  Random baseline: {1/3:.4f}  (3 classes)")

    return all_results


def main():
    global QUANTUM_DIR, SPECTRUM_DIR, OUT_DIR

    if len(sys.argv) < 2:
        print("Usage: python3 run_spectrum_experiment.py <config.yaml> [mode]")
        print("  mode: z_only (default) | multi_basis | shadows")
        sys.exit(1)

    mode = sys.argv[2] if len(sys.argv) >= 3 else "z_only"
    if mode not in _MODE_DIRS:
        print(f"Unknown mode '{mode}'. Choose from: {list(_MODE_DIRS)}")
        sys.exit(1)

    QUANTUM_DIR, SPECTRUM_DIR = _MODE_DIRS[mode]
    OUT_DIR = Path(__file__).resolve().parent / "plots" / mode
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg         = load_cfg(sys.argv[1])
    qubit_range = cfg["qubit_range"]
    seed        = cfg["seed"]

    print(f"Spectrum Classification Experiments")
    print(f"  Quantum data : {QUANTUM_DIR}")
    print(f"  Spectrum data: {SPECTRUM_DIR}")
    print(f"  Qubits       : {list(qubit_range)}")

    all_qubit_results = {}
    for n_qubits in qubit_range:
        all_qubit_results[n_qubits] = run_for_qubits(n_qubits, seed)

    print(f"\n{'═'*62}")
    print("  OVERALL SUMMARY")
    print(f"{'═'*62}")
    for n_qubits, results in all_qubit_results.items():
        cross = results["Cross"]
        best  = max(cross, key=cross.get)
        print(f"  qubits={n_qubits:<3}  cross best: {best} → {cross[best]:.4f}")


if __name__ == "__main__":
    main()
