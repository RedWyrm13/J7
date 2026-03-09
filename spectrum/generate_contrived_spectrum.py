"""
generate_contrived_spectrum.py
==============================
Generates a contrived spectrum dataset that mimics power spectral density
(PSD) snapshots, where each snapshot is a binarized frequency-bin observation.

The mapping is:
    - n_qubits  → number of frequency bins monitored
    - 1 snapshot → 1 bitstring  (bin=1 means "active/above threshold")
    - N snapshots → counts dictionary → same feature pipeline as circuits

For the contrived case, we sample bitstrings directly from the empirical
distributions learned from existing circuit data. This validates the
end-to-end pipeline before real spectrum data is available.

Usage:
    python3 generate_contrived_spectrum.py

Output:
    ../data/spectrum_contrived_<family>_qubits<n>_snapshots<N>_seed<s>.npz
    
    Each file has the same X, y, meta, featdicts structure as circuit data,
    so existing model code (models/models.py) requires zero changes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from collections import Counter
from utils.circuit_statistics import summarize_counts_dict, flatten_feature_dict
from utils.utils import cfgCircuit

# ── Configuration ─────────────────────────────────────────────────────────

N_SNAPSHOTS   = 1000   # Number of PSD snapshots per synthetic sensor session
N_SESSIONS    = 1000    # Number of independent 'sensor sessions' (like circuits)
SHOTS         = 1000   # Snapshots per session (analogous to shots_per_datapoint)
N_QUBITS      = 4      # Frequency bins = qubit count (match your circuit data)
SEED          = 42
DATA_DIR      = Path("../data/quantum")
OUT_DIR       = Path("../data/spectrum")

FAMILIES = ["iqp", "clifford", "clifford_t"]

# ── Helpers ───────────────────────────────────────────────────────────────

def load_empirical_distribution(family: str, n_qubits: int, data_dir: Path) -> dict:
    """
    Load an existing circuit npz file and reconstruct the empirical
    bitstring probability distribution by inverting the feature pipeline.

    Since we don't store raw bitstrings (only statistics), we reconstruct
    a synthetic counts dict that matches the marginal probabilities and
    Hamming weight histogram of the stored distribution.

    Returns a dict {bitstring: probability} over all 2^n bitstrings.
    """
    # Find a matching file for this family + qubit count
    # Use _qubits prefix to avoid clifford matching clifford_t files
    pattern = f"*circuitFamily_{family}_qubits{n_qubits}_*.npz"
    matches = list(data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No data file found for family='{family}', n_qubits={n_qubits} "
            f"in {data_dir}. Run generate_distributions.py first."
        )

    # Average the marginals and hamming histogram across all circuits
    # to get a stable empirical distribution
    all_marginals = []
    all_hw_hists  = []

    data = np.load(matches[0], allow_pickle=True)
    featdicts = data['featdicts']

    for fd in featdicts:
        all_marginals.append(fd['qubit_marginals'])
        all_hw_hists.append(fd['hamming_weight']['hist'])

    avg_marginals = np.mean(all_marginals, axis=0)   # shape: (n_qubits,)
    avg_hw_hist   = np.mean(all_hw_hists,  axis=0)   # shape: (n_qubits+1,)

    return {
        "marginals": avg_marginals,
        "hw_hist":   avg_hw_hist,
        "n_qubits":  n_qubits,
        "family":    family,
        "source_file": str(matches[0].name),
    }


def sample_bitstrings_from_marginals(
    marginals: np.ndarray,
    hw_hist: np.ndarray,
    n_snapshots: int,
    rng: np.random.Generator,
) -> list[str]:
    """
    Sample bitstrings that approximately match the target marginal
    probabilities and Hamming weight distribution.

    Strategy:
      1. Sample a Hamming weight w from hw_hist
      2. Sample w qubit positions to be '1', weighted by their marginals
      3. This respects both the weight distribution and qubit biases

    This is an approximation — it won't capture ZZ correlations — but it
    produces bitstrings whose first-order statistics match the circuit family,
    which is sufficient for the contrived demonstration.
    """
    n = len(marginals)
    weights = np.arange(n + 1)
    bitstrings = []

    for _ in range(n_snapshots):
        # Sample Hamming weight from the circuit family's distribution
        w = rng.choice(weights, p=hw_hist / hw_hist.sum())

        if w == 0:
            bitstrings.append('0' * n)
            continue
        if w == n:
            bitstrings.append('1' * n)
            continue

        # Sample which qubits are '1', weighted by marginal P(qubit_i = 1)
        probs = marginals / marginals.sum() if marginals.sum() > 0 else None
        active_qubits = rng.choice(n, size=w, replace=False, p=probs)

        bits = ['0'] * n
        for q in active_qubits:
            bits[q] = '1'
        bitstrings.append(''.join(bits))

    return bitstrings


def bitstrings_to_counts(bitstrings: list[str]) -> dict:
    """Convert a list of bitstrings to a Qiskit-style counts dictionary."""
    return dict(Counter(bitstrings))


def make_spectrum_metadata(
    family: str,
    n_qubits: int,
    session_index: int,
    n_snapshots: int,
    seed: int,
    source_file: str,
) -> dict:
    """Metadata dict mirroring the circuit pipeline's make_metadata()."""
    return {
        "family":         f"spectrum_{family}",
        "n_qubits":       n_qubits,
        "n_frequency_bins": n_qubits,
        "n_snapshots":    n_snapshots,
        "session_index":  session_index,
        "seed":           seed,
        "source_circuit_file": source_file,
        "note": (
            "Contrived spectrum dataset. Each 'snapshot' is a binarized PSD "
            "observation (1=bin active, 0=bin inactive). Bitstrings sampled "
            f"from empirical {family} circuit distribution."
        ),
    }


# ── Main generation loop ───────────────────────────────────────────────────

def generate_contrived_spectrum(
    family: str,
    n_qubits: int,
    n_sessions: int,
    shots_per_session: int,
    data_dir: Path,
    out_dir: Path,
    seed: int,
) -> Path:
    """
    Generate a contrived spectrum dataset for one circuit family.

    Each 'session' mimics a sensor collecting `shots_per_session` PSD
    snapshots over some observation window (e.g. 24 hours). The session
    produces one feature vector, just like one circuit run produces one
    feature vector in the circuit pipeline.
    """
    print(f"\n[{family} | {n_qubits} qubits | {n_sessions} sessions]")

    # Load the empirical distribution from circuit data
    dist = load_empirical_distribution(family, n_qubits, data_dir)
    print(f"  Loaded distribution from: {dist['source_file']}")
    print(f"  Avg marginals: {np.round(dist['marginals'], 3)}")

    rng = np.random.default_rng(seed)

    X_list    = []
    y_list    = []
    meta_list = []
    feat_list = []

    for session_idx in range(n_sessions):

        # Sample bitstrings — these are the "PSD snapshots"
        bitstrings = sample_bitstrings_from_marginals(
            marginals   = dist['marginals'],
            hw_hist     = dist['hw_hist'],
            n_snapshots = shots_per_session,
            rng         = rng,
        )

        # Convert to counts dict (same format as Qiskit sampler output)
        counts = bitstrings_to_counts(bitstrings)

        # Extract features using the EXACT same pipeline as circuits
        featdict = summarize_counts_dict(
            counts,
            n_qubits    = n_qubits,
            reverse_bits_for_marginals = False,
            zz_pairs    = [(i, i+1) for i in range(n_qubits - 1)],
        )

        feature_vector = flatten_feature_dict(featdict, include_zz=True)

        meta = make_spectrum_metadata(
            family        = family,
            n_qubits      = n_qubits,
            session_index = session_idx,
            n_snapshots   = shots_per_session,
            seed          = seed,
            source_file   = dist['source_file'],
        )

        X_list.append(feature_vector)
        y_list.append(f"spectrum_{family}")
        meta_list.append(meta)
        feat_list.append(featdict)

        if (session_idx + 1) % 25 == 0:
            print(f"  Session {session_idx + 1}/{n_sessions} done.")

    X = np.vstack(X_list)
    y = np.array(y_list)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"spectrum_contrived_{family}"
        f"_qubits{n_qubits}"
        f"_sessions{n_sessions}"
        f"_snapshots{shots_per_session}"
        f"_seed{seed}.npz"
    )

    np.savez(
        out_path,
        X        = X,
        y        = y,
        meta     = np.array(meta_list),
        featdicts= np.array(feat_list),
    )

    print(f"  Saved: {out_path.name}  |  X shape: {X.shape}")
    return out_path


def main():
    rng_seed = SEED
    generated = []

    for family in FAMILIES:
        path = generate_contrived_spectrum(
            family            = family,
            n_qubits          = N_QUBITS,
            n_sessions        = N_SESSIONS,
            shots_per_session = SHOTS,
            data_dir          = DATA_DIR,
            out_dir           = OUT_DIR,
            seed              = rng_seed,
        )
        generated.append(path)
        rng_seed += 1   # different seed per family

    print("\n── Summary ───────────────────────────────────────")
    print(f"Generated {len(generated)} contrived spectrum datasets:")
    for p in generated:
        print(f"  {p.name}")
    print("\nThese files are drop-in compatible with models/models.py.")
    print("Labels are prefixed 'spectrum_' to distinguish from circuit data.")
    print("To classify, load alongside circuit npz files and train as normal.")


if __name__ == "__main__":
    main()