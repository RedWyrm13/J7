# statistics.py
# Dict-first statistics for Qiskit counts -> features

from __future__ import annotations

from typing import Dict, Any, Optional, Sequence, Tuple
import numpy as np

Counts = Dict[str, int]


# ---------------------------
# Helpers
# ---------------------------

def _validate_counts(counts: Counts) -> None:
    if not isinstance(counts, dict) or len(counts) == 0:
        raise ValueError("counts must be a non-empty dict: {bitstring: int}")
    for k, v in counts.items():
        if not isinstance(k, str) or any(c not in "01" for c in k):
            raise ValueError(f"Invalid bitstring key: {k!r}")
        if not isinstance(v, (int, np.integer)) or v < 0:
            raise ValueError(f"Invalid count value for {k!r}: {v!r}")


def infer_n_qubits(counts: Counts) -> int:
    _validate_counts(counts)
    lengths = {len(k) for k in counts.keys()}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent bitstring lengths: {sorted(lengths)}")
    return next(iter(lengths))


def total_shots(counts: Counts) -> int:
    return int(sum(counts.values()))


def _maybe_reverse(bitstring: str, reverse_bits: bool) -> str:
    # Qiskit commonly displays bitstrings with qubit-0 on the RIGHT (little-endian).
    # reverse_bits=True flips so index 0 corresponds to qubit 0.
    return bitstring[::-1] if reverse_bits else bitstring


# ---------------------------
# Core stats
# ---------------------------

def hamming_weight_histogram(counts: Counts, n_qubits: int) -> np.ndarray:
    """
    Normalized histogram over Hamming weight w=0..n.
    hist[w] = P(|x| = w)
    """
    shots = total_shots(counts)
    hist = np.zeros(n_qubits + 1, dtype=np.float64)
    if shots == 0:
        return hist
    for bitstring, c in counts.items():
        hist[bitstring.count("1")] += c
    hist_sum = hist.sum()
    return hist / hist_sum if hist_sum > 0 else hist


def hamming_weight_moments(hw_hist: np.ndarray) -> Dict[str, float]:
    """
    Mean/variance of Hamming weight given the histogram.
    """
    n = len(hw_hist) - 1
    ws = np.arange(n + 1, dtype=np.float64)
    mean = float(np.dot(ws, hw_hist))
    var = float(np.dot((ws - mean) ** 2, hw_hist))
    return {"mean": mean, "var": var}


def single_qubit_marginals(counts: Counts, n_qubits: int, reverse_bits: bool = True) -> np.ndarray:
    """
    marg[i] = P(x_i = 1)
    """
    shots = total_shots(counts)
    marg = np.zeros(n_qubits, dtype=np.float64)
    if shots == 0:
        return marg

    for bitstring, c in counts.items():
        b = _maybe_reverse(bitstring, reverse_bits)
        if len(b) != n_qubits:
            raise ValueError("Bitstring length mismatch; check n_qubits.")
        for i, ch in enumerate(b):
            if ch == "1":
                marg[i] += c

    return marg / shots


def parity_bias(counts: Counts) -> float:
    """
    parity_bias = P(even parity) - P(odd parity)
    """
    shots = total_shots(counts)
    if shots == 0:
        return 0.0
    even = 0
    odd = 0
    for bitstring, c in counts.items():
        if (bitstring.count("1") % 2) == 0:
            even += c
        else:
            odd += c
    return float((even - odd) / shots)


def shannon_entropy_bits(counts: Counts) -> float:
    """
    Shannon entropy H(P) in bits using empirical probabilities.
    """
    shots = total_shots(counts)
    if shots == 0:
        return 0.0
    ent = 0.0
    for _, c in counts.items():
        if c <= 0:
            continue
        p = c / shots
        ent -= p * np.log2(p)
    return float(ent)


def pauli_marginals_from_counts(counts: Counts, n_qubits: int) -> np.ndarray:
    """
    Estimate <P_i> = 3 * mean((-1)^b_i) for a globally-rotated measurement.
    Works for any single global basis: the caller rotated the circuit, so
    computational-basis outcomes already encode the chosen Pauli.
    Returns array of length n_qubits.
    """
    shots = total_shots(counts)
    expectation = np.zeros(n_qubits, dtype=np.float64)
    for bitstring, c in counts.items():
        b = bitstring[::-1]  # qubit-0 first
        for i, ch in enumerate(b):
            expectation[i] += c * (1.0 if ch == "0" else -1.0)
    return expectation / shots


def pauli_connected_correlators(counts: Counts, n_qubits: int,
                                 pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """
    Connected correlator <P_i P_j> - <P_i><P_j> for a globally-rotated measurement.
    """
    shots = total_shots(counts)
    exp1 = pauli_marginals_from_counts(counts, n_qubits)
    exp2 = np.zeros(len(pairs), dtype=np.float64)
    for bitstring, c in counts.items():
        b = bitstring[::-1]
        z = np.array([1.0 if ch == "0" else -1.0 for ch in b])
        for k, (i, j) in enumerate(pairs):
            exp2[k] += c * z[i] * z[j]
    exp2 /= shots
    corr = np.array([exp2[k] - exp1[i] * exp1[j]
                     for k, (i, j) in enumerate(pairs)], dtype=np.float64)
    return corr


def shadow_pauli_expectations(rotations: np.ndarray, outcomes: np.ndarray,
                               n_qubits: int) -> np.ndarray:
    """
    Estimate single-qubit Pauli expectations from shadow data.
    rotations: (S, n) int array  0=Z  1=X  2=Y
    outcomes:  (S, n) int array  0/1
    Returns (3, n) array: row 0=Z, 1=X, 2=Y expectations.
    """
    result = np.zeros((3, n_qubits), dtype=np.float64)
    counts = np.zeros((3, n_qubits), dtype=np.int64)
    signs = 1 - 2 * outcomes.astype(np.float64)  # (S, n): +1 if 0, -1 if 1
    for basis in range(3):
        mask = (rotations == basis)           # (S, n) bool
        for q in range(n_qubits):
            col_mask = mask[:, q]
            n_match = col_mask.sum()
            if n_match > 0:
                result[basis, q] = 3.0 * signs[col_mask, q].mean()
                counts[basis, q] = n_match
    return result


def shadow_pauli_correlators(rotations: np.ndarray, outcomes: np.ndarray,
                              n_qubits: int,
                              pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """
    Estimate all 6 connected two-qubit Pauli correlators from shadow data.
    Returns array of shape (6, len(pairs)):
      rows: ZZ, XX, YY, XY, XZ, YZ
    """
    exp1 = shadow_pauli_expectations(rotations, outcomes, n_qubits)
    signs = 1 - 2 * outcomes.astype(np.float64)  # (S, n)

    pauli_pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]  # ZZ XX YY XY XZ YZ
    result = np.zeros((6, len(pairs)), dtype=np.float64)

    for row, (pa, pb) in enumerate(pauli_pairs):
        for col, (i, j) in enumerate(pairs):
            mask = (rotations[:, i] == pa) & (rotations[:, j] == pb)
            n_match = mask.sum()
            if n_match > 0:
                exp2 = 9.0 * (signs[mask, i] * signs[mask, j]).mean()
                result[row, col] = exp2 - exp1[pa, i] * exp1[pb, j]

    return result


def collision_probability(counts: Counts) -> float:
    """
    Collision probability sum_x p(x)^2.
    """
    shots = total_shots(counts)
    if shots == 0:
        return 0.0
    s = 0.0
    for _, c in counts.items():
        p = c / shots
        s += p * p
    return float(s)


def connected_zz_correlations(
    counts: Counts,
    pairs: Sequence[Tuple[int, int]],
    n_qubits: int,
    reverse_bits: bool = True,
) -> np.ndarray:
    """
    Connected correlation:
      C_ij = <Z_i Z_j> - <Z_i><Z_j>

    where bit 0 -> z=+1, bit 1 -> z=-1.
    """
    shots = total_shots(counts)
    if shots == 0:
        return np.zeros(len(pairs), dtype=np.float64)

    # <Z_i> = 1 - 2 P(x_i=1)
    p1 = single_qubit_marginals(counts, n_qubits, reverse_bits=reverse_bits)
    z_mean = 1.0 - 2.0 * p1

    zz = np.zeros(len(pairs), dtype=np.float64)

    for bitstring, c in counts.items():
        b = _maybe_reverse(bitstring, reverse_bits)
        z = np.array([1.0 if ch == "0" else -1.0 for ch in b], dtype=np.float64)
        for k, (i, j) in enumerate(pairs):
            zz[k] += c * (z[i] * z[j])

    zz /= shots

    corr = np.zeros_like(zz)
    for k, (i, j) in enumerate(pairs):
        corr[k] = zz[k] - (z_mean[i] * z_mean[j])
    return corr


# ---------------------------
# Public API: dict + flattener
# ---------------------------

def summarize_counts_dict(
    counts: Counts,
    n_qubits: Optional[int] = None,
    reverse_bits_for_marginals: bool = True,
    zz_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Return a readable dictionary of statistics.
    """
    _validate_counts(counts)
    n = infer_n_qubits(counts) if n_qubits is None else int(n_qubits)
    shots = total_shots(counts)

    hw_hist = hamming_weight_histogram(counts, n)
    hw_mom = hamming_weight_moments(hw_hist)
    marg = single_qubit_marginals(counts, n, reverse_bits=reverse_bits_for_marginals)

    out: Dict[str, Any] = {
        "meta": {
            "n_qubits": n,
            "shots": shots,
            "reverse_bits_for_marginals": reverse_bits_for_marginals,
        },
        "hamming_weight": {
            "hist": hw_hist.tolist(),
            "mean": hw_mom["mean"],
            "var": hw_mom["var"],
        },
        "qubit_marginals": marg.tolist(),
        "parity_bias": parity_bias(counts),
    }

    if zz_pairs is not None:
        corr = connected_zz_correlations(counts, zz_pairs, n_qubits=n, reverse_bits=reverse_bits_for_marginals)
        out["zz_connected"] = {
            "pairs": [[int(i), int(j)] for (i, j) in zz_pairs],
            "values": corr.tolist(),
        }

    return out


def flatten_feature_dict(
    feat: Dict[str, Any],
    include_zz: bool = True,
) -> np.ndarray:
    """
    Flatten z_only feature dict into a fixed feature vector.
    Order: hw_hist | z_marginals | parity_bias | zz_connected
    """
    hw = np.array(feat["hamming_weight"]["hist"], dtype=np.float64)
    marg = np.array(feat["qubit_marginals"], dtype=np.float64)
    scalars = np.array([feat["parity_bias"]], dtype=np.float64)
    parts = [hw, marg, scalars]
    if include_zz and "zz_connected" in feat:
        zz = np.array(feat["zz_connected"]["values"], dtype=np.float64)
        parts.append(zz)
    return np.concatenate(parts)


def summarize_multi_basis(
    z_counts: Counts,
    x_counts: Counts,
    y_counts: Counts,
    n_qubits: int,
    pairs: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """
    Build feature vector from three global-basis measurement runs.
    Order: hw_hist | z_marg | parity_bias | zz_corr | x_marg | xx_corr | y_marg | yy_corr
    """
    n = n_qubits
    hw_hist = hamming_weight_histogram(z_counts, n)
    hw_mom  = hamming_weight_moments(hw_hist)
    z_marg  = single_qubit_marginals(z_counts, n)
    pb      = parity_bias(z_counts)
    zz_corr = pauli_connected_correlators(z_counts, n, pairs)
    x_marg  = pauli_marginals_from_counts(x_counts, n)
    xx_corr = pauli_connected_correlators(x_counts, n, pairs)
    y_marg  = pauli_marginals_from_counts(y_counts, n)
    yy_corr = pauli_connected_correlators(y_counts, n, pairs)

    return np.concatenate([
        hw_hist,
        np.array([hw_mom["mean"], hw_mom["var"]]),
        z_marg, np.array([pb]), zz_corr,
        x_marg, xx_corr,
        y_marg, yy_corr,
    ])


def summarize_shadows(
    rotations: np.ndarray,
    outcomes: np.ndarray,
    n_qubits: int,
    pairs: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """
    Build feature vector from classical shadow measurements.
    Order: z_marg | x_marg | y_marg | zz_corr | xx_corr | yy_corr | xy_corr | xz_corr | yz_corr
    """
    exp1 = shadow_pauli_expectations(rotations, outcomes, n_qubits)   # (3, n)
    corr = shadow_pauli_correlators(rotations, outcomes, n_qubits, pairs)  # (6, P)

    return np.concatenate([
        exp1[0], exp1[1], exp1[2],   # Z, X, Y single-qubit expectations
        corr[0], corr[1], corr[2],   # ZZ, XX, YY
        corr[3], corr[4], corr[5],   # XY, XZ, YZ
    ])



def prettyprint_features(data: dict, width: int = 60, bar_width: int = 40, decimals: int = 4) -> None:
    """
    Pretty-print the feature dictionary produced by summarize_counts_dict().

    Order:
      1) hamming_weight
      2) marginals
      3) parity_bias
      4) entropy
      5) collision
      6) zz pairs (optional)
    """
    import numpy as np

    def fmt(x):
        if x is None:
            return "None"
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.{decimals}f}"
        return str(x)

    def title(s: str):
        print("\n" + "=" * width)
        print(f"{s:^{width}}")
        print("=" * width)

    def section(s: str):
        print("\n" + f"[ {s} ]")

    title("J7 SAMPLING PROJECT RESULTS")

    # -------------------------
    # META (keep as-is at top)
    # -------------------------
    meta = data.get("meta", {})
    section("METADATA")
    print(f"  n_qubits:                   {fmt(meta.get('n_qubits'))}")
    print(f"  shots:                      {fmt(meta.get('shots'))}")
    print(f"  reverse_bits_for_marginals: {fmt(meta.get('reverse_bits_for_marginals'))}")

    # -------------------------
    # 1) HAMMING WEIGHT
    # -------------------------
    hw = data.get("hamming_weight", {})
    hist = hw.get("hist", [])
    section("HAMMING WEIGHT")
    print(f"  mean: {fmt(hw.get('mean'))} | var: {fmt(hw.get('var'))}")
    print("  hist (P(|x|=w)):")
    for w, p in enumerate(hist):
        p_float = float(p) if p is not None else 0.0
        bar = "█" * int(round(p_float * bar_width))
        print(f"    w={w:>2}: {p_float:.{decimals}f} {bar}")

    # -------------------------
    # 2) MARGINALS
    # -------------------------
    marg = data.get("qubit_marginals", [])
    section("QUBIT MARGINALS (P(x_i=1))")
    if len(marg) == 0:
        print("  (none)")
    else:
        header = "  qubit: " + " ".join([f"{i:>7d}" for i in range(len(marg))])
        values = "  p(1):  " + " ".join([f"{float(p):>7.{decimals}f}" for p in marg])
        print(header)
        print(values)

    # -------------------------
    # 3) PARITY BIAS
    # -------------------------
    section("PARITY BIAS")
    pb = data.get("parity_bias", None)
    if pb is not None:
        skew = "Even-skewed" if pb > 0 else ("Odd-skewed" if pb < 0 else "Neutral")
    else:
        skew = "Unknown"
    print(f"  parity_bias: {fmt(pb)} ({skew})")

    # -------------------------
    # 4) ENTROPY
    # -------------------------
    section("ENTROPY")
    print(f"  entropy_bits: {fmt(data.get('entropy_bits'))}")

    # -------------------------
    # 5) COLLISION
    # -------------------------
    section("COLLISION")
    print(f"  collision_prob: {fmt(data.get('collision_prob'))}")

    # -------------------------
    # 6) ZZ PAIRS (optional)
    # -------------------------
    if "zz_connected" in data:
        zz = data["zz_connected"]
        pairs = zz.get("pairs", [])
        vals = zz.get("values", [])
        section("CONNECTED ZZ CORRELATIONS")
        if len(pairs) == 0 or len(vals) == 0:
            print("  (present but empty)")
        else:
            for (i, j), v in zip(pairs, vals):
                print(f"  C({i},{j}) = {float(v):.{decimals}f}")

    # -------------------------
    # SUMMARY OF KEYS
    # -------------------------
    section("TOP-LEVEL KEYS")
    print("  " + ", ".join(sorted(list(data.keys()))))