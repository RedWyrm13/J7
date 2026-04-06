from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
def load_npz(filename, allow_pickle = False):
    data = np.load(filename, allow_pickle = allow_pickle)
    return data

# This returns a glob pattern that can be used with folder.glob() to pull all relevant files.
# Specify parameters you are interested in, leave others as '*' (wildcard).
def make_filename(circuit_family='*',
                  num_qubits='*',
                  num_ops='*',
                  shots_per_datapoint='*',
                  num_circuits='*',
                  resamples_per_circuit='*',
                  ):
    filename = (
        f"circuitFamily_{circuit_family}"
        f"_qubits{num_qubits}"
        f"_ops{num_ops}"
        f"_shotsPerDatapoint{shots_per_datapoint}"
        f"_numCircuits{num_circuits}"
        f"_resamplesPerCircuit{resamples_per_circuit}"
        "_masterSeed*.npz"
    )
    return filename

# Initialize 4 basic models for testing
def initialize_models():
    models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        solver='lbfgs'
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "SVM": SVC(kernel='rbf')
    }
    
    return models

def prepare_data(filename, folder):
    X_list = []
    y_list = []

    for file in folder.glob(filename):
        data = load_npz(file, allow_pickle=True)
        X_list.append(data['X'])
        y_list.append(data['y'])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


# Evaluates model accuracy over n_splits repeated random train/test splits.
# Returns (mean_accuracy, std_accuracy).
def eval_model(model, X, y, n_splits=10):
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 100_000, size=n_splits)
    scores = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=int(seed)
        )
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    return float(np.mean(scores)), float(np.std(scores))


def plot_qubits_dict(qubits_dict, filename=None, title=None):
    x_axis = sorted(qubits_dict.keys())
    model_names = list(next(iter(qubits_dict.values())).keys())

    for model_name in model_names:
        means = [qubits_dict[q][model_name][0] for q in x_axis]
        stds  = [qubits_dict[q][model_name][1] for q in x_axis]
        plt.errorbar(x_axis, means, yerr=stds, marker='o', capsize=3,
                     linewidth=1.5, label=model_name)

    plt.axhline(y=1/3, color="gray", linestyle="--", linewidth=0.8,
                alpha=0.6, label="Chance (0.33)")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Accuracy")
    plt.title(title or "Average Model Accuracy vs Number of Qubits")
    plt.legend()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()


def plot_combined(all_strategies, filename=None, title=None):
    """
    all_strategies: dict of {strategy_label: qubits_dict}
    where qubits_dict: {qubit_count: {model_name: (mean, std)}}
    Produces a 2x2 figure — one subplot per classifier, all strategies overlaid with error bars.
    """
    first_qd = next(iter(all_strategies.values()))
    model_names = list(next(iter(first_qd.values())).keys())

    strategy_styles = {
        "Z-only":      {"color": "tab:blue",   "marker": "o"},
        "Multi-basis": {"color": "tab:orange",  "marker": "s"},
        "Shadows":     {"color": "tab:green",   "marker": "^"},
        "NN":          {"color": "tab:red",     "marker": "D"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(model_names):
        ax = axes_flat[idx]
        for label, qubits_dict in all_strategies.items():
            x_axis = sorted(qubits_dict.keys())
            means = [qubits_dict[q][model_name][0] for q in x_axis]
            stds  = [qubits_dict[q][model_name][1] for q in x_axis]
            style = strategy_styles.get(label, {})
            ax.errorbar(x_axis, means, yerr=stds,
                        marker=style.get("marker", "o"),
                        color=style.get("color"),
                        capsize=3, linewidth=1.5, label=label)
        ax.axhline(y=1/3, color="gray", linestyle="--", linewidth=0.8,
                   alpha=0.6, label="Chance (0.33)")
        ax.set_title(model_name)
        ax.set_xlabel("Number of Qubits")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle(title or "Classifier Accuracy vs Qubit Count by Measurement Strategy")
    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()


def _build_qubits_dict(folder, models, shots_base, qubits):
    qubits_dict = {}
    for qubit in qubits:
        shots = shots_base * qubit ** 2
        filename = make_filename(num_qubits=qubit, shots_per_datapoint=shots)
        try:
            X, y = prepare_data(filename=filename, folder=folder)
            results = {}
            for model_name, model in models.items():
                results[model_name] = eval_model(model, X, y)
            qubits_dict[qubit] = results
        except Exception as e:
            print(f"Skipping qubits={qubit} in {folder}: {e}")
    return qubits_dict


def main():
    shots_base = 16
    qubits = list(range(4, 21))

    measurement_modes = {
        "Z-only":      Path("../data/quantum"),
        "Multi-basis": Path("../data/quantum_mb"),
        "Shadows":     Path("../data/quantum_shadows"),
        "NN":          Path("../data/quantum_nn"),
    }

    for mode_label, folder in measurement_modes.items():
        if not folder.exists():
            print(f"Skipping {mode_label}: {folder} not found.")
            continue

        models = initialize_models()
        qubits_dict = _build_qubits_dict(folder, models, shots_base, qubits)

        if not qubits_dict:
            print(f"No data loaded for {mode_label}.")
            continue

        print(f"\n=== {mode_label} — best accuracy per model ===")
        for model_name in next(iter(qubits_dict.values())).keys():
            best_mean, best_std = max(qubits_dict[q][model_name] for q in qubits_dict)
            print(f"  {model_name}: {best_mean:.4f} ± {best_std:.4f}")

        safe_label = mode_label.lower().replace("-", "_").replace(" ", "_")
        plot_qubits_dict(
            qubits_dict=qubits_dict,
            filename=f"../latex/images/qubit_accuracy_{safe_label}.png",
            title=f"Model Accuracy vs Qubit Count ({mode_label})",
        )
        print(f"Plot saved to ../latex/images/qubit_accuracy_{safe_label}.png")
            
if __name__ == '__main__':
    main()