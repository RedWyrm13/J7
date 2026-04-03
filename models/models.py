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

# Evaluates model accuracy
def eval_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy

def prepare_data(filename, folder):
    X_list = []
    y_list = []

    for file in folder.glob(filename):
        data = load_npz(file, allow_pickle=True)
        X_list.append(data['X'])
        y_list.append(data['y'])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, X_test, y_test


def plot_qubits_dict(qubits_dict, filename=None, title=None):
    x_axis = sorted(qubits_dict.keys())
    model_names = list(next(iter(qubits_dict.values())).keys())

    for model_name in model_names:
        y_axis = [qubits_dict[qubit][model_name] for qubit in x_axis]
        plt.plot(x_axis, y_axis, marker='o', label=model_name)

    plt.xlabel("Number of Qubits")
    plt.ylabel("Average Accuracy")
    plt.title(title or "Average Model Accuracy vs Number of Qubits")
    plt.legend()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def _build_qubits_dict(folder, models, shots_base, qubits):
    qubits_dict = {}
    for qubit in qubits:
        shots = shots_base * qubit ** 2
        filename = make_filename(num_qubits=qubit, shots_per_datapoint=shots)
        try:
            X_train, y_train, X_test, y_test = prepare_data(filename=filename, folder=folder)
            results = {}
            for model_name, model in models.items():
                results[model_name] = eval_model(model, X_train, y_train, X_test, y_test)
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
            best = max(qubits_dict[q][model_name] for q in qubits_dict)
            print(f"  {model_name}: {best:.4f}")

        safe_label = mode_label.lower().replace("-", "_").replace(" ", "_")
        plot_qubits_dict(
            qubits_dict=qubits_dict,
            filename=f"qubit_accuracy_{safe_label}.png",
            title=f"Model Accuracy vs Qubit Count ({mode_label})",
        )
            
if __name__ == '__main__':
    main()