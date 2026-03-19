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

        data = load_npz(file, allow_pickle= True)
        
        X_list.append(data['X'])
        y_list.append(data['y'])

    X = np.concat(X_list, axis = 0)
    y = np.concat(y_list, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
    return X_train, y_train, X_test, y_test

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def get_accuracy(models, folder, shot, qubit):
    filename = make_filename(
        shots_per_datapoint=shot,
        num_qubits=qubit
    )

    X_train, y_train, X_test, y_test = prepare_data(
        filename=filename,
        folder=folder
    )

    results = {}
    for model_name, model in models.items():
        acc = eval_model(model, X_train, y_train, X_test, y_test)
        results[model_name] = acc

    return results


def build_average_dict(models, folder, outer_values, inner_values, outer_name):
    avg_dict = {}

    for outer in outer_values:
        models_dict = {}

        for model_name in models.keys():
            accs = []

            for inner in inner_values:
                if outer_name == "shots":
                    shot = outer
                    qubit = inner
                elif outer_name == "qubits":
                    shot = inner
                    qubit = outer
                else:
                    raise ValueError("outer_name must be 'shots' or 'qubits'")

                results = get_accuracy(models, folder, shot, qubit)
                accs.append(results[model_name])

            models_dict[model_name] = np.mean(accs)

        avg_dict[outer] = models_dict

    return avg_dict


def plot_shots_dict(shots_dict, filename=None):
    x_axis = sorted(shots_dict.keys())
    model_names = list(next(iter(shots_dict.values())).keys())

    for model_name in model_names:
        y_axis = [shots_dict[shot][model_name] for shot in x_axis]
        plt.plot(x_axis, y_axis, marker='o', label=model_name)

    plt.xlabel("Shot Count")
    plt.ylabel("Average Accuracy")
    plt.title("Average Model Accuracy vs Shot Count")
    plt.legend()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def plot_qubits_dict(qubits_dict, filename=None):
    x_axis = sorted(qubits_dict.keys())
    model_names = list(next(iter(qubits_dict.values())).keys())

    for model_name in model_names:
        y_axis = [qubits_dict[qubit][model_name] for qubit in x_axis]
        plt.plot(x_axis, y_axis, marker='o', label=model_name)

    plt.xlabel("Number of Qubits")
    plt.ylabel("Average Accuracy")
    plt.title("Average Model Accuracy vs Number of Qubits")
    plt.legend()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def main():
    folder = Path("../data/quantum")
    models = initialize_models()
    shots_base = 16
    qubits = list(range(4, 21))

    # Shots scale quadratically with qubit count: shots = shots_base * n_qubits^2
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
            print(f"Skipping qubits={qubit}: {e}")

    plot_qubits_dict(qubits_dict=qubits_dict, filename="qubit_accuracy.png")
            
if __name__ == '__main__':
    main()