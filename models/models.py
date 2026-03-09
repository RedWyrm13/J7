from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
def load_npz(filename, allow_pickle = False):
    data = np.load(filename, allow_pickle = allow_pickle)
    return data

# This returns a filename that can be used in conjunction with the folder.glob(filename) function to pull all relevant files.
# Specify parameters you are interested in, leave the others blank.
def make_filename(circuit_family = '_',
                  num_qubits = '_',
                  num_ops = '_',
                  shots_per_datapoint = '_',
                  num_circuits = '_',
                  resamples_per_circuit = '_',
                  ):
    
    filename = (
        f"*circuitFamily*{circuit_family}"
        f"*qubits*{num_qubits}"
        f"*ops*{num_ops}"
        f"*shotsPer*{shots_per_datapoint}"
        f"*numCir*{num_circuits}"
        f"*resamples*{resamples_per_circuit}"
        "*.npz"
)
    return filename

# Initialize 4 basic models for testing
def initialize_models():
    models = {
    "Logistic Regression":LogisticRegression(
        max_iter=1000,
        solver='lbfgs'
    ),
    "DecisionTreeClassifier": DecisionTreeClassifier(
        max_depth= None,
        random_state=42
    ),
    "Gradient Boosting CLassifier" : GradientBoostingClassifier(),
    
    "Extra Trees Classifier" : ExtraTreesClassifier(n_estimators=200)
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
    folder = Path("../data")
    models = initialize_models()
    shots = (256, 512, 1024)
    qubits = list(range(4, 11))

    shots_dict = build_average_dict(
        models=models,
        folder=folder,
        outer_values=shots,
        inner_values=qubits,
        outer_name="shots"
    )

    qubits_dict = build_average_dict(
        models=models,
        folder=folder,
        outer_values=qubits,
        inner_values=shots,
        outer_name="qubits"
    )

    plot_shots_dict(shots_dict=shots_dict, filename="shots_test.png")
    plot_qubits_dict(qubits_dict=qubits_dict, filename="qubit_test.png")
            
if __name__ == '__main__':
    main()