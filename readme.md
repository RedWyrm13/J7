# J7 Sampling Project Documentation

## Overview
This repository is for J7 Sampling project. We aim to generate different probability distributions related to different circuit classes such as instantaneous quantum polynomial (IQP) circuits, clifford circuits, and potentially more. We will then train a classification model to identify these different circuit familes based on their distributions. We believe we can then map this to relevant work in classification for J7 through spectrum monitoring by mapping spectrum activity to different distributions and classifying it based on the distribution's stats. There are three main folders: (I) ```circuit_builders```,  (II) ```data```, and (III) ```utils```.  Here is a brief overview of each

- ```circuit_builders``` - This folder holds logic for building the different circuit classes such as iqp and clifford circuits. More may come later. The ```builder.py``` file contains the logic for building the correct circuit based off the yaml file. Specifically, it is the "label" variable in the yaml file/ cfgCircuit class which determines the circuit that is built. 

- ```data``` - This folder contains any data I have generated. The name convention is as follows: "./data/circuitFamily_{cfg.label}_qubits{cfg.n_qubits}_ops{cfg.n_ops}_shotsPerDatapoint{cfg.shots_per_datapoint}_numCircuits{cfg.n_circuits}_resamplesPerCircuit{cfg.resamples_per_circuit}_masterSeed{cfg.master_seed}{file_extension}" The logic for this is located in ```utils.utils.make_filename```.

```utils```- This folder contains any helper logic that is associated with the pipeline, which is not directly building the circuit such as transpilation, sampling, and getting certain statistics. Below is an overview of the statistics obtained for each distribution.

I think I can implement this in three ways. (I) comparing probability distributions, (II) with state tomography, which has exponential overhead, and (III) classical shadows. Then we can compare which of the three are better,

## Generating data
The ```circuit_configuration.yaml``` file contains the following seven fields:
- label: The circuit family. Currently only IQP and Clifford are supported
- n_qubits: The number of qubits in the circuit. For HPC simulations it is recommended to not go above 25.
- n_ops: Number of operations applied. Each circuit is comprised of a family of gates. We randomly select a gate and qubit(s) to apply it to a number of times equal to n_ops
- n_circuits: The number of unique circuits in this family to sample from.
- resamples_per_circuit: How many samples are obtained from each circuit
- shots_per_datapoint: number of shots which reveal the distribution.
- master_seed: Used for reproducibility

Once the yaml file is populated appropriately, from the terminal, run (linux)
```
python3 generate_distributions.py circuit_configuration.yaml
```
This will create an npz file with the naming convention, discussed above, in the ```data``` folder.

## Features

### Metadata

* **`n_qubits`**: The total number of qubits (i.e., the length of each sampled bitstring).
* **`shots`**: The number of measurement samples used to estimate the probability distribution.
* **`reverse_bits_for_marginals`**: A boolean flag indicating whether bitstrings were reversed to align qubit indexing with Qiskit’s internal ordering (LSB vs. MSB).

---

### Hamming Weight

The **Hamming Weight** is the count of '1's in a bitstring. For example, the string `0110` has a Hamming weight of 2, while `0001` has a Hamming weight of 1.

We track the probability distribution of Hamming weights across all samples. For a set of 4-bit strings where each possible Hamming weight (0 through 4) occurs in 20% of the samples, the distribution list would be `[0.2, 0.2, 0.2, 0.2, 0.2]`, where the index corresponds to the weight.

* `hamming_weight.hist`
* `hamming_weight.mean`
* `hamming_weight.var`

> **To Do:** Add support for median, mode, and standard deviation.

---

### Qubit Marginals

A **Qubit Marginal** is a value between 0 and 1 representing the probability that a specific qubit is measured in the $|1\rangle$ state.
* **0.0**: The qubit is always measured as `0`.
* **1.0**: The qubit is always measured as `1`.
* **0.5**: The qubit is unbiased (measured as `0` or `1` with equal frequency).

Marginals are presented as a list where the $i$-th entry corresponds to the $i$-th qubit.


---

### Parity Bias

**Parity Bias** measures the imbalance between bitstrings with even parity and those with odd parity. It is calculated as:
$$P_{bias} = N_{even} - N_{odd}$$
* **Positive values**: The distribution skews toward an even number of '1's.
* **Negative values**: The distribution skews toward an odd number of '1's.

---

### Shannon Entropy

$$H(P) = -\sum_{x} p(x) \log_2 p(x)$$

**Shannon Entropy** quantifies the overall randomness of the samples. 
* **High Entropy**: Indicates a uniform distribution (maximum randomness).
* **Low Entropy**: Indicates a concentrated distribution where samples are clustered around a few specific states.

---

### Collision Probability

$$\mathcal{C} = \sum_{x} p(x)^2$$

The **Collision Probability** is the purity of the distribution, often viewed as the inverse behavior of Shannon Entropy. 
* **High $\mathcal{C}$**: The distribution is highly concentrated (low uncertainty).
* **Low $\mathcal{C}$**: The distribution is spread out/random (high uncertainty).

---

### Connected ZZ Correlations

By default, the ZZ correlations measured are linear. That is qubits 0,1; 1,2; 2,3; ... n-1,n.

The connected correlation between two qubits $i$ and $j$ measures the statistical dependence between them, subtracting the contribution of their individual biases:

$$C_{ij} = \langle Z_i Z_j \rangle - \langle Z_i \rangle \langle Z_j \rangle$$

Where $Z \in \{1, -1\}$ represents the expectation value of the qubit in the computational basis.