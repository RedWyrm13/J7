from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler

def build_and_transpile(circuit, simulator = None):
    """
    circuit: qiskit QuantumCircuit
    simulator AerSimulator configured with specific options. If None, default simulator will be used
    """
    circuit.remove_final_measurements()
    circuit.measure_all()
    
    if simulator == None:
        simulator = AerSimulator()
        
    transpiled_circuit = transpile(backend = simulator, circuits=circuit)
    
    return transpiled_circuit

def run_sampler(transpiled_circuit, sim = None, shots = None):
    if sim == None:
        sim = AerSimulator()
        
    result = sim.run(transpiled_circuit, shots = shots).result()
    
    
    return result.get_counts()