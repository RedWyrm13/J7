import numpy as np

# Helper functions to go along with AI training 

def load_npz(filename, allow_pickle = False):
    data = np.load(filename, allow_pickle = allow_pickle)
    return data