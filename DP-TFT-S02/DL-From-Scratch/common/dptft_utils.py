import numpy as np

def flatten(data, flatten_size):
    return data.reshape(-1, flatten_size)

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
