import numpy as np

def matricize(X):
    """Transform a list of lists into a matrix with 1s and 0s"""
    labels = set([l for x in X for l in x])
    matrix = np.zeros((len(X), len(labels)))
    for i, x in enumerate(X):
        for l in x:
            matrix[i, list(labels).index(l)] = x.count(l)
    return matrix
        