def rearrange(indices, labels):
    """Rearrange a list of indices according to a list of labels"""
    assert len(indices) == len(labels)
    indices_by_label = {label: [] for label in set(labels)}
    for i, label in zip(indices, labels):
        indices_by_label[label].append(i)
    while len(indices):
        for label in indices_by_label:
            if len(indices_by_label[label]):
                yield indices_by_label[label].pop(0)
                indices.pop(0)
            else:
                del indices_by_label[label]
                break

def matricize(X):
    """Transform a list of lists into a matrix with 1s and 0s"""
    labels = set([l for x in X for l in x])
    matrix = np.zeros((len(X), len(labels)))
    for i, x in enumerate(X):
        for l in x:
            matrix[i, list(labels).index(l)] = x.count(l)
    return matrix
        