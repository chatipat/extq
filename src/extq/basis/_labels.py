import numpy as np
import scipy.sparse


def labels_to_basis(labels, num=None, sparse=True):
    if num is None:
        num = max(np.max(indices) for indices in labels) + 1
    basis = []
    for indices in labels:
        basis.append(_labels_to_basis(indices, num, sparse=sparse))
    return basis


def _labels_to_basis(indices, cols, sparse=True, mask=None):
    rows = len(indices)
    row_ind = np.arange(rows)
    col_ind = indices
    if mask is not None:
        row_ind = row_ind[mask]
        col_ind = col_ind[mask]
    assert np.all(col_ind >= 0) and np.all(col_ind < cols)
    if sparse:
        return scipy.sparse.csr_matrix(
            (np.ones(len(row_ind)), (row_ind, col_ind)), shape=(rows, cols)
        )
    else:
        x = np.zeros((rows, cols))
        x[row_ind, col_ind] = 1.0
        return x


def renumber_labels(labels):
    unique = np.unique(np.concatenate(labels))
    assert np.min(unique) >= 0
    renumber = np.empty(np.max(unique) + 1, dtype=unique.dtype)
    renumber[unique] = np.arange(len(unique))
    return [renumber[indices] for indices in labels]


def renumber_basis(basis):
    mask = sum(np.ravel((x != 0).sum(axis=0)).astype(bool) for x in basis)
    return [x[:, mask] for x in basis]
