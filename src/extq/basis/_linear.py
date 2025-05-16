"""Functions for linear/bilinear/trilinear bases."""

import numpy as np
import scipy as sp


def linear_basis(cv, nodes):
    n_basis = len(nodes)
    out = []
    for v in cv:
        data, row, col = _linear(v, nodes)
        out.append(
            sp.sparse.csr_array(
                (np.ravel(data), (np.ravel(row), np.ravel(col))),
                shape=(len(data), n_basis),
            )
        )
    return out


def bilinear_basis(cv1, cv2, nodes1, nodes2):
    n_basis = len(nodes1) * len(nodes2)
    out = []
    for v1, v2 in zip(cv1, cv2, strict=True):
        data, row, col = _bilinear(v1, v2, nodes1, nodes2)
        out.append(
            sp.sparse.csr_array(
                (np.ravel(data), (np.ravel(row), np.ravel(col))),
                shape=(len(data), n_basis),
            )
        )
    return out


def trilinear_basis(cv1, cv2, cv3, nodes1, nodes2, nodes3):
    n_basis = len(nodes1) * len(nodes2) * len(nodes3)
    out = []
    for v1, v2, v3 in zip(cv1, cv2, cv3, strict=True):
        data, row, col = _trilinear(v1, v2, v3, nodes1, nodes2, nodes3)
        out.append(
            sp.sparse.csr_array(
                (np.ravel(data), (np.ravel(row), np.ravel(col))),
                shape=(len(data), n_basis),
            )
        )
    return out


def _linear(cv, nodes):
    (n,) = cv.shape
    (k,) = nodes.shape

    indices = np.searchsorted(nodes, cv)

    mask0 = indices == 0
    mask1 = indices == k
    mask = ~(mask0 | mask1)

    d0 = np.empty(n)
    d0[mask] = cv[mask] - nodes[indices[mask] - 1]
    d0[mask0] = 0
    d0[mask1] = 1

    d1 = np.empty(n)
    d1[mask] = nodes[indices[mask]] - cv[mask]
    d1[mask0] = 1
    d1[mask1] = 0

    data = np.empty((n, 2))
    data[:, 0] = d1 / (d0 + d1)
    data[:, 1] = d0 / (d0 + d1)

    row = np.broadcast_to(np.arange(n)[:, np.newaxis], (n, 2))

    col = np.empty((n, 2), np.int_)
    col[:, 0] = np.maximum(indices - 1, 0)
    col[:, 1] = np.minimum(indices, k - 1)

    return data, row, col


def _bilinear(cv1, cv2, nodes1, nodes2):
    assert cv1.shape == cv2.shape
    (n,) = cv1.shape
    (k1,) = nodes1.shape
    (k2,) = nodes2.shape

    data1, row1, col1 = _linear(cv1, nodes1)
    data2, row2, col2 = _linear(cv2, nodes2)

    data = data1[:, :, np.newaxis] * data2[:, np.newaxis, :]
    row = np.broadcast_to(np.arange(n)[:, np.newaxis, np.newaxis], (n, 2, 2))
    col = np.ravel_multi_index(
        (col1[:, :, np.newaxis], col2[:, np.newaxis, :]), (k1, k2)
    )

    return data, row, col


def _trilinear(cv1, cv2, cv3, nodes1, nodes2, nodes3):
    assert cv1.shape == cv2.shape, cv3.shape
    (n,) = cv1.shape
    (k1,) = nodes1.shape
    (k2,) = nodes2.shape
    (k3,) = nodes3.shape

    data1, row1, col1 = _linear(cv1, nodes1)
    data2, row2, col2 = _linear(cv2, nodes2)
    data3, row3, col3 = _linear(cv3, nodes3)

    data = (
        data1[:, :, np.newaxis, np.newaxis]
        * data2[:, np.newaxis, :, np.newaxis]
        * data3[:, np.newaxis, np.newaxis, :]
    )
    row = np.broadcast_to(
        np.arange(n)[:, np.newaxis, np.newaxis, np.newaxis], (n, 2, 2, 2)
    )
    col = np.ravel_multi_index(
        (
            col1[:, :, np.newaxis, np.newaxis],
            col2[:, np.newaxis, :, np.newaxis],
            col3[:, np.newaxis, np.newaxis, :],
        ),
        (k1, k2, k3),
    )

    return data, row, col
