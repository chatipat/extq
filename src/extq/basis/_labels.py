"""Functions for manipulating labels."""

import numpy as np
import scipy.sparse
from more_itertools import zip_equal

__all__ = [
    "labels_to_basis",
    "renumber_labels",
    "renumber_basis",
]


def labels_to_basis(labels, num=None, sparse=True, in_domain=None):
    """
    Construct a basis of indicator functions given labels.

    Parameters
    ----------
    labels : sequence of (n_frames[i],) ndarray of int
        Label at each frame. Labels are assumed to be consecutive
        integers starting from zero.
    num : int, optional
        Number of labels / basis functions. If None (default), determine
        this from `labels`.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.
    in_domain : sequence of (n_frames[i],) ndarray of bool, optional
        Whether each frame is in the domain.

    Returns
    -------
    list of (n_frames[i], num) {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    if num is None:
        num = _num(labels)
    basis = []
    if in_domain is None:
        for indices in labels:
            basis.append(_labels_to_basis(indices, num, sparse=sparse))
    else:
        for indices, mask in zip_equal(labels, in_domain):
            basis.append(
                _labels_to_basis(indices, num, sparse=sparse, mask=mask)
            )
    return basis


def _num(labels):
    """Find the number of clusters given labeled trajectories."""
    return max(np.max(indices) for indices in labels if len(indices) > 0) + 1


def _labels_to_basis(indices, cols, sparse=True, mask=None):
    """
    Construct a basis of indicator functions given labels for a single
    trajectory.

    Parameters
    ----------
    indices : (n_frames,) ndarray of int
        Label at each frame. Labels are assumed to be consecutive
        integers starting from zero.
    cols : int
        Number of labels / basis functions.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.
    mask : (n_frames,) ndarray of bool, optional
        Whether each frame in the domain. Basis functions evaluate to
        zero for frames outside of the domain. If None (default), assume
        that every frame is in the domain.

    Returns
    -------
    list of (n_frames, cols) {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
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
    """
    Make labels consecutive (starting from zero).

    Parameters
    ----------
    labels : list of (n_frames[i],) ndarray of int
        Label at each frame.

    Returns
    -------
    list of (n_frames[i],) ndarray of int
        New label at each frame. These new labels are consecutive,
        starting from zero, and each label appears at least once.

    """
    unique = np.unique(np.concatenate(labels))
    assert np.min(unique) >= 0
    renumber = np.empty(np.max(unique) + 1, dtype=unique.dtype)
    renumber[unique] = np.arange(len(unique))
    return [renumber[indices] for indices in labels]


def renumber_basis(basis):
    """
    Remove basis functions that evaluate to zero at every frame.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis functions at each frame.

    Returns
    -------
    list of (n_frames[i], n_nonzero_basis) {ndarray, sparse matrix} of float
        Basis functions that are nonzero in at least one frame of the
        data set.

    """
    mask = sum(np.ravel((x != 0).sum(axis=0)).astype(bool) for x in basis)
    return [x[:, mask] for x in basis]
