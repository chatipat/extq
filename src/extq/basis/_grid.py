"""Functions for constructing grid bases."""

import numpy as np

from ._labels import _labels_to_basis

__all__ = [
    "grid1d_labels",
    "grid2d_labels",
    "grid3d_labels",
    "grid1d_basis",
    "grid2d_basis",
    "grid3d_basis",
    "grid1d_domain_basis",
    "grid2d_domain_basis",
    "grid3d_domain_basis",
]


def grid1d_labels(cv, edges):
    """
    Label each frame with the index on a 1D grid.

    Parameters
    ----------
    cv : sequence of (n_frames[i],) ndarray of float
        Collective variable at each frame.
    edges : 1D array-like of float
        Bin edges for the collective variable.

    Returns
    -------
    list of (n_frames[i],) ndarray of int
        Grid index at each frame.

    """
    labels = []
    for v in cv:
        labels.append(_labels1(v, edges))
    return labels


def grid2d_labels(cv1, cv2, edges1, edges2):
    """
    Label each frame with the flat index on a 2D grid.

    Parameters
    ----------
    cv1, cv2 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    edges1, edges2 : 1D array-like of float
        Bin edges for each collective variable.

    Returns
    -------
    list of (n_frames[i],) ndarray of int
        Grid index at each frame.

    """
    labels = []
    for v1, v2 in zip(cv1, cv2):
        labels.append(_labels2(v1, v2, edges1, edges2))
    return labels


def grid3d_labels(cv1, cv2, cv3, edges1, edges2, edges3):
    """
    Label each frame with the flat index on a 3D grid.

    Parameters
    ----------
    cv1, cv2, cv3 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    edges1, edges2, edges3 : 1D array-like of float
        Bin edges for each collective variable.

    Returns
    -------
    list of (n_frames[i],) ndarray of int
        Grid index at each frame.

    """
    labels = []
    for v1, v2, v3 in zip(cv1, cv2, cv3):
        labels.append(_labels3(v1, v2, v3, edges1, edges2, edges3))
    return labels


def grid1d_basis(cv, edges, sparse=True):
    """
    Construct a basis of indicator functions on a 1D grid.

    Parameters
    ----------
    cv : sequence of (n_frames[i],) ndarray of float
        Collective variable at each frame.
    edges : 1D array-like of float
        Bin edges for the collective variable.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    num = len(edges) + 1
    basis = []
    for v in cv:
        indices = _labels1(v, edges)
        basis.append(_labels_to_basis(indices, num, sparse=sparse))
    return basis


def grid2d_basis(cv1, cv2, edges1, edges2, sparse=True):
    """
    Construct a basis of indicator functions on a 2D grid.

    Parameters
    ----------
    cv1, cv2 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    edges1, edges2 : 1D array-like of float
        Bin edges for each collective variable.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    num = (len(edges1) + 1) * (len(edges2) + 1)
    basis = []
    for v1, v2 in zip(cv1, cv2):
        indices = _labels2(v1, v2, edges1, edges2)
        basis.append(_labels_to_basis(indices, num, sparse=sparse))
    return basis


def grid3d_basis(cv1, cv2, cv3, edges1, edges2, edges3, sparse=True):
    """
    Construct a basis of indicator functions on a 3D grid.

    Parameters
    ----------
    cv1, cv2, cv3 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    edges1, edges2, edges3 : 1D array-like of float
        Bin edges for each collective variable.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    num = (len(edges1) + 1) * (len(edges2) + 1) * (len(edges3) + 1)
    basis = []
    for v1, v2, v3 in zip(cv1, cv2, cv3):
        indices = _labels3(v1, v2, v3, edges1, edges2, edges3)
        basis.append(_labels_to_basis(indices, num, sparse=sparse))
    return basis


def grid1d_domain_basis(cv, in_domain, edges, sparse=True):
    """
    Construct a basis of indicator functions on a 1D grid within a
    specified domain.

    Parameters
    ----------
    cv : sequence of (n_frames[i],) ndarray of float
        Collective variable at each frame.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain. The basis is zero outside
        of the domain.
    edges : 1D array-like of float
        Bin edges for the collective variable.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    num = len(edges) + 1
    basis = []
    for v, d in zip(cv, in_domain):
        indices = _labels1(v, edges)
        basis.append(_labels_to_basis(indices, num, sparse=sparse, mask=d))
    return basis


def grid2d_domain_basis(cv1, cv2, in_domain, edges1, edges2, sparse=True):
    """
    Construct a basis of indicator functions on a 2D grid within a
    specified domain.

    Parameters
    ----------
    cv1, cv2 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain. The basis is zero outside
        of the domain.
    edges1, edges2 : 1D array-like of float
        Bin edges for each collective variable.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    num = (len(edges1) + 1) * (len(edges2) + 1)
    basis = []
    for v1, v2, d in zip(cv1, cv2, in_domain):
        indices = _labels2(v1, v2, edges1, edges2)
        basis.append(_labels_to_basis(indices, num, sparse=sparse, mask=d))
    return basis


def grid3d_domain_basis(
    cv1, cv2, cv3, in_domain, edges1, edges2, edges3, sparse=True
):
    """
    Construct a basis of indicator functions on a 3D grid within a
    specified domain.

    Parameters
    ----------
    cv1, cv2, cv3 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain. The basis is zero outside
        of the domain.
    edges1, edges2, edges3 : 1D array-like of float
        Bin edges for each collective variable.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    num = (len(edges1) + 1) * (len(edges2) + 1) * (len(edges3) + 1)
    basis = []
    for v1, v2, v3, d in zip(cv1, cv2, cv3, in_domain):
        indices = _labels3(v1, v2, v3, edges1, edges2, edges3)
        basis.append(_labels_to_basis(indices, num, sparse=sparse, mask=d))
    return basis


def _labels1(v, edges):
    """Return the index on a 1D grid."""
    return np.searchsorted(edges, v)


def _labels2(v1, v2, edges1, edges2):
    """Return the flat index on a 2D grid."""
    indices1 = np.searchsorted(edges1, v1)
    indices2 = np.searchsorted(edges2, v2)
    return np.ravel_multi_index(
        (indices1, indices2), (len(edges1) + 1, len(edges2) + 1)
    )


def _labels3(v1, v2, v3, edges1, edges2, edges3):
    """Return the flat index on a 3D grid."""
    indices1 = np.searchsorted(edges1, v1)
    indices2 = np.searchsorted(edges2, v2)
    indices3 = np.searchsorted(edges3, v3)
    return np.ravel_multi_index(
        (indices1, indices2, indices3),
        (len(edges1) + 1, len(edges2) + 1, len(edges3) + 1),
    )
