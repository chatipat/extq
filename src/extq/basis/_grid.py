"""Functions for constructing grid bases."""

import numpy as np
from more_itertools import zip_equal

from ._labels import labels_to_basis

__all__ = [
    "grid1d_labels",
    "grid2d_labels",
    "grid3d_labels",
    "grid1d_basis",
    "grid2d_basis",
    "grid3d_basis",
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
    return grid_labels(_stack(cv), (edges,))


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
    return grid_labels(_stack(cv1, cv2), (edges1, edges2))


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
    return grid_labels(_stack(cv1, cv2, cv3), (edges1, edges2, edges3))


def grid_labels(cvs, edges):
    """
    Label each frame with the flat index on an n-dimensional grid.

    Parameters
    ----------
    cvs : sequence of (n_frames[i], n_cvs) ndarray of float
        Collective variables at each frame.
    edges : (n_cvs, n_edges[k]) array-like of float
        Bin edges for each collective variable.

    Returns
    -------
    list of (n_frames[i],) ndarray of int
        Grid index at each frame.

    """
    return [_labels(vs, edges) for vs in cvs]


def grid1d_basis(cv, edges, sparse=True, in_domain=None):
    """
    Construct a basis of indicator functions on a 1D grid within a
    specified domain.

    Parameters
    ----------
    cv : sequence of (n_frames[i],) ndarray of float
        Collective variable at each frame.
    edges : 1D array-like of float
        Bin edges for the collective variable.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.
    in_domain : sequence of (n_frames[i],) ndarray of bool, optional
        Whether each frame is in the domain. The basis is zero outside
        of the domain.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    num = len(edges) + 1
    labels = grid1d_labels(cv, edges)
    return labels_to_basis(labels, num, sparse=sparse, in_domain=in_domain)


def grid2d_basis(cv1, cv2, edges1, edges2, sparse=True, in_domain=None):
    """
    Construct a basis of indicator functions on a 2D grid within a
    specified domain.

    Parameters
    ----------
    cv1, cv2 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    edges1, edges2 : 1D array-like of float
        Bin edges for each collective variable.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.
    in_domain : sequence of (n_frames[i],) ndarray of bool, optional
        Whether each frame is in the domain. The basis is zero outside
        of the domain.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    num = (len(edges1) + 1) * (len(edges2) + 1)
    labels = grid2d_labels(cv1, cv2, edges1, edges2)
    return labels_to_basis(labels, num, sparse=sparse, in_domain=in_domain)


def grid3d_basis(
    cv1, cv2, cv3, edges1, edges2, edges3, sparse=True, in_domain=None
):
    """
    Construct a basis of indicator functions on a 3D grid within a
    specified domain.

    Parameters
    ----------
    cv1, cv2, cv3 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    edges1, edges2, edges3 : 1D array-like of float
        Bin edges for each collective variable.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.
    in_domain : sequence of (n_frames[i],) ndarray of bool, optional
        Whether each frame is in the domain. The basis is zero outside
        of the domain.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    num = (len(edges1) + 1) * (len(edges2) + 1) * (len(edges3) + 1)
    labels = grid3d_labels(cv1, cv2, cv3, edges1, edges2, edges3)
    return labels_to_basis(labels, num, sparse=sparse, in_domain=in_domain)


def grid_basis(cvs, edges, sparse=True, in_domain=None):
    """
    Construct a basis of indicator functions on an n-dimensional grid
    within a specified domain.

    Parameters
    ----------
    cvs : sequence of (n_frames[i], n_cvs) ndarray of float
        Collective variables at each frame.
    edges : (n_cvs, n_edges[k]) array-like of float
        Bin edges for each collective variable.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.
    in_domain : sequence of (n_frames[i],) ndarray of bool, optional
        Whether each frame is in the domain. The basis is zero outside
        of the domain.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    num = np.product([len(e) + 1 for e in edges])
    labels = grid_labels(cvs, edges)
    return labels_to_basis(labels, num, sparse=sparse, in_domain=in_domain)


def _labels(v_list, edges_list):
    """Return the flat index on an n-dimensional grid."""
    indices = []
    shape = []
    for v, edges in zip_equal(v_list.T, edges_list):
        indices.append(np.searchsorted(edges, v))
        shape.append(len(edges) + 1)
    return np.ravel_multi_index(indices, shape)


def _stack(*cvs):
    """Stack sequences of collective variables along the last axis."""
    return [np.stack(vs, axis=-1) for vs in zip(*cvs)]
