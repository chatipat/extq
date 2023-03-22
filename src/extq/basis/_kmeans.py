"""Functions for constructing k-means bases."""

import numpy as np
import sklearn.cluster
from more_itertools import zip_equal

from ._labels import _labels_to_basis, renumber_labels

__all__ = [
    "kmeans_labels",
    "kmeans1d_labels",
    "kmeans2d_labels",
    "kmeans3d_labels",
    "kmeans_basis",
    "kmeans1d_basis",
    "kmeans2d_basis",
    "kmeans3d_basis",
    "kmeans_domain_basis",
    "kmeans1d_domain_basis",
    "kmeans2d_domain_basis",
    "kmeans3d_domain_basis",
]


def kmeans_labels(cvs, num, **kwargs):
    """
    Cluster frames on a collective variable space using k-means.

    Parameters
    ----------
    cvs : sequence of (n_frames[i], n_cvs) ndarray of float
        Collective variables at each frame.
    num : int
        Maximum number of clusters.

    Returns
    -------
    list of (n_frames[i],) ndarray of int
        Cluster index at each frame.

    """
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num, **kwargs)
    kmeans.fit(np.concatenate(cvs))
    indices = np.cumsum([len(v) for v in cvs])[:-1]
    return renumber_labels(np.split(kmeans.labels_, indices))


def kmeans1d_labels(cv, num, **kwargs):
    """
    Cluster frames on a 1D collective variable space using k-means.

    Parameters
    ----------
    cv : sequence of (n_frames[i],) ndarray of float
        Collective variable at each frame.
    num : int
        Maximum number of clusters.

    Returns
    -------
    list of (n_frames[i],) ndarray of int
        Cluster index at each frame.

    """
    return kmeans_labels(_stack(cv), num, **kwargs)


def kmeans2d_labels(cv1, cv2, num, **kwargs):
    """
    Cluster frames on a 2D collective variable space using k-means.

    Parameters
    ----------
    cv1, cv2 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    num : int
        Maximum number of clusters.

    Returns
    -------
    list of (n_frames[i],) ndarray of int
        Cluster index at each frame.

    """
    return kmeans_labels(_stack(cv1, cv2), num, **kwargs)


def kmeans3d_labels(cv1, cv2, cv3, num, **kwargs):
    """
    Cluster frames on a 3D collective variable space using k-means.

    Parameters
    ----------
    cv1, cv2, cv3 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    num : int
        Maximum number of clusters.

    Returns
    -------
    list of (n_frames[i],) ndarray of int
        Cluster index at each frame.

    """
    return kmeans_labels(_stack(cv1, cv2, cv3), num, **kwargs)


def kmeans_basis(cvs, num, sparse=True, **kwargs):
    """
    Construct a basis of indicator functions using k-means.

    Parameters
    ----------
    cvs : sequence of (n_frames[i], n_cvs) ndarray of float
        Collective variables at each frame.
    num : int
        Maximum number of clusters.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    labels = kmeans_labels(cvs, num, **kwargs)
    num = _num(labels)
    basis = []
    for indices in labels:
        basis.append(_labels_to_basis(indices, num, sparse=sparse))
    return basis


def kmeans1d_basis(cv, num, sparse=True, **kwargs):
    """
    Construct a basis of indicator functions on a 1D collective variable
    space using k-means.

    Parameters
    ----------
    cv : sequence of (n_frames[i],) ndarray of float
        Collective variable at each frame.
    num : int
        Maximum number of clusters.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    return kmeans_basis(_stack(cv), num, sparse=sparse, **kwargs)


def kmeans2d_basis(cv1, cv2, num, sparse=True, **kwargs):
    """
    Construct a basis of indicator functions on a 2D collective variable
    space using k-means.

    Parameters
    ----------
    cv1, cv2 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    num : int
        Maximum number of clusters.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    return kmeans_basis(_stack(cv1, cv2), num, sparse=sparse, **kwargs)


def kmeans3d_basis(cv1, cv2, cv3, num, sparse=True, **kwargs):
    """
    Construct a basis of indicator functions on a 3D collective variable
    space using k-means.

    Parameters
    ----------
    cv1, cv2, cv3 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    num : int
        Maximum number of clusters.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    return kmeans_basis(_stack(cv1, cv2, cv3), num, sparse=sparse, **kwargs)


def kmeans_domain_basis(cvs, in_domain, num, sparse=True, **kwargs):
    """
    Construct a basis of indicator functions within a specified domain
    using k-means.

    Parameters
    ----------
    cvs : sequence of (n_frames[i], n_cvs) ndarray of float
        Collective variables at each frame.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain. The basis is zero outside
        of the domain.
    num : int
        Maximum number of clusters.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    cvs_d = [v[d] for v, d in zip_equal(cvs, in_domain)]
    labels = kmeans_labels(cvs_d, num, **kwargs)
    num = _num(labels)
    basis = []
    for indices_d, d in zip_equal(labels, in_domain):
        indices = np.empty(len(d), dtype=indices_d.dtype)
        indices[d] = indices_d
        basis.append(_labels_to_basis(indices, num, sparse=sparse, mask=d))
    return basis


def kmeans1d_domain_basis(cv, in_domain, num, sparse=True, **kwargs):
    """
    Construct a basis of indicator functions within a specified domain
    in a 1D collective variable space using k-means.

    Parameters
    ----------
    cv : sequence of (n_frames[i],) ndarray of float
        Collective variable at each frame.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain. The basis is zero outside
        of the domain.
    num : int
        Maximum number of clusters.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    return kmeans_domain_basis(
        _stack(cv), in_domain, num, sparse=sparse, **kwargs
    )


def kmeans2d_domain_basis(cv1, cv2, in_domain, num, sparse=True, **kwargs):
    """
    Construct a basis of indicator functions within a specified domain
    in a 2D collective variable space using k-means.

    Parameters
    ----------
    cv1, cv2 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain. The basis is zero outside
        of the domain.
    num : int
        Maximum number of clusters.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    return kmeans_domain_basis(
        _stack(cv1, cv2), in_domain, num, sparse=sparse, **kwargs
    )


def kmeans3d_domain_basis(
    cv1, cv2, cv3, in_domain, num, sparse=True, **kwargs
):
    """
    Construct a basis of indicator functions within a specified domain
    in a 3D collective variable space using k-means.

    Parameters
    ----------
    cv1, cv2, cv3 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain. The basis is zero outside
        of the domain.
    num : int
        Maximum number of clusters.
    sparse : bool, optional
        If True (default), return a list of sparse matrices instead of
        a list of dense arrays.

    Returns
    -------
    list of {ndarray, sparse matrix} of float
        Basis of indicator functions.

    """
    return kmeans_domain_basis(
        _stack(cv1, cv2, cv3), in_domain, num, sparse=sparse, **kwargs
    )


def _num(labels):
    """Find the number of clusters given labeled trajectories."""
    return max(np.max(indices) for indices in labels if len(indices) > 0) + 1


def _stack(*cvs):
    """Stack sequences of collective variables along the last axis."""
    return [np.stack(vs, axis=-1) for vs in zip_equal(*cvs)]
