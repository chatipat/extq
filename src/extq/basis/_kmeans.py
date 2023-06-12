"""Functions for constructing k-means bases."""

import numpy as np
import sklearn.cluster
from more_itertools import zip_equal

from ._labels import labels_to_basis
from ._voronoi import voronoi_labels

__all__ = [
    "kmeans_centers",
    "kmeans_labels",
    "kmeans1d_labels",
    "kmeans2d_labels",
    "kmeans3d_labels",
    "kmeans_basis",
    "kmeans1d_basis",
    "kmeans2d_basis",
    "kmeans3d_basis",
]


def kmeans_centers(cvs, num, in_domain=None, **kwargs):
    """
    Compute cluster centers using k-means.

    Parameters
    ----------
    cvs : sequence of (n_frames[i], n_cvs) ndarray of float
        Collective variables at each frame.
    num : int
        Maximum number of clusters.
    in_domain : sequence of (n_frames[i],) ndarray of bool, optional
        Whether each frame is in the domain. Clustering is performed
        only on frames within the domain.

    Returns
    -------
    (num,) ndarray of float
        Cluster centers.

    """
    if in_domain is None:
        data = np.concatenate(cvs)
    else:
        data = np.concatenate([v[d] for v, d in zip(cvs, in_domain)])
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num, **kwargs)
    kmeans.fit(data)
    return kmeans.cluster_centers_


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
    centers = kmeans_centers(cvs, num, **kwargs)
    labels = voronoi_labels(cvs, centers)
    return labels


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


def kmeans_basis(cvs, num, sparse=True, in_domain=None, **kwargs):
    """
    Construct a basis of indicator functions within a specified domain
    using k-means.

    Parameters
    ----------
    cvs : sequence of (n_frames[i], n_cvs) ndarray of float
        Collective variables at each frame.
    num : int
        Maximum number of clusters.
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
    centers = kmeans_centers(cvs, num, in_domain=in_domain, **kwargs)
    labels = voronoi_labels(cvs, centers)
    return labels_to_basis(labels, sparse=sparse, in_domain=in_domain)


def kmeans1d_basis(cv, num, sparse=True, in_domain=None, **kwargs):
    """
    Construct a basis of indicator functions within a specified domain
    in a 1D collective variable space using k-means.

    Parameters
    ----------
    cv : sequence of (n_frames[i],) ndarray of float
        Collective variable at each frame.
    num : int
        Maximum number of clusters.
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
    return kmeans_basis(
        _stack(cv), num, sparse=sparse, in_domain=in_domain, **kwargs
    )


def kmeans2d_basis(cv1, cv2, num, sparse=True, in_domain=None, **kwargs):
    """
    Construct a basis of indicator functions within a specified domain
    in a 2D collective variable space using k-means.

    Parameters
    ----------
    cv1, cv2 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    num : int
        Maximum number of clusters.
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
    return kmeans_basis(
        _stack(cv1, cv2), num, sparse=sparse, in_domain=in_domain, **kwargs
    )


def kmeans3d_basis(cv1, cv2, cv3, num, sparse=True, in_domain=None, **kwargs):
    """
    Construct a basis of indicator functions within a specified domain
    in a 3D collective variable space using k-means.

    Parameters
    ----------
    cv1, cv2, cv3 : sequence of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    num : int
        Maximum number of clusters.
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
    return kmeans_basis(
        _stack(cv1, cv2, cv3),
        num,
        sparse=sparse,
        in_domain=in_domain,
        **kwargs
    )


def _stack(*cvs):
    """Stack sequences of collective variables along the last axis."""
    return [np.stack(vs, axis=-1) for vs in zip_equal(*cvs)]
