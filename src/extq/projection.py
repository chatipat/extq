import numpy as np


def density1d(cv, weights, edges):
    """Estimate a density on a collective variable.

    Parameters
    ----------
    cv : list of (n_frames[i],) ndarray of float
        Collective variable at each frame.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    edges : 1d array_like of float
        Bin edges for the collective variable.

    Returns
    -------
    1d ndarray of float
        Histogram of the weights on the collective variable. Note that
        the value of each bin is normalized by the width of the bin.

    """
    numer = 0.0
    for x, w in zip(cv, weights):
        numer += np.histogram(x, bins=edges, weights=w)[0]
    denom = np.diff(edges)
    return numer / denom


def density2d(cv1, cv2, weights, edges1, edges2):
    """Estimate a density on a 2D collective variable space.

    Parameters
    ----------
    cv1, cv2 : list of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    edges1, edges2 : 1d array_like of float
        Bin edges for each collective variable.

    Returns
    -------
    2d ndarray of float
        Histogram of the weights on the collective variable space. Note
        that the value of each bin is normalized by the area of the bin.

    """
    numer = 0.0
    for x, y, w in zip(cv1, cv2, weights):
        numer += np.histogram2d(x, y, bins=(edges1, edges2), weights=w)[0]
    denom = np.einsum("i,j->ij", np.diff(edges1), np.diff(edges2))
    return numer / denom


def density3d(cv1, cv2, cv3, weights, edges1, edges2, edges3):
    """Estimate a density on a 3D collective variable space.

    Parameters
    ----------
    cv1, cv2, cv3 : list of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    edges1, edges2, edges3 : 1d array_like of float
        Bin edges for each collective variable.

    Returns
    -------
    3d ndarray of float
        Histogram of the weights on the collective variable space. Note
        that the value of each bin is normalized by the volume of the
        bin.

    """
    numer = 0.0
    for x, y, z, w in zip(cv1, cv2, cv3, weights):
        numer += np.histogramdd(
            (x, y, z), bins=(edges1, edges2, edges3), weights=w
        )[0]
    denom = np.einsum(
        "i,j,k->ijk", np.diff(edges1), np.diff(edges2), np.diff(edges3)
    )
    return numer / denom


def average1d(cv, func, weights, edges):
    """Estimate an expectation on a collective variable.

    Parameters
    ----------
    cv : list of (n_frames[i],) ndarray of float
        Collective variable at each frame.
    func : list of (n_frames[i],) ndarray of float
        Random variable of which to estimate the expectation.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    edges : 1d array_like of float
        Bin edges for the collective variable.

    Returns
    -------
    1d ndarray of float
        Expectation on the collective variable.

    """
    numer = 0.0
    denom = 0.0
    for x, f, w in zip(cv, func, weights):
        numer += np.histogram(x, bins=edges, weights=f * w)[0]
        denom += np.histogram(x, bins=edges, weights=w)[0]
    return numer / denom


def average2d(cv1, cv2, func, weights, edges1, edges2):
    """Estimate an expectation on a 2D collective variable space.

    Parameters
    ----------
    cv1, cv2 : list of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    func : list of (n_frames[i],) ndarray of float
        Random variable of which to estimate the expectation.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    edges1, edges2 : 1d array_like of float
        Bin edges for each collective variable.

    Returns
    -------
    2d ndarray of float
        Expectation on the collective variable space.

    """
    numer = 0.0
    denom = 0.0
    for x, y, f, w in zip(cv1, cv2, func, weights):
        numer += np.histogram2d(x, y, bins=(edges1, edges2), weights=f * w)[0]
        denom += np.histogram2d(x, y, bins=(edges1, edges2), weights=w)[0]
    return numer / denom


def average3d(cv1, cv2, cv3, func, weights, edges1, edges2, edges3):
    """Estimate an expectation on a 3D collective variable space.

    Parameters
    ----------
    cv1, cv2, cv3 : list of (n_frames[i],) ndarray of float
        Collective variables at each frame.
    func : list of (n_frames[i],) ndarray of float
        Random variable of which to estimate the expectation.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    edges1, edges2, edges3 : 1d array_like of float
        Bin edges for each collective variable.

    Returns
    -------
    3d ndarray of float
        Expectation on the collective variable space.

    """
    numer = 0.0
    denom = 0.0
    for x, y, z, f, w in zip(cv1, cv2, cv3, func, weights):
        numer += np.histogramdd(
            (x, y, z), bins=(edges1, edges2, edges3), weights=f * w
        )[0]
        denom += np.histogramdd(
            (x, y, z), bins=(edges1, edges2, edges3), weights=w
        )[0]
    return numer / denom
