import numpy as np
import scipy.linalg
import scipy.sparse
from more_itertools import zip_equal

__all__ = [
    "whiten",
    "center",
    "add_constant_feature",
    "remove_constant_feature",
]


def whiten(trajs, weights=None, rtol=None, with_mean=True, with_std=True):
    """Whiten the data using PCA.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) array-like
        Input features.
    weights : list of (n_frames[i],) array-like, optional
        Weight of each frame of the trajectories.
        If None, assign uniform weights.
    rtol : float, optional
        Relative tolerance to cutoff near-zero eigenvalues.
    with_mean : bool, optional
        If True, remove the mean of each feature before decorrelating.
    with_std : bool, optional
        If True, set variances to 1 after decorrelating.

    Returns
    -------
    list of (n_frames[i], n_whitened_features) ndarray
        Whitened features, with dependent components removed.

    """
    # center the features
    if with_mean:
        trajs = center(trajs)

    # compute the covariance matrix
    numer = 0.0
    denom = 0.0
    if weights is None:
        for x in trajs:
            numer += x.T @ x
            denom += len(x)
    else:
        for x, w in zip_equal(trajs, weights):
            numer += x.T @ scipy.sparse.diags(w) @ x
            denom += np.sum(w)
    cov = numer / denom

    # solve the PCA eigenproblem
    evals, evecs = scipy.linalg.eigh(cov)

    # remove near-zero eigenvalues
    # rtol, tol calculation based on scipy.linalg.orth
    if rtol is None:
        rtol = np.finfo(evals.dtype).eps * len(evals)
    tol = np.max(evals) * rtol
    num = np.sum(evals > tol, dtype=int)
    coeffs = evecs[:, :num]
    if with_std:
        coeffs /= np.sqrt(evals[:num])
    return [x @ coeffs for x in trajs]


def center(trajs, weights=None):
    """Subtract the mean from each feature.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) array-like
        Input features.
    weights : list of (n_frames[i],) array-like, optional
        Weight of each frame of the trajectories.
        If None, assign uniform weights.

    Returns
    -------
    list of (n_frames[i], n_features) ndarray
        Centered features.

    """
    numer = 0.0
    denom = 0.0
    if weights is None:
        for x in trajs:
            numer += np.sum(x, axis=0)
            denom += len(x)
    else:
        for x, w in trajs:
            numer += np.dot(np.moveaxis(x, 0, -1), w)
            denom += np.sum(w)
    mean = numer / denom
    return [x - mean for x in trajs]


def add_constant_feature(trajs):
    """Append the constant feature to the data.

    Parameters
    ----------
    traj : list of (n_frames[i], n_features) array-like
        Input features.

    Returns
    -------
    list of (n_frames[i], n_features+1) ndarray
        Features with the constant feature appended.

    """
    return [
        np.concatenate((x, np.ones((len(x), 1), dtype=x.dtype)), axis=-1)
        for x in trajs
    ]


def remove_constant_feature(trajs, weights=None, tol=0.0):
    """Remove the constant feature from the span of the features.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) array-like
        Input features. This must be able to represent the constant
        feature.
    weights : list of (n_frames[i],) array-like, optional
        Weight of each frame of the trajectories.
        If None, assign uniform weights.
    tol : float, optional
        If the magnitude of the mean of a feature is less than this
        value, assume that it is zero.

    Returns
    -------
    list of (n_frames[i], n_features-1) ndarray
        Features with the constant feature removed.

    """
    numer = 0.0
    denom = 0.0
    if weights is None:
        for x in trajs:
            numer += x.sum(axis=0)
            denom += x.shape[0]
    else:
        for x, w in zip_equal(trajs, weights):
            numer += w @ x
            denom += np.sum(w)
    means = np.ravel(numer) / denom
    n_out = len(means) - 1

    # order the features by the magnitudes of the means, and remove the
    # mean of each feature by subtracting from it the scaled feature
    # with the next greater magnitude of the mean

    order = np.argsort(np.abs(means))
    means = means[order]
    factor = np.ones(n_out)
    mask = means[:-1] >= tol
    factor[mask] = -means[:-1][mask] / means[1:][mask]

    data = np.concatenate([np.ones(n_out), factor])
    row = np.concatenate([order[:-1], order[1:]])
    col = np.concatenate([np.arange(n_out), np.arange(n_out)])
    mat = scipy.sparse.csr_matrix((data, (row, col)), shape=(n_out + 1, n_out))
    return [x @ mat for x in trajs]
