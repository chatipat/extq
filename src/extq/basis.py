import numpy as np
import scipy.linalg
import scipy.sparse


def whiten(trajs, weights=None, rtol=None):
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

    Returns
    -------
    list of (n_frames[i], n_whitened_features) ndarray
        Whitened features, with dependent components removed.

    """
    # center the features
    trajs = center(trajs)

    # compute the covariance matrix
    numer = 0.0
    denom = 0.0
    if weights is None:
        for x in trajs:
            numer += x.T @ x
            denom += len(x)
    else:
        for x, w in zip(trajs, weights):
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
    coeffs = evecs[:, :num] / np.sqrt(evals[:num])
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
