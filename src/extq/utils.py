from itertools import pairwise

import numpy as np
import scipy.signal

__all__ = [
    "uniform_weights",
    "shift_weights",
    "distribute_weights",
    "normalize_weights",
    "lengths",
    "flatten",
    "unflatten",
]


def uniform_weights(trajs, maxlag, *, normalize=True):
    """Make uniform weights.

    Parameters
    ----------
    trajs : list of (n_frames[i], ...) array-like
        List of trajectories.
    maxlag : int
        Number of frames at the end of each trajectory that are required
        to have zero weight. This is the maximum lag time the output
        weights can be used with by other methods.
    normalize : bool, optional
        If True (default), normalize the weights so that they sum to
        one. If False, the nonzero weights are one.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Weight at each frame of the trajectory. The weights of the first
        n_frames[i]-maxlag trajectories are constant.

    """
    assert maxlag >= 0
    weights = []
    for n_frames in lengths(trajs):
        w = np.ones(n_frames)
        w[max(0, n_frames - maxlag) :] = 0.0
        weights.append(w)
    if normalize:
        weights = normalize_weights(weights)
    return weights


def shift_weights(weights, lag, maxlag):
    """Convert weights for length maxlag short trajectories to
    right-aligned length lag short trajectories.

    Parameters
    ----------
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
        The last maxlag frames must be zero.
    lag : int
        New length of short trajectories, in units of frames.
    maxlag : int
        Original length of short trajectories, in units of frames.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        New change of measure to the invariant distribution for each
        frame. The last lag frames are zero.

    """
    assert 0 <= lag <= maxlag
    result = []
    for w in weights:
        (n_frames,) = w.shape
        assert np.all(w[max(0, n_frames - maxlag) :] == 0.0)
        new_w = np.roll(w, maxlag - lag)
        result.append(new_w)
    return result


def distribute_weights(weights, lag, maxlag):
    """Uniformly distribute weights for length maxlag short trajectories
    to length lag short trajectories.

    Parameters
    ----------
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
        The last maxlag frames must be zero.
    lag : int
        New length of short trajectories, in units of frames.
    maxlag : int
        Original length of short trajectories, in units of frames.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        New change of measure to the invariant distribution for each
        frame. The last lag frames are zero.

    """
    assert 0 <= lag <= maxlag
    nlags = maxlag - lag + 1
    result = []
    for w in weights:
        (n_frames,) = w.shape
        assert np.all(w[np.max(0, n_frames - maxlag) :] == 0.0)
        new_w = np.zeros(n_frames, dtype=w.dtype)
        if n_frames > maxlag:
            window = np.full(nlags, 1 / nlags, dtype=w.dtype)
            temp = scipy.signal.convolve(w[: n_frames - maxlag], window)
            new_w[: n_frames - lag] = temp
        result.append(new_w)
    return result


def normalize_weights(weights):
    """
    Normalize input weights to sum to one.

    Parameters
    ----------
    weights : sequence of (n_frames[i],) ndarray of float
        Input weights.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Normalized weights.

    """
    scale = 1.0 / np.sum([np.sum(w) for w in weights])
    return [scale * w for w in weights]


def lengths(trajs):
    """
    Return the length of each trajectory.

    Parameters
    ----------
    trajs : list of (n_frames[i], ...) array-like
        List of trajectories.

    Returns
    -------
    lengths : (n_trajs,) ndarray of int
        Length of each trajectory.

    """
    return np.array([np.shape(traj)[0] for traj in trajs])


def flatten(trajs):
    """
    Concatenate trajectories.

    Parameters
    ----------
    trajs : list of (n_frames[i], *data_shape) {ndarray, sparse matrix}
        List of trajectories.

    Returns
    -------
    flat : (sum(n_frames), *data_shape) {ndarray, sparse matrix}
        Concatenated trajectory.

    """
    if any(scipy.sparse.issparse(traj) for traj in trajs):
        return scipy.sparse.vstack(trajs)
    else:
        return np.concatenate(trajs)


def unflatten(flat, lengths):
    """
    Split a trajectory given resulting lengths.

    Parameters
    ----------
    flat : (sum(n_frames), *data_shape) {ndarray, sparse matrix}
        Concatenated trajectory.

    Returns
    -------
    trajs : list of (n_frames[i], *data_shape) {ndarray, sparse matrix}
        List of trajectories.

    """
    assert flat.shape[0] == np.sum(lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    return [flat[start:end] for start, end in pairwise(offsets)]
