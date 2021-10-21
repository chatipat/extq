import numpy as np
import scipy.signal


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
        assert np.all(w[len(w) - maxlag :] == 0.0)
        new_w = np.roll(w, maxlag - lag)
        result.append(new_w)
    return result


def splatter_weights(weights, lag, maxlag):
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
        assert np.all(w[len(w) - maxlag :] == 0.0)
        window = np.full(nlags, 1 / nlags, dtype=w.dtype)
        temp = scipy.signal.convolve(w[: len(w) - maxlag], window)
        new_w = np.concatenate((temp, np.zeros(lag, dtype=w.dtype)))
        result.append(new_w)
    return result
