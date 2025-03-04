import numpy as np
import scipy as sp


def sum_windows(a, start, end):
    r"""
    Return the sum of the elements within each window.

    The output of this function is::

        out[i] = np.sum(a[start[i] : end[i]], axis=0)

    Parameters
    ----------
    a : (n, *shape) array_like
        Input array.
    start : (n_windows,) ndarray of int
        Starting index (inclusive) of each window.
    end : (n_windows,) ndarray of int
        Ending index (exclusive) of each window.

    Returns
    -------
    out : (n_windows, *shape) ndarray
        Sum of the elements within each window.

    """
    c = np.concatenate([
        np.zeros((1, *a.shape[1:]), a.dtype),
        np.cumsum(a, axis=0),
    ])
    return c[end] - c[start]


def count_transition_paths(in_domain, reactant, product):
    r"""
    Count the number of complete transition paths within the trajectory.

    Parameters
    ----------
    in_domain : (n,) ndarray of bool
        Whether each frame is in the domain.
    reactant : (n,) ndarray of {bool, int, float}
        Whether each frame is in the reactant.
    product : (n,) ndarray of {bool, int, float}
        Whether each frame is in the product.

    Returns
    -------
    int or float
        Number of complete transition paths.

    """
    assert in_domain.shape == reactant.shape == product.shape
    (t,) = np.nonzero(np.logical_not(in_domain))
    return np.sum(reactant[t[:-1]] * product[t[1:]])


def count_transition_paths_windows(in_domain, reactant, product, start, end):
    r"""
    Count the number of complete transition paths within each window.

    Parameters
    ----------
    in_domain : (n,) ndarray of bool
        Whether each frame is in the domain.
    reactant : (n,) ndarray of {bool, int, float}
        Whether each frame is in the reactant.
    product : (n,) ndarray of {bool, int, float}
        Whether each frame is in the product.
    start : (n_windows,) ndarray of int
        Starting index (inclusive) of each window.
    end : (n_windows,) ndarray of int
        Ending index (exclusive) of each window.

    Returns
    -------
    ndarray of {int, float}
        Number of complete transition paths within each window.

    """
    assert in_domain.shape == reactant.shape == product.shape
    assert start.shape == end.shape
    assert np.all(0 <= start) and np.all(start <= end) and np.all(end <= len(in_domain))

    # partition trajectory into segments
    # segment i starts at t[i] and ends at t[i+1] (inclusive)
    # last frame of segment i is first frame of segment i+1
    # t.shape == (n_segments+1,)
    (t,) = np.nonzero(np.logical_not(in_domain))
    t = np.concatenate([[-1], t, [len(in_domain)]])

    # w[i] is the probability that segment i is a transition path
    # w.shape == (n_segments,)
    w = np.concatenate([[0], reactant[t[1:-2]] * product[t[2:-1]], [0]])

    # find segment containing each start/end edge
    idx = np.repeat(np.arange(len(w)), np.diff(t))
    idx_start = idx[start]
    idx_end = idx[end]

    # sum segments between start[i] and end[i],
    # excluding the ones containing start[i] and end[i]
    # out[i] = sum(w[idx_start[i]+1:idx_end[i]])

    # c[i] = sum(w[:i])
    c = np.concatenate([[0], np.cumsum(w)])

    assert np.all(idx_end - idx_start >= 0)

    out = c[idx_end] - c[idx_start + 1]

    # correct for double subtraction
    # if start and end are both in the same segment,
    # this segment is excluded twice
    out[idx_end == idx_start] = 0

    return out


def weighted_lagged_covariance(x, y, w, lag_w):
    r"""
    Compute a weighted sum of time-lagged covariances.

    This function efficiently calculates
    ``sum(c * x[:-t].T * w[:-t] @ y[t:] for t, c in enumerate(lag_w))``
    in linearithmic time with respect to `n_frames` and `max_lag`.

    Parameters
    ----------
    x : (n_frames, n_left_features) ndarray of float
        First set of features for the time series. Each row is a time
        point and each column is a feature.
    y : (n_frames, n_right_features) ndarray of float
        Second set of features for the time series. Each row is a time
        point and each column is a features.
    w : (n_frames,) ndarray of float
        Weight of the length ``max_lag + 1`` window starting at the
        time point. Note that the weights of the last `max_lag` time
        points must be zero (because their windows extend beyond the
        end of the time series).
    lag_w : (max_lag + 1,) ndarray of float
        Weight of each lag time, starting at zero lag time.

    Returns
    -------
    (n_left_features, n_right_features) ndarray of float
        Weighted sum of time-lagged covariances.

    """
    maxlag = len(lag_w) - 1
    if maxlag == 0:
        return x.T * (lag_w * w) @ y
    assert np.all(w[-maxlag:] == 0)
    wt = np.concatenate([[0], np.cumsum(w[:-maxlag])])
    wy = np.concatenate([np.full(maxlag, wt[0]), wt[:-1]])
    wx = np.concatenate([wt[1:], np.full(maxlag, wt[-1])])
    xt = sp.signal.fftconvolve(x, lag_w[:, None], axes=0)[:-maxlag]
    yt = sp.signal.fftconvolve(y, lag_w[::-1, None], axes=0)[maxlag:]
    return x.T * wx @ yt - xt.T * wy @ y
