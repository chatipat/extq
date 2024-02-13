import numba as nb
import numpy as np


def forward_stop(in_domain):
    """
    Find the first exit time from the domain.

    Parameters
    ----------
    in_domain : (N,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (N,) ndarray of int
        First exit time from the domain for trajectories starting at
        each frame of the input trajectory. A first exit time not within
        the trajectory is indicated by len(d).

    """
    (t,) = np.nonzero(np.logical_not(in_domain))
    t = np.concatenate([[-1], t, [len(in_domain)]])
    return np.repeat(t[1:], np.diff(t))[:-1]


def backward_stop(in_domain):
    """
    Find the last entry time into the domain.

    Parameters
    ----------
    in_domain : (N,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (N,) ndarray of int
        Last entry time into the domain for trajectories starting at
        each frame of the input trajectory. A last entry time not within
        the trajectory is indicated by -1.

    """
    (t,) = np.nonzero(np.logical_not(in_domain))
    t = np.concatenate([[-1], t, [len(in_domain)]])
    return np.repeat(t[:-1], np.diff(t))[1:]


@nb.njit
def forward_stop_numba(in_domain):
    """
    Find the first exit time from the domain.

    This function is compiled with numba.

    Parameters
    ----------
    in_domain : (N,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (N,) ndarray of int
        First exit time from the domain for trajectories starting at
        each frame of the input trajectory. A first exit time not within
        the trajectory is indicated by len(in_domain).

    """
    n = len(in_domain)
    result = np.empty(n, dtype=np.int_)
    stop_time = n
    for t in range(n - 1, -1, -1):
        if not in_domain[t]:
            stop_time = t
        result[t] = stop_time
    return result


@nb.njit
def backward_stop_numba(in_domain):
    """
    Find the last entry time into the domain.

    This function is compiled with numba.

    Parameters
    ----------
    in_domain : (N,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (N,) ndarray of int
        Last entry time into the domain for trajectories starting at
        each frame of the input trajectory. A last entry time not within
        the trajectory is indicated by -1.

    """
    n = len(in_domain)
    result = np.empty(n, dtype=np.int_)
    stop_time = -1
    for t in range(n):
        if not in_domain[t]:
            stop_time = t
        result[t] = stop_time
    return result
