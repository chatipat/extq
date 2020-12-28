import numba as nb
import numpy as np


@nb.njit
def forward_stop(in_domain):
    """Find the first exit time from the domain.

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
    result = np.empty(n, dtype=np.int32)
    stop_time = n
    for t in range(n - 1, -1, -1):
        if not in_domain[t]:
            stop_time = t
        result[t] = stop_time
    return result


@nb.njit
def backward_stop(in_domain):
    """Find the last entry time into the domain.

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
    result = np.empty(n, dtype=np.int32)
    stop_time = -1
    for t in range(n):
        if not in_domain[t]:
            stop_time = t
        result[t] = stop_time
    return result
