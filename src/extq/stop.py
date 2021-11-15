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


@nb.njit
def forward_integrate(in_domain, function):
    """Integrate a function to the first exit time from the domain.

    Parameters
    ----------
    in_domain : (N,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.
    function : (N,) ndarray of float
        Value of the integrand at each frame.

    Returns
    -------
    (N,) ndarray of float
        Integral of the function from the current time to the first exit
        time from the domain.

    """
    n = len(in_domain)
    result = np.empty(n)
    integral = 0.0
    for t in range(n - 1, -1, -1):
        if in_domain[t]:
            integral += function[t]
        else:
            integral = 0.0
        result[t] = integral
    return result


@nb.njit
def backward_integrate(in_domain, function):
    """Integrate a function from the last entry time into the domain.

    Parameters
    ----------
    in_domain : (N,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.
    function : (N,) ndarray of float
        Value of the integrand at each frame.

    Returns
    -------
    (N,) ndarray of float
        Integrate the function from the last entry time into the domain
        to the current time.

    """
    n = len(in_domain)
    result = np.empty(n)
    integral = 0.0
    for t in range(n):
        if in_domain[t]:
            integral += function[t]
        else:
            integral = 0.0
        result[t] = integral
    return result
