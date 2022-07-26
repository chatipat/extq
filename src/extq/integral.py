import numba as nb
import numpy as np

from .moving_semigroup import moving_semigroup


@nb.njit
def integral_coeffs(u, kl, kr, lag):
    """Compute pointwise coefficients for integral-type statistics.

    Parameters
    ----------
    u : (n_frames - lag, n_left_indices, n_right_indices) ndarray
        Outer product of left statistic at time t and right statistic at
        time t + lag.
    kl : (n_frames - 1, n_left_indices, n_left_indices) ndarray
        Transition kernel for left statistic.
    kr : (n_frames - 1, n_right_indices, n_right_indices) ndarray
        Transition kernel for right statistic.
    lag : int
        Lag time.

    Returns
    -------
    coeffs : (n_frames - 1, n_left_indices, n_right_indices) ndarray
        Coefficients for each transition.

    """
    ns, nl, nr = u.shape
    nt = ns + lag - 1
    assert kl.shape == (nt, nl, nl)
    assert kr.shape == (nt, nr, nr)
    m = np.zeros((nt + lag - 1, nr + nl, nr + nl))
    for t in range(0, nt - 1):
        for i in range(nr):
            for j in range(nr):
                m[t, i, j] = kr[t + 1, i, j]
    for t in range(lag - 1, nt):
        for i in range(nr):
            for j in range(nl):
                m[t, i, nr + j] = u[t - lag + 1, j, i]
    for t in range(lag, nt + lag - 1):
        for i in range(nl):
            for j in range(nl):
                m[t, nr + i, nr + j] = kl[t - lag, i, j]
    m = moving_semigroup(m, lag, np.dot)
    assert m.shape == (nt, nr + nl, nr + nl)
    coeffs = np.empty((nt, nl, nr))
    for t in range(nt):
        for i in range(nl):
            for j in range(nr):
                coeffs[t, i, j] = m[t, j, nr + i] / lag
    return coeffs
