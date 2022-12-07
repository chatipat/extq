import numpy as np

from .moving_semigroup import moving_matmul


def integral_coeffs(u, kl, kr, obslag, lag, normalize=False):
    """
    Compute pointwise coefficients for integral-type statistics.

    Parameters
    ----------
    u : (n_frames - lag, n_left_indices, n_right_indices) ndarray of float
        Outer product of left statistic at time t and right statistic at
        time t + lag.
    kl : (n_frames - 1, n_left_indices, n_left_indices) ndarray of float
        Transition kernel for left statistic.
    kr : (n_frames - 1, n_right_indices, n_right_indices) ndarray of float
        Transition kernel for right statistic.
    obslag : int
        Lag time of the observable.
    lag : int
        Total lag time.
    normalize : bool, optional
        If False (default), return the integral of the observable.
        If True, return the mean of the observable. This is the integral
        divided by `lag - obslag + 1`.

    Returns
    -------
    (n_frames - obslag, n_left_indices, n_right_indices) ndarray of float
        Coefficients for each observation.

    """
    ns, nl, nr = u.shape  # ns = number of windows (n_frames - lag)
    nf = ns + lag  # number of frames (n_frames)
    nt = nf - 1  # number of steps (n_frames - 1)
    nint = lag - obslag + 1  # number of times over which to integrate
    assert 0 <= obslag <= lag < nf
    assert kl.shape == (nt, nl, nl)
    assert kr.shape == (nt, nr, nr)
    if lag == obslag:
        return u
    m = np.zeros((nt + lag - 1, nr + nl, nr + nl))
    m[: nt - 1, :nr, :nr] = kr[1:]
    m[lag - 1 : nt, :nr, nr:] = np.swapaxes(u, 1, 2)
    m[lag:, nr:, nr:] = kl[:-1]
    m = moving_matmul(m, nint)
    assert m.shape == (nf - obslag, nr + nl, nr + nl)
    out = np.swapaxes(m[:, :nr, nr:], 1, 2)
    if normalize:
        out /= nint
    return out


def integral_windows(kl, kr, obs, obslag, lag, normalize=False):
    """
    Compute integral-type statistics over each window.

    Parameters
    ----------
    kl : (n_frames - 1, n_left_indices, n_left_indices) ndarray of float
        Transition kernel for left statistic.
    kr : (n_frames - 1, n_right_indices, n_right_indices) ndarray of float
        Transition kernel for right statistic.
    obs : (n_frames - obslag, n_left_indices, n_right_indices) ndarray of float
        Transition kernel for the observable.
    obslag : int
        Lag time of the observable.
    lag : int
        Total lag time.
    normalize : bool, optional
        If False (default), return the integral of the observable.
        If True, return the mean of the observable. This is the integral
        divided by `lag - obslag + 1`.

    Returns
    -------
    (n_frames - lag, n_left_indices, n_right_indices) ndarray of float
        Integral (or mean) of the observable over each window.

    """
    nobs, nl, nr = obs.shape  # nobs = number of obs (n_frames - obslag)
    nf = nobs + obslag  # number of frames (n_frames)
    nt = nf - 1  # number of steps (n_frames - 1)
    nint = lag - obslag + 1  # number of times over which to integrate
    assert 0 <= obslag <= lag < nf
    assert kl.shape == (nt, nl, nl)
    assert kr.shape == (nt, nr, nr)
    if lag == obslag:
        return obs
    m = np.zeros((nobs, nl + nr, nl + nr))
    m[:-1, :nl, :nl] = kl[: nt - obslag]
    m[:, :nl, nl:] = obs
    m[1:, nl:, nl:] = kr[obslag:]
    m = moving_matmul(m, nint)
    assert m.shape == (nf - lag, nl + nr, nl + nr)
    out = m[:, :nl, nl:]
    if normalize:
        out /= nint
    return out
