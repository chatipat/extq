"""Kernels for calculating correlation matrices."""

import numba as nb
import numpy as np

from ..moving_semigroup import mm2
from ..moving_semigroup import mm3
from ..moving_semigroup import mm4
from ..moving_semigroup import moving_semigroup


@nb.njit
def reweight_kernel(w, lag):
    """
    Correlation matrix kernel for reweighting.

    Parameters
    ----------
    w : (n_frames,) ndarray of float
        Weight of each frame. Note that the last `lag` frames must have
        zero weight.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    (1, 1, n_frames-lag) ndarray of float
        Correlation matrix kernel for computing ``reweight_matrix``.

    """
    _check_lag(w, lag)
    k = np.ones((len(w) - lag, 1, 1))
    return _kernel(k, w, lag)


@nb.njit
def forward_kernel(w, d_f, f_f, g_f, lag):
    """
    Correlation matrix kernel for forecasting.

    Parameters
    ----------
    w : (n_frames,) ndarray of float
        Weight of each frame. Note that the last `lag` frames must have
        zero weight.
    d_f : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    f_f : (n_frames-1,) ndarray of float
        Function to integrate. Note that this is defined at each *step*,
        not at each frame.
    g_f : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    (2, 2, n_frames-lag) ndarray of float
        Correlation matrix kernel for computing forecast correlation
        matrices.

    """
    _check_lag(w, lag)
    if lag == 0:
        k = forward_identity(d_f, g_f)
    else:
        k = forward_transitions(d_f, f_f, g_f)
        k = moving_semigroup(k, lag, mm2)
    forward_guess(k, d_f, g_f, lag)
    return _kernel(k, w, lag)


@nb.njit
def backward_kernel(w, d_b, f_b, g_b, lag):
    """
    Correlation matrix kernel for aftcasting.

    Parameters
    ----------
    w : (n_frames,) ndarray of float
        Weight of each frame. Note that the last `lag` frames must have
        zero weight.
    d_b : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    f_b : (n_frames-1,) ndarray of float
        Function to integrate. Note that this is defined at each *step*,
        not at each frame.
    g_b : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    (2, 2, n_frames-lag) ndarray of float
        Correlation matrix kernel for computing aftcast correlation
        matrices.

    """
    _check_lag(w, lag)
    if lag == 0:
        k = backward_identity(d_b, g_b)
    else:
        k = backward_transitions(d_b, f_b, g_b)
        k = moving_semigroup(k, lag, mm2)
    backward_guess(k, d_b, g_b, lag)
    return _kernel(k, w, lag)


@nb.njit
def reweight_integral_kernel(w, v, lag):
    r"""
    Correlation matrix kernel for computing ergodic averages of
    single-step observables.

    The average computed is :math:`\langle v(X_t,X_{t+1}) \rangle`.

    Parameters
    ----------
    w : (n_frames,) ndarray of float
        Weight of each frame. Note that the last `lag` frames must have
        zero weight.
    v : (n_frames-1,) ndarray of float
        Observable for which to compute the ergodic average. Note that
        this is defined at each *step*, not at each frame, so ``v[t]``
        is a function of frames ``t`` and ``t+1``.
        Weight of each frame.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    (1, 1, n_frames-lag) ndarray of float
        Correlation matrix for computing ``reweight_integral_matrix``.

    """
    _check_lag(w, lag)
    if lag == 0:
        k = np.zeros((len(w), 1, 1))
    else:
        k = np.zeros((len(w) - 1, 2, 2))
        k[:, 0, 0] = 1.0
        k[:, 0, 1] = v
        k[:, 1, 1] = 1.0
        k = moving_semigroup(k, lag, mm2)[:1, 1:]
    return _kernel(k, w, lag)


@nb.njit
def forward_integral_kernel(w, d_f, v, f_f, g_f, lag):
    r"""
    Correlation matrix kernel for computing ergodic averages involving
    forecasts.

    For the forecast :math:`u_+`, the average computed is
    :math:`\langle v(X_t,X_{t+1}) u_+(X_{t+1}) \rangle`.

    Parameters
    ----------
    w : (n_frames,) ndarray of float
        Weight of each frame. Note that the last `lag` frames must have
        zero weight.
    d_f : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    v : (n_frames-1,) ndarray of float
        Observable for which to compute the ergodic average. Note that
        this is defined at each *step*, not at each frame, so ``v[t]``
        is a function of frames ``t`` and ``t+1``.
    f_f : (n_frames-1,) ndarray of float
        Function to integrate. Note that this is defined at each *step*,
        not at each frame.
    g_f : (n_frames,) ndarray of float
        Guess of the forecast, with the correct boundary conditions.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    (1, 2, n_frames-lag) ndarray of float
        Correlation matrix for computing correlation matrices for
        ergodic averages involving forecasts.

    """
    _check_lag(w, lag)
    if lag == 0:
        k = np.zeros((len(w), 1, 2))
    else:
        k = np.zeros((len(w) - 1, 3, 3))
        k[:, 0, 0] = 1.0
        observable(k[:, :1, 1:], v)
        forward_transitions(d_f, f_f, g_f, out=k[:, 1:, 1:])
        k = moving_semigroup(k, lag, mm3)[:1, 1:]
        forward_guess(k, d_f, g_f, lag)
    return _kernel(k, w, lag)


@nb.njit
def backward_integral_kernel(w, d_b, v, f_b, g_b, lag):
    r"""
    Correlation matrix kernel for computing ergodic averages involving
    aftcasts.

    For the aftcast :math:`u_-`, the average computed is
    :math:`\langle u_-(X_t) v(X_t,X_{t+1}) \rangle`.

    Parameters
    ----------
    w : (n_frames,) ndarray of float
        Weight of each frame. Note that the last `lag` frames must have
        zero weight.
    d_b : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    v : (n_frames-1,) ndarray of float
        Observable for which to compute the ergodic average. Note that
        this is defined at each *step*, not at each frame, so ``v[t]``
        is a function of frames ``t`` and ``t+1``.
    f_b : (n_frames-1,) ndarray of float
        Function to integrate. Note that this is defined at each *step*,
        not at each frame.
    g_b : (n_frames,) ndarray of float
        Guess of the aftcast, with the correct boundary conditions.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    (2, 1, n_frames-lag) ndarray of float
        Correlation matrix for computing correlation matrices for
        ergodic averages involving aftcasts.

    """
    _check_lag(w, lag)
    if lag == 0:
        k = np.zeros((len(w), 2, 1))
    else:
        k = np.zeros((len(w) - 1, 3, 3))
        backward_transitions(d_b, f_b, g_b, out=k[:, :2, :2])
        observable(k[:, :2, 2:], v)
        k[:, 2, 2] = 1.0
        k = moving_semigroup(k, lag, mm3)[:2, 2:]
        backward_guess(k, d_b, g_b, lag)
    return _kernel(k, w, lag)


@nb.njit
def integral_kernel(w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag):
    r"""
    Correlation matrix kernel for computing ergodic averages involving
    both forecasts and aftcasts.

    For the forecast :math:`u_+` and the aftcast :math:`u_-`, the
    average computed is :math:`\langle u_-(X_t) v(X_t,X_{t+1}) \rangle`.

    Parameters
    ----------
    w : (n_frames,) ndarray of float
        Weight of each frame. Note that the last `lag` frames must have
        zero weight.
    d_b : (n_frames,) ndarray of bool
        For the aftcast, whether each frame is in the domain.
    d_f : (n_frames,) ndarray of bool
        For the forecast, whether each frame is in the domain.
    v : (n_frames-1,) ndarray of float
        Observable for which to compute the ergodic average. Note that
        this is defined at each *step*, not at each frame, so ``v[t]``
        is a function of frames ``t`` and ``t+1``.
    f_b : (n_frames-1,) ndarray of float
        Function to integrate for the aftcast. Note that this is
        defined at each *step*, not at each frame.
    f_f : (n_frames-1,) ndarray of float
        Function to integrate for the forecast. Note that this is
        defined at each *step*, not at each frame.
    g_b : (n_frames,) ndarray of float
        Guess of the aftcast, with the correct boundary conditions.
    g_f : (n_frames,) ndarray of float
        Guess of the forecast, with the correct boundary conditions.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    (2, 2, n_frames-lag) ndarray of float
        Correlation matrix for computing correlation matrices for
        ergodic averages involving forecasts and aftcasts.

    """
    _check_lag(w, lag)
    if lag == 0:
        k = np.zeros((len(w), 2, 2))
    else:
        k = np.zeros((len(w) - 1, 4, 4))
        backward_transitions(d_b, f_b, g_b, out=k[:, :2, :2])
        observable(k[:, :2, 2:], v)
        forward_transitions(d_f, f_f, g_f, out=k[:, 2:, 2:])
        k = moving_semigroup(k, lag, mm4)[:2, 2:]
        backward_guess(k, d_b, g_b, lag)
        forward_guess(k, d_f, g_f, lag)
    return _kernel(k, w, lag)


@nb.njit
def forward_identity(d_f, g_f, out=None):
    r"""
    Identity element of the forecast kernel semigroup.

    For frame `t`, this is ::

        k[t] = [[ d_f[t], b_f[t] ],
                [      0,      1 ]]

    where ``b_f = ~d_f * g_f``.

    Note that this is a projection matrix because the forecast is
    restricted to an affine subspace.

    Parameters
    ----------
    d_f : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    g_f : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.
    out : (n_frames, 2, 2) ndarray of float, optional
        Array to which the output is written.

    Returns
    -------
    (n_frames, 2, 2) ndarray of float
        Identity element of the forecast kernel semigroup at each frame.

    """
    n = len(d_f)
    assert d_f.shape == (n,)
    out = _zeros((n, 2, 2), out=out)
    for t in range(n):
        if d_f[t]:
            out[t, 0, 0] = 1.0
        else:
            out[t, 0, 1] = g_f[t]
        out[t, 1, 1] = 1.0
    return out


@nb.njit
def forward_transitions(d_f, f_f, g_f, out=None):
    r"""
    Single-step transition, from the forecast kernel semigroup.

    For step `t`, this is ::

        k[t] = [[ d_f[t]*d_f[t+1], b_f[t] + d_f[t]*(b_f[t+1]+f_f[t]) ],
                [               0,                                 1 ]]

    where ``b_f = ~d_f * g_f``.

    Parameters
    ----------
    d_f : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    f_f : (n_frames-1,) ndarray of float
        Function to integrate. Note that this is defined at each *step*,
        not at each frame.
    g_f : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.
    out : (n_frames-1, 2, 2) ndarray of float, optional
        Array to which the output is written.

    Returns
    -------
    (n_frames-1, 2, 2) ndarray of float
        At each step ``t``, the kernel corresponding to the transition
        from frame ``t`` to frame ``t+1``.

    """
    n = len(d_f)
    assert d_f.shape == (n,)
    assert f_f.shape == (n - 1,)
    assert g_f.shape == (n,)
    out = _zeros((n - 1, 2, 2), out=out)
    for t in range(n - 1):
        if d_f[t]:
            if d_f[t + 1]:
                out[t, 0, 0] = 1.0
                out[t, 0, 1] = f_f[t]
            else:
                out[t, 0, 1] = g_f[t + 1] + f_f[t]
        else:
            out[t, 0, 1] = g_f[t]
        out[t, 1, 1] = 1.0
    return out


@nb.njit
def backward_identity(d_b, g_b, out=None):
    r"""
    Identity element of the aftcast kernel semigroup.

    For frame `t`, this is ::

        k[t] = [[ 1, b_b[t] ],
                [ 0, d_b[t] ]]

    where ``b_b = ~d_b * g_b``.

    Note that this is a projection matrix because the aftcast is
    restricted to an affine subspace.

    Parameters
    ----------
    d_b : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    g_b : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.
    out : (n_frames, 2, 2) ndarray of float, optional
        Array to which the output is written.

    Returns
    -------
    (n_frames, 2, 2) ndarray of float
        Identity element of the aftcast kernel semigroup at each frame.

    """
    n = len(d_b)
    assert d_b.shape == (n,)
    out = _zeros((n, 2, 2), out=out)
    for t in range(n):
        out[t, 0, 0] = 1.0
        if d_b[t]:
            out[t, 1, 1] = 1.0
        else:
            out[t, 0, 1] = g_b[t]
    return out


@nb.njit
def backward_transitions(d_b, f_b, g_b, out=None):
    r"""
    Single-step transition, from the aftcast kernel semigroup.

    For step `t`, this is ::

        k[t] = [[ 1, (b_b[t]+f_b[t+1])*d_b[t+1] + b_b[t+1] ],
                [ 0,                       d_b[t]*d_b[t+1] ]]

    where ``b_b = ~d_b * g_b``.

    Parameters
    ----------
    d_b : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    f_b : (n_frames-1,) ndarray of float
        Function to integrate. Note that this is defined at each *step*,
        not at each frame.
    g_b : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.
    out : (n_frames-1, 2, 2) ndarray of float, optional
        Array to which the output is written.

    Returns
    -------
    (n_frames-1, 2, 2) ndarray of float
        At each step ``t``, the kernel corresponding to the transition
        from frame ``t`` to frame ``t+1``.

    """
    n = len(d_b)
    assert d_b.shape == (n,)
    assert f_b.shape == (n - 1,)
    assert g_b.shape == (n,)
    out = _zeros((n - 1, 2, 2), out=out)
    for t in range(n - 1):
        out[t, 0, 0] = 1.0
        if d_b[t + 1]:
            if d_b[t]:
                out[t, 0, 1] = f_b[t]
                out[t, 1, 1] = 1.0
            else:
                out[t, 0, 1] = g_b[t] + f_b[t]
        else:
            out[t, 0, 1] = g_b[t + 1]
    return out


@nb.njit
def observable(k, v):
    """
    Write a single-step observable to a kernel.

    Note that the appropriate boundary conditions must be applied after
    this function (e.g., using ``forward_boundary`` and
    ``backward_boundary``).

    Parameters
    ----------
    k : (n_frames-1, n_left_indices, n_right_indices) ndarray of float
        Input kernel.
    v : (n_frames-1,) ndarray of float
        Value of the observable at each step. ``v[t]`` is calculated
        from frames ``t`` and ``t+1``.

    """
    k[:, -1, 0] = v


@nb.njit
def forward_guess(k, d_f, g_f, lag):
    """
    Apply the guess function to a forecast kernel in-place.

    This is equivalent to ``k[t] = k[t] @ m[t]``, where ::

        m[t] = [[ d_f[t+lag], g_f[t+lag] ],
                [ 0,                   1 ]]

    Note that this also applies the boundary conditions.

    Parameters
    ----------
    k : (n_frames-lag, n_indices, 2) ndarray of float
        Input kernel.
    d_f : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    g_f : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.

    """
    n = len(g_f)
    ni = k.shape[1]
    assert k.shape == (n - lag, ni, 2)
    assert g_f.shape == (n,)
    for t in range(n - lag):
        for i in range(ni):
            k[t, i, 1] += k[t, i, 0] * g_f[t + lag]
        if not d_f[t + lag]:
            for i in range(ni):
                k[t, i, 0] = 0.0


@nb.njit
def backward_guess(k, d_b, g_b, lag):
    """
    Apply the guess function to a aftcast kernel in-place.

    This is equivalent to ``k[t] = m[t] @ k[t]``, where ::

        m[t] = [[ 1, g_b[t] ],
                [ 0, d_b[t] ]]

    Parameters
    ----------
    k : (n_frames-lag, 2, n_indices) ndarray of float
        Input kernel.
    d_b : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    g_b : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.

    """
    n = len(g_b)
    nj = k.shape[2]
    assert k.shape == (n - lag, 2, nj)
    assert g_b.shape == (n,)
    for t in range(n - lag):
        for j in range(nj):
            k[t, 0, j] += g_b[t] * k[t, 1, j]
        if not d_b[t]:
            for j in range(nj):
                k[t, 1, j] = 0.0


@nb.njit
def _kernel(k, w, lag):
    """
    Multiply input kernel by weights and make time the last axis.

    This is equivalent to ``w[:-lag] * numpy.moveaxis(k, 0, -1)``.

    Parameters
    ----------
    k : (n_frames-lag, n_left_indices, n_right_indices) ndarray of float
        Input kernel.
    w : (n_frames,) ndarray of float
        Weight of each frame. Note that the last `lag` frames must have
        zero weight.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    (n_left_indices, n_right_indices, n_frames-lag) ndarray of float
        Input kernel multiplied by weights, with the time axis moved
        to the last axis.

    """
    n = len(w)
    _, ni, nj = k.shape
    assert w.shape == (n,)
    assert k.shape == (n - lag, ni, nj)

    # the kernel can't be computed for the last `lag` frames,
    # so they should have weight zero
    for t in range(n - lag, n):
        assert w[t] == 0.0

    # equivalent to `m = w[:-lag] * np.moveaxis(k, 0, -1)`
    m = np.zeros((ni, nj, n - lag))
    for t in range(n - lag):
        for i in range(ni):
            for j in range(nj):
                m[i, j, t] = w[t] * k[t, i, j]
    return m


@nb.njit
def _check_lag(w, lag):
    assert 0 <= lag <= len(w)  # assumed by code in this module


@nb.njit
def _zeros(shape, out=None):
    if out is None:
        return np.zeros(shape)
    else:
        assert out.shape == shape
        out[:] = 0.0
        return out
