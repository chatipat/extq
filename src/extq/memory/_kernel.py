"""Kernels for calculating correlation matrices."""

import numpy as np

from ..integral import integral_windows
from ..stop import backward_stop
from ..stop import forward_stop


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
    k = constant_transitions(len(w), lag)
    return _kernel(k, w, lag)


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
    if lag == 0:
        k = forward_guess(d_f, g_f)
    else:
        k = forward_transitions(d_f, f_f, g_f, lag)
        k = k @ forward_guess(d_f, g_f)[lag:]
    return _kernel(k, w, lag)


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
    if lag == 0:
        k = backward_guess(d_b, g_b)
    else:
        k = backward_transitions(d_b, f_b, g_b, lag)
        k = backward_guess(d_b, g_b)[:-lag] @ k
    return _kernel(k, w, lag)


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
    if lag == 0:
        k = np.zeros((len(w), 1, 1))
    else:
        kb = kf = constant_transitions(len(w), 1)
        kv = observable(v, 1, 1)
        k = integral_windows(kb, kf, kv, 1, lag)
    return _kernel(k, w, lag)


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
    if lag == 0:
        k = np.zeros((len(w), 1, 2))
    else:
        kb = constant_transitions(len(w), 1)
        kf = forward_transitions(d_f, f_f, g_f, 1)
        kv = observable(v, 1, 2)
        k = integral_windows(kb, kf, kv, 1, lag)
        k = k @ forward_guess(d_f, g_f)[lag:]
    return _kernel(k, w, lag)


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
    if lag == 0:
        k = np.zeros((len(w), 2, 1))
    else:
        kb = backward_transitions(d_b, f_b, g_b, 1)
        kf = constant_transitions(len(w), 1)
        kv = observable(v, 2, 1)
        k = integral_windows(kb, kf, kv, 1, lag)
        k = backward_guess(d_b, g_b)[:-lag] @ k
    return _kernel(k, w, lag)


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
    if lag == 0:
        k = np.zeros((len(w), 2, 2))
    else:
        kb = backward_transitions(d_b, f_b, g_b, 1)
        kf = forward_transitions(d_f, f_f, g_f, 1)
        kv = observable(v, 2, 2)
        k = integral_windows(kb, kf, kv, 1, lag)
        k = backward_guess(d_b, g_b)[:-lag] @ k @ forward_guess(d_f, g_f)[lag:]
    return _kernel(k, w, lag)


def constant_transitions(n_frames, lag):
    """
    Transition kernel for a constant function.

    The element for the transition `t` to `t+lag` is ::

        k[t] = [[1]]

    Parameters
    ----------
    n_frames : int
        Number of frames in the trajectory.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    (n_frames[i]-lag, 1, 1) ndarray of float
        Constant function transition kernel at each frame.

    """
    assert n_frames >= lag
    return np.ones((n_frames - lag, 1, 1))


def forward_transitions(d, f, g, lag):
    """
    Transition kernel for a forecast.

    The element for the transition `t` to `t+lag` is ::

        k[t] = [[ d[s], b[s] + sum(f[t:s]) ],
                [    0,                  1 ]]

    where ``b = ~d * g`` and ``s = t + min(lag, argmin(~d[t:]))``.

    Note that the identity element (``lag == 0``) is ::

        k[t] = [[ d[t], b[t] ],
                [    0,    1 ]]

    which is a projection matrix because the forecast is restricted to
    an affine subspace.

    Parameters
    ----------
    d : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    f : (n_frames-1,) ndarray of float
        Function to integrate. Note that this is defined at each *step*,
        not at each frame.
    g : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    k : (n_frames-lag, 2, 2) ndarray of float
        At each step ``t``, the kernel corresponding to the transition
        from frame ``t`` to frame ``t+lag``.

    """
    n = len(d)
    f = np.broadcast_to(f, n - 1)
    assert n >= lag
    assert d.shape == (n,)
    assert f.shape == (n - 1,)
    assert g.shape == (n,)
    b = np.where(d, 0.0, g)
    k = np.zeros((n - lag, 2, 2))
    if lag == 0:
        k[:, 0, 0] = d
        k[:, 0, 1] = b
        k[:, 1, 1] = 1.0
    else:
        stop = np.minimum(np.arange(lag, n), forward_stop(d)[:-lag])
        intf = np.insert(np.cumsum(f), 0, 0.0)
        k[:, 0, 0] = d[stop]
        k[:, 0, 1] = b[stop] + (intf[stop] - intf[:-lag])
        k[:, 1, 1] = 1.0
    return k


def backward_transitions(d, f, g, lag):
    """
    Transition kernel for an aftcast.

    The element for the transition `t` to `t+lag` is ::

        k[t] = [[ 1, b[s] + sum(f[s:t+lag]) ],
                [ 0,                   d[s] ]]

    where ``b = ~d * g`` and ``s = t + max(0, lag-argmin(~d[t::-1]))``.

    Note that the identity element (``lag == 0``) is ::

        k[t] = [[ 1, b[t] ],
                [ 0, d[t] ]]

    which is a projection matrix because the aftcast is restricted to
    an affine subspace.

    Parameters
    ----------
    d : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    f : (n_frames-1,) ndarray of float
        Function to integrate. Note that this is defined at each *step*,
        not at each frame.
    g : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    k : (n_frames-lag, 2, 2) ndarray of float
        At each step ``t``, the kernel corresponding to the transition
        from frame ``t`` to frame ``t+lag``.

    """
    n = len(d)
    f = np.broadcast_to(f, n - 1)
    assert n >= lag
    assert d.shape == (n,)
    assert f.shape == (n - 1,)
    assert g.shape == (n,)
    b = np.where(d, 0.0, g)
    k = np.zeros((n - lag, 2, 2))
    if lag == 0:
        k[:, 0, 0] = 1.0
        k[:, 0, 1] = b
        k[:, 1, 1] = d
    else:
        stop = np.maximum(np.arange(n - lag), backward_stop(d)[lag:])
        intf = np.insert(np.cumsum(f), 0, 0.0)
        k[:, 0, 0] = 1.0
        k[:, 0, 1] = b[stop] + (intf[lag:] - intf[stop])
        k[:, 1, 1] = d[stop]
    return k


def observable(v, n_left_indices, n_right_indices):
    """
    Transition kernel for a single-step observable.

    Parameters
    ----------
    v : (n_frames-1,) ndarray of float
        Value of the observable at each step. ``v[t]`` is calculated
        from frames ``t`` and ``t+1``.
    n_left_indices : int
        Dimension of the backward-in-time transition kernel.
    n_right_indices : int
        Dimension of the forward-in-time transition kernel.

    Returns
    -------
    (n_frames, n_left_indices, n_right_indices) ndarray of float
        Transition kernel for the observable.

    """
    out = np.zeros((len(v), n_left_indices, n_right_indices))
    out[:, -1, 0] = v
    return out


def forward_guess(d, g):
    """
    Matrices to apply the guess function to a forecast kernel.

    This is ::

        m[t] = [[ d[t], g[t] ],
                [    0,    1 ]]

    Note that this also applies the boundary conditions.

    Parameters
    ----------
    d : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    g : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.

    Returns
    -------
    (n_frames, 2, 2) ndarray of float
        At each frame, a matrix to apply the guess function to an
        forecast kernel.

    """
    n = len(d)
    assert d.shape == (n,)
    assert g.shape == (n,)
    out = np.zeros((n, 2, 2))
    out[:, 0, 0] = d
    out[:, 0, 1] = g
    out[:, 1, 1] = 1.0
    return out


def backward_guess(d, g):
    """
    Matrices to apply the guess function to an aftcast kernel.

    This is ::

        m[t] = [[ 1, g[t] ],
                [ 0, d[t] ]]

    Note that this also applies the boundary conditions.

    Parameters
    ----------
    d : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    g : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.

    Returns
    -------
    (n_frames, 2, 2) ndarray of float
        At each frame, a matrix to apply the guess function to an
        aftcast kernel.

    """
    n = len(d)
    assert d.shape == (n,)
    assert g.shape == (n,)
    out = np.zeros((n, 2, 2))
    out[:, 0, 0] = 1.0
    out[:, 0, 1] = g
    out[:, 1, 1] = d
    return out


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
    assert 0 <= lag <= len(w)
    end = len(w) - lag
    assert np.all(w[end:] == 0.0)
    return w[:end] * np.moveaxis(k, 0, -1)
