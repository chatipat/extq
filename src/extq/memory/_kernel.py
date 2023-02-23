"""Kernels for calculating correlation matrices."""

import numpy as np

from ..integral import integral_windows
from ._transitions import (
    backward_feynman_kac_transitions,
    constant_transitions,
    forward_feynman_kac_transitions,
)


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


def forward_feynman_kac_kernel(w, d, f, g, lag):
    """
    Correlation matrix kernel for solving a forward-in-time Feynman-Kac
    problem.

    Parameters
    ----------
    w : (n_frames,) ndarray of float
        Weight of each frame. Note that the last `lag` frames must have
        zero weight.
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
    (2, 2, n_frames-lag) ndarray of float
        Correlation matrix kernel for computing correlation matrices for
        the forward-in-time Feynman-Kac problem.

    """
    k = forward_feynman_kac_transitions(d, f, g, lag)
    return _kernel(k, w, lag)


def backward_feynman_kac_kernel(w, d, f, g, lag):
    """
    Correlation matrix kernel for solving a backward-in-time Feynman-Kac
    problem.

    Parameters
    ----------
    w : (n_frames,) ndarray of float
        Weight of each frame. Note that the last `lag` frames must have
        zero weight.
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
    (2, 2, n_frames-lag) ndarray of float
        Correlation matrix kernel for computing correlation matrices for
        the backward-in-time Feynman-Kac problem.

    """
    k = backward_feynman_kac_transitions(d, f, g, lag)
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
    a forward-in-time Feynman-Kac statistic.

    For the statistic :math:`u_+`, the average computed is
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
        Guess of the statistic, with the correct boundary conditions.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    (1, 2, n_frames-lag) ndarray of float
        Correlation matrix for computing correlation matrices for
        ergodic averages involving a forward-in-time Feynman-Kac
        statistic.

    """
    if lag == 0:
        k = np.zeros((len(w), 1, 2))
    else:
        kb = constant_transitions(len(w), 1)
        kf = forward_feynman_kac_transitions(d_f, f_f, g_f, 1)
        kv = observable(v, 1, 2) @ forward_guess(d_f, g_f)[1:]
        k = integral_windows(kb, kf, kv, 1, lag)
    return _kernel(k, w, lag)


def backward_integral_kernel(w, d_b, v, f_b, g_b, lag):
    r"""
    Correlation matrix kernel for computing ergodic averages involving
    a backward-in-time Feynman-Kac statistic.

    For the statistic :math:`u_-`, the average computed is
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
        Guess of the statistic, with the correct boundary conditions.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    (2, 1, n_frames-lag) ndarray of float
        Correlation matrix for computing correlation matrices for
        ergodic averages involving a backward-in-time Feynman-Kac
        statistic.

    """
    if lag == 0:
        k = np.zeros((len(w), 2, 1))
    else:
        kb = backward_feynman_kac_transitions(d_b, f_b, g_b, 1)
        kf = constant_transitions(len(w), 1)
        kv = backward_guess(d_b, g_b)[:-1] @ observable(v, 2, 1)
        k = integral_windows(kb, kf, kv, 1, lag)
    return _kernel(k, w, lag)


def integral_kernel(w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag):
    r"""
    Correlation matrix kernel for computing ergodic averages involving
    both forward-in-time and backward-in-time Feynman-Kac statistics.

    For the forward-in-time statistic :math:`u_+` and the
    backward-in-time statistic :math:`u_-`, the average computed is
    :math:`\langle u_-(X_t) v(X_t,X_{t+1}) \rangle`.

    Parameters
    ----------
    w : (n_frames,) ndarray of float
        Weight of each frame. Note that the last `lag` frames must have
        zero weight.
    d_b : (n_frames,) ndarray of bool
        For the backward-in-time statistic, whether each frame is in the
        domain.
    d_f : (n_frames,) ndarray of bool
        For the forward-in-time statistic, whether each frame is in the
        domain.
    v : (n_frames-1,) ndarray of float
        Observable for which to compute the ergodic average. Note that
        this is defined at each *step*, not at each frame, so ``v[t]``
        is a function of frames ``t`` and ``t+1``.
    f_b : (n_frames-1,) ndarray of float
        Function to integrate for the backward-in-time statistic. Note
        that this is defined at each *step*, not at each frame.
    f_f : (n_frames-1,) ndarray of float
        Function to integrate for the forward-in-time statistic. Note
        that this is defined at each *step*, not at each frame.
    g_b : (n_frames,) ndarray of float
        Guess of the backward-in-time statistic, with the correct
        boundary conditions.
    g_f : (n_frames,) ndarray of float
        Guess of the forward-in-time statistic, with the correct
        boundary conditions.
    lag : int
        Lag time, in units of frames.

    Returns
    -------
    (2, 2, n_frames-lag) ndarray of float
        Correlation matrix for computing correlation matrices for
        ergodic averages involving both forward-in-time and
        backward-in-time Feynman-Kac statistics.

    """
    if lag == 0:
        k = np.zeros((len(w), 2, 2))
    else:
        kb = backward_feynman_kac_transitions(d_b, f_b, g_b, 1)
        kf = forward_feynman_kac_transitions(d_f, f_f, g_f, 1)
        kv = (
            backward_guess(d_b, g_b)[:-1]
            @ observable(v, 2, 2)
            @ forward_guess(d_f, g_f)[1:]
        )
        k = integral_windows(kb, kf, kv, 1, lag)
    return _kernel(k, w, lag)


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
    (n_frames-1, n_left_indices, n_right_indices) ndarray of float
        Transition kernel for the observable.

    """
    out = np.zeros((len(v), n_left_indices, n_right_indices))
    out[:, 0, 0] = v
    return out


def forward_guess(d, g):
    """
    Matrices to unhomogenize the forward-in-time Feynman-Kac problem.

    This is ::

        m[t] = [[ d[t], g[t] ],
                [    0,    1 ]]

    Parameters
    ----------
    d : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    g : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.

    Returns
    -------
    (n_frames, 2, 2) ndarray of float
        At each frame, a matrix to unhomogenize the Feynman-Kac problem.

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
    Matrices to unhomogenize the backward-in-time Feynman-Kac problem.

    This is ::

        m[t] = [[ d[t], 0 ],
                [ g[t], 1 ]]

    Parameters
    ----------
    d : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    g : (n_frames,) ndarray of float
        Guess of the solution, with the correct boundary conditions.

    Returns
    -------
    (n_frames, 2, 2) ndarray of float
        At each frame, a matrix to unhomogenize the Feynman-Kac problem.

    """
    n = len(d)
    assert d.shape == (n,)
    assert g.shape == (n,)
    out = np.zeros((n, 2, 2))
    out[:, 0, 0] = d
    out[:, 1, 0] = g
    out[:, 1, 1] = 1.0
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
