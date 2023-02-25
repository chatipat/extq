"""Augmented process transition kernels."""

import numpy as np

from ..stop import backward_stop, forward_stop


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


def forward_feynman_kac_transitions(d, f, g, lag):
    """
    Transition kernel for a forward-in-time Feynman-Kac statistic.

    The element for the transition `t` to `t+lag` is ::

        k[t] = [[ d[s], g[s] - g[0] + sum(f[t:s]) ],
                [    0,                  1 ]]

    where ``s = t + min(lag, argmin(~d[t:]))``.

    Note that the identity element (``lag == 0``) is ::

        k[t] = [[ d[t], 0 ],
                [    0, 1 ]]

    which is a projection matrix because the homogenized solution of
    the Feynman-Kac problem must be zero outside of the domain.

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
    k = np.zeros((n - lag, 2, 2))
    if lag == 0:
        k[:, 0, 0] = d
        k[:, 1, 1] = 1.0
    else:
        stop = np.minimum(np.arange(lag, n), forward_stop(d)[:-lag])
        intf = np.insert(np.cumsum(f), 0, 0.0)
        k[:, 0, 0] = d[stop]
        k[:, 0, 1] = (g[stop] - g[:-lag]) + (intf[stop] - intf[:-lag])
        k[:, 1, 1] = 1.0
    return k


def backward_feynman_kac_transitions(d, f, g, lag):
    """
    Transition kernel for a backward-in-time Feynman-Kac statistic.

    The element for the transition `t` to `t+lag` is ::

        k[t] = [[                            d[s], 0 ],
                [ g[s] - g[lag] + sum(f[s:t+lag]), 1 ]]

    where ``s = t + max(0, lag-argmin(~d[t::-1]))``.

    Note that the identity element (``lag == 0``) is ::

        k[t] = [[ d[t], 0 ],
                [    0, 1 ]]

    which is a projection matrix because the homogenized solution of
    the Feynman-Kac problem must be zero outside of the domain.

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
    k = np.zeros((n - lag, 2, 2))
    if lag == 0:
        k[:, 0, 0] = d
        k[:, 1, 1] = 1.0
    else:
        stop = np.maximum(np.arange(n - lag), backward_stop(d)[lag:])
        intf = np.insert(np.cumsum(f), 0, 0.0)
        k[:, 0, 0] = d[stop]
        k[:, 1, 0] = (g[stop] - g[lag:]) + (intf[lag:] - intf[stop])
        k[:, 1, 1] = 1.0
    return k
