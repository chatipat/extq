import numpy as np

from . import linalg
from ._utils import sum_windows
from .stop import backward_stop, forward_stop

__all__ = [
    "gram_matrix",
    "reweight_matrices",
    "forward_feynman_kac_matrices",
    "backward_feynman_kac_matrices",
]


def gram_matrix(basis, weights, test_basis=None):
    """
    Compute the Gram matrix between basis and test basis functions.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis functions evaluated at each frame.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    test_basis : list of (n_frames[i], n_basis) ndarray of float, optional
        Test basis functions evaluated at each frame. Must have the
        same dimensions as `basis`. If None, `basis` is used.

    Returns
    -------
    c : (n_basis, n_basis) ndarray or sparse array of float
        The Gram matrix.

    """
    if test_basis is None:
        test_basis = basis
    n_basis = None
    c = 0.0
    for x, y, w in zip(test_basis, basis, weights, strict=True):
        n_frames = x.shape[0]
        n_basis = x.shape[1] if n_basis is None else n_basis
        assert x.shape == y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)
        t = np.flatnonzero(w)
        if len(t) == 0:
            continue
        xw = linalg.scale_rows(w[t], x[t]).T
        c += xw @ y[t]
    return c


def reweight_matrices(basis, lag, guess, test_basis=None):
    """
    Compute DGA matrices for the invariant distribution.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the change of measure.
    lag : int
        Lag time in unit of frames.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the change of measure. The last lag frames of each
        trajectory must be zero.
    test_basis : list of (n_frames[i], n_basis) ndarray of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the change of
        measure. If None, use the basis that is used to estimate the
        change of measure.

    Returns
    -------
    a : (n_basis, n_basis) ndarray or sparse array of float
        DGA matrix for the homogeneous term.
    b : (n_basis,) ndarray of float
        DGA matrix for the nonhomogeneous term.

    """
    assert lag > 0
    if test_basis is None:
        test_basis = basis
    n_basis = None
    a = 0.0
    b = 0.0
    for x, y, w in zip(test_basis, basis, guess, strict=True):
        n_frames = x.shape[0]
        n_basis = x.shape[1] if n_basis is None else n_basis
        assert x.shape == y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)

        t0 = np.flatnonzero(w)  # initial time
        if len(t0) == 0:
            continue
        t1 = t0 + lag  # final time
        assert t1[-1] < n_frames  # all times < n_frames

        dxw = linalg.scale_rows(w[t0], x[t1] - x[t0]).T
        a += dxw @ y[t0]
        b += dxw @ np.ones(len(t0), dtype=dxw.dtype)
    return a, b


def forward_feynman_kac_matrices(
    basis, weights, in_domain, function, guess, lag, test_basis=None
):
    """
    Compute DGA matrices for the forward Feynman-Kac problem.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the solution of the Feynman-Kac formula.
        Must be zero outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    function : list of (n_frames[i]-1,) ndarray of float
        Function to integrate. Note that is defined over transitions,
        not frames.
    guess : list of (n_frames[i],) ndarray of float
        Guess of the solution. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.

    Returns
    -------
    a : (n_basis, n_basis) ndarray or sparse array of float
        DGA matrix for the homogeneous term.
    b : (n_basis,) ndarray of float
        DGA matrix for the nonhomogeneous term.

    """
    assert lag > 0
    if test_basis is None:
        test_basis = basis
    function = _broadcast_integrand(function, guess)
    n_basis = None
    a = 0.0
    b = 0.0
    for x, y, w, d, f, g in zip(
        test_basis, basis, weights, in_domain, function, guess, strict=True
    ):
        n_frames = x.shape[0]
        n_basis = x.shape[1] if n_basis is None else n_basis
        assert x.shape == y.shape == (n_frames, n_basis)
        assert w.shape == d.shape == g.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)

        t0 = np.flatnonzero(w)  # initial time
        if len(t0) == 0:
            continue
        t1 = np.minimum(t0 + lag, forward_stop(d)[t0])  # final time
        assert t1[-1] < n_frames  # all times < n_frames

        xw = linalg.scale_rows(w[t0], x[t0]).T
        a += xw @ (y[t1] - y[t0])
        b += xw @ (g[t1] - g[t0] + sum_windows(f, t0, t1))
    return a, b


def backward_feynman_kac_matrices(
    basis, weights, in_domain, function, guess, lag, test_basis=None
):
    """
    Compute DGA matrices for the backward Feynman-Kac problem.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the solution of the Feynman-Kac formula.
        Must be zero outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    function : list of (n_frames[i]-1,) ndarray of float
        Function to integrate. Note that is defined over transitions,
        not frames.
    guess : list of (n_frames[i],) ndarray of float
        Guess of the solution. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.

    Returns
    -------
    a : (n_basis, n_basis) ndarray or sparse array of float
        DGA matrix for the homogeneous term.
    b : (n_basis,) ndarray of float
        DGA matrix for the nonhomogeneous term.

    """
    assert lag > 0
    if test_basis is None:
        test_basis = basis
    function = _broadcast_integrand(function, guess)
    n_basis = None
    a = 0.0
    b = 0.0
    for x, y, w, d, f, g in zip(
        test_basis, basis, weights, in_domain, function, guess, strict=True
    ):
        n_frames = x.shape[0]
        n_basis = x.shape[1] if n_basis is None else n_basis
        assert x.shape == y.shape == (n_frames, n_basis)
        assert w.shape == d.shape == g.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)

        t0 = np.flatnonzero(w)  # initial time
        if len(t0) == 0:
            continue
        t1 = np.maximum(t0 - lag, backward_stop(d)[t0])  # final time
        assert t1[0] >= 0  # all times >= 0

        xw = linalg.scale_rows(w[t0], x[t0]).T
        a += xw @ (y[t1] - y[t0])
        b += xw @ (g[t1] - g[t0] + sum_windows(f, t1, t0))
    return a, b


def _broadcast_integrand(f, trajs):
    if not np.iterable(f):
        f = [np.broadcast_to(f, traj.shape[0] - 1) for traj in trajs]
    return f
