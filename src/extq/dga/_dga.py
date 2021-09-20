import numba as nb
import numpy as np
import scipy.sparse

from ..moving_semigroup import moving_semigroup
from ..stop import backward_stop
from ..stop import forward_stop
from ._utils import solve
from ._utils import transform


def forward_committor(basis, weights, in_domain, guess, lag, test_basis=None):
    """Estimate the forward committor using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the committor.
        If None, use the basis that is used to estimate the committor.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the forward committor at each frame of the
        trajectory.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, d, g in zip(test_basis, basis, weights, in_domain, guess):
        ai, bi = _dga_helper(x, y, w, d, 0.0, g, lag, forward=True)
        a += ai
        b += bi
    coeffs = solve(a, b)
    return transform(coeffs, basis, guess)


def backward_committor(basis, weights, in_domain, guess, lag, test_basis=None):
    """Estimate the backward committor using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the committor.
        If None, use the basis that is used to estimate the committor.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the backward committor at each frame of the
        trajectory.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, d, g in zip(test_basis, basis, weights, in_domain, guess):
        ai, bi = _dga_helper(x, y, w, d, 0.0, g, lag, backward=True)
        a += ai
        b += bi
    coeffs = solve(a, b)
    return transform(coeffs, basis, guess)


def reversible_committor(
    basis, weights, in_domain, guess, lag, test_basis=None
):
    """Estimate the committor for reversible processes using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the committor.
        If None, use the basis that is used to estimate the committor.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the committor at each frame of the trajectory.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, d, g in zip(test_basis, basis, weights, in_domain, guess):
        ai, bi = _dga_helper(
            x, y, w, d, 0.0, g, lag, forward=True, backward=True
        )
        a += ai
        b += bi
    coeffs = solve(a, b)
    return transform(coeffs, basis, guess)


def forward_mfpt(basis, weights, in_domain, guess, lag, test_basis=None):
    """Estimate the forward mean first passage time using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the mean first passage time. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the forward mean first passage time at each frame of
        the trajectory.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, d, g in zip(test_basis, basis, weights, in_domain, guess):
        ai, bi = _dga_helper(x, y, w, d, 1.0, g, lag, forward=True)
        a += ai
        b += bi
    coeffs = solve(a, b)
    return transform(coeffs, basis, guess)


def backward_mfpt(basis, weights, in_domain, guess, lag, test_basis=None):
    """Estimate the backward mean first passage time using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the mean first passage time. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the backward mean first passage time at each frame of
        the trajectory.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, d, g in zip(test_basis, basis, weights, in_domain, guess):
        ai, bi = _dga_helper(x, y, w, d, 1.0, g, lag, backward=True)
        a += ai
        b += bi
    coeffs = solve(a, b)
    return transform(coeffs, basis, guess)


def reversible_mfpt(basis, weights, in_domain, guess, lag, test_basis=None):
    """Estimate the mean first passage time for reversible processes using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the mean first passage time. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the mean first passage time at each frame of the trajectory.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, d, g in zip(test_basis, basis, weights, in_domain, guess):
        ai, bi = _dga_helper(
            x, y, w, d, 1.0, g, lag, forward=True, backward=True
        )
        a += ai
        b += bi
    coeffs = solve(a, b)
    return transform(coeffs, basis, guess)


def forward_feynman_kac(
    basis, weights, in_domain, function, guess, lag, test_basis=None
):
    """Solve the forward Feynman-Kac formula using DGA.

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
    list of (n_frames[i],) ndarray
        Estimate of the solution of the forward Feynman-Kac formula at
        each frame of the trajectory.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, d, f, g in zip(
        test_basis, basis, weights, in_domain, function, guess
    ):
        ai, bi = _dga_helper(x, y, w, d, f, g, lag, forward=True)
        a += ai
        b += bi
    coeffs = solve(a, b)
    return transform(coeffs, basis, guess)


def backward_feynman_kac(
    basis, weights, in_domain, function, guess, lag, test_basis=None
):
    """Solve the backward Feynman-Kac formula using DGA.

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
    list of (n_frames[i],) ndarray
        Estimate of the solution of the backward Feynman-Kac formula at
        each frame of the trajectory.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, d, f, g in zip(
        test_basis, basis, weights, in_domain, function, guess
    ):
        ai, bi = _dga_helper(x, y, w, d, f, g, lag, backward=True)
        a += ai
        b += bi
    coeffs = solve(a, b)
    return transform(coeffs, basis, guess)


def reversible_feynman_kac(
    basis, weights, in_domain, function, guess, lag, test_basis=None
):
    """Solve the Feynman-Kac formula for reversible processes using DGA.

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
    list of (n_frames[i],) ndarray
        Estimate of the solution of the Feynman-Kac formula at each
        frame of the trajectory.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, d, f, g in zip(
        test_basis, basis, weights, in_domain, function, guess
    ):
        ai, bi = _dga_helper(
            x, y, w, d, f, g, lag, forward=True, backward=True
        )
        a += ai
        b += bi
    coeffs = solve(a, b)
    return transform(coeffs, basis, guess)


def _dga_helper(x, y, w, d, f, g, lag, *, forward=False, backward=False):
    assert forward or backward
    assert np.all(w[-lag:] == 0.0)
    if np.ndim(f) == 0:
        f = np.broadcast_to(f, len(d) - 1)
    a = 0.0
    b = 0.0
    if forward:
        m, integral = _forward_helper(d, f, g, lag)
        xw = scipy.sparse.diags(w[:-lag]) @ x[:-lag]
        a += xw.T @ (scipy.sparse.diags(m) @ y[lag:] - y[:-lag])
        b -= xw.T @ (scipy.sparse.diags(m) @ g[lag:] - g[:-lag] + integral)
    if backward:
        m, integral = _backward_helper(d, f, g, lag)
        xw = scipy.sparse.diags(w[:-lag]) @ x[lag:]
        a += xw.T @ (scipy.sparse.diags(m) @ y[:-lag] - y[lag:])
        b -= xw.T @ (scipy.sparse.diags(m) @ g[:-lag] - g[lag:] + integral)
    return a, b


@nb.njit
def _forward_helper(d, f, g, lag):
    nt = len(f)
    assert len(d) == nt + 1
    assert len(g) == nt + 1

    a = np.empty((nt, 2))
    for t in range(nt):
        if d[t]:
            a[t, 0] = 1.0
            a[t, 1] = f[t]
        else:
            a[t, 0] = 0.0
            a[t, 1] = g[t]

    def op(in1, in2, out):
        out[0] = in1[0] * in2[0]
        out[1] = in1[1] + in1[0] * in2[1]

    a = moving_semigroup(a, lag, op)
    return a[:, 0], a[:, 1]


@nb.njit
def _backward_helper(d, f, g, lag):
    nt = len(f)
    assert len(d) == nt + 1
    assert len(g) == nt + 1

    a = np.empty((nt, 2))
    for t in range(nt):
        if d[t + 1]:
            a[t, 0] = 1.0
            a[t, 1] = f[t]
        else:
            a[t, 0] = 0.0
            a[t, 1] = g[t + 1]

    def op(in1, in2, out):
        out[0] = in1[0] * in2[0]
        out[1] = in1[1] * in2[0] + in2[1]

    a = moving_semigroup(a, lag, op)
    return a[:, 0], a[:, 1]
