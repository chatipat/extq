import numba as nb
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from .stop import backward_stop
from .stop import forward_stop


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
        assert np.all(w[-lag:] == 0.0)
        iy = np.minimum(np.arange(lag, len(d)), forward_stop(d)[:-lag])
        assert np.all(iy < len(d))
        wx = scipy.sparse.diags(w[:-lag]) @ x[:-lag]
        a += wx.T @ (y[iy] - y[:-lag])
        b -= wx.T @ (g[iy] - g[:-lag])
    coeffs = _solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


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
        assert np.all(w[-lag:] == 0.0)
        iy = np.maximum(np.arange(len(d) - lag), backward_stop(d)[lag:])
        assert np.all(iy >= 0)
        wx = scipy.sparse.diags(w[:-lag]) @ x[lag:]
        a += wx.T @ (y[iy] - y[lag:])
        b -= wx.T @ (g[iy] - g[lag:])
    coeffs = _solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


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
        assert np.all(w[-lag:] == 0.0)
        ix = np.arange(len(d) - lag)
        iy = np.minimum(np.arange(lag, len(d)), forward_stop(d)[:-lag])
        assert np.all(iy < len(d))
        integral = iy - ix
        wx = scipy.sparse.diags(w[:-lag]) @ x[:-lag]
        a += wx.T @ (y[iy] - y[:-lag])
        b -= wx.T @ (g[iy] - g[:-lag] + integral)
    coeffs = _solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


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
        assert np.all(w[-lag:] == 0.0)
        ix = np.arange(lag, len(d))
        iy = np.maximum(np.arange(len(d) - lag), backward_stop(d)[lag:])
        assert np.all(iy >= 0)
        integral = ix - iy
        wx = scipy.sparse.diags(w[:-lag]) @ x[lag:]
        a += wx.T @ (y[iy] - y[lag:])
        b -= wx.T @ (g[iy] - g[lag:] + integral)
    coeffs = _solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


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
        assert np.all(w[-lag:] == 0.0)
        iy = np.minimum(np.arange(lag, len(d)), forward_stop(d)[:-lag])
        assert np.all(iy < len(d))
        integral = _forward_feynman_kac_helper(iy, f, lag)
        wx = scipy.sparse.diags(w[:-lag]) @ x[:-lag]
        a += wx.T @ (y[iy] - y[:-lag])
        b -= wx.T @ (g[iy] - g[:-lag] + integral)
    coeffs = _solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


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
        assert np.all(w[-lag:] == 0.0)
        iy = np.maximum(np.arange(len(d) - lag), backward_stop(d)[lag:])
        assert np.all(iy >= 0)
        integral = _backward_feynman_kac_helper(iy, f, lag)
        wx = scipy.sparse.diags(w[:-lag]) @ x[lag:]
        a += wx.T @ (y[iy] - y[lag:])
        b -= wx.T @ (g[iy] - g[lag:] + integral)
    coeffs = _solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


@nb.njit
def _forward_feynman_kac_helper(iy, f, lag):
    assert len(iy) + lag == len(f) + 1
    result = np.zeros(len(iy))
    for i in range(len(iy)):
        assert iy[i] <= i + lag
        for j in range(i, iy[i]):
            result[i] += f[j]
    return result


@nb.njit
def _backward_feynman_kac_helper(iy, f, lag):
    assert len(iy) + lag == len(f) + 1
    result = np.zeros(len(iy))
    for i in range(len(iy)):
        assert i <= iy[i]
        for j in range(iy[i], i + lag):
            result[i] += f[j]
    return result


def reweight(basis, lag, maxlag=None, guess=None, test_basis=None):
    """Estimate the change of measure to the invariant distribution.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the change of measure.
    lag : int
        Lag time in unit of frames.
    maxlag : int
        Number of frames at the end of each trajectory that are required
        to have zero weight. This is the maximum lag time the output
        weights can be used with by other methods.
    guess : list of (n_frames[i],) ndarray of float, optional
        Guess for the change of measure. The last maxlag frames of
        each trajectory must be zero.
        If None, use uniform weights (except for the last lag frames).
    test_basis : list of (n_frames[i], n_basis) ndarray of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the change of
        measure. If None, use the basis that is used to estimate the
        change of measure.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the change of measure at each frame of the
        trajectory.

    """
    if maxlag is None:
        maxlag = lag
    assert maxlag >= lag
    if test_basis is None:
        test_basis = basis
    if guess is None:
        guess = []
        for x in basis:
            w = np.ones(x.shape[0])
            w[-maxlag:] = 0.0
            guess.append(w)
    a = 0.0
    b = 0.0
    for x, y, w in zip(test_basis, basis, guess):
        assert np.all(w[-maxlag:] == 0.0)
        wdx = scipy.sparse.diags(w[:-lag]) @ (x[lag:] - x[:-lag])
        a += wdx.T @ y[:-lag]
        b -= np.ravel(wdx.sum(axis=0))
    coeffs = _solve(a, b)
    return [w * (y @ coeffs + 1.0) for y, w in zip(basis, guess)]


def _solve(a, b):
    if scipy.sparse.issparse(a):
        return scipy.sparse.linalg.spsolve(a, b)
    else:
        return scipy.linalg.solve(a, b)
