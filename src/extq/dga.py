import numpy as np
from scipy import linalg
from scipy import sparse

from .stop import backward_stop
from .stop import forward_stop


def forward_committor(basis, weights, in_domain, guess, lag, test_basis=None):
    """Estimate the forward committor using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : list of (n_frames[i],) ndarray of float
        Reweighting factor to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray of float, optional
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
        a += (x[:-lag].T * w[:-lag]) @ (y[iy] - y[:-lag])
        b -= (x[:-lag].T * w[:-lag]) @ (g[iy] - g[:-lag])
    coeffs = linalg.solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


def backward_committor(basis, weights, in_domain, guess, lag, test_basis=None):
    """Estimate the backward committor using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : list of (n_frames[i],) ndarray of float
        Reweighting factor to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray of float, optional
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
        a += (x[lag:].T * w[:-lag]) @ (y[iy] - y[lag:])
        b -= (x[lag:].T * w[:-lag]) @ (g[iy] - g[lag:])
    coeffs = linalg.solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


def forward_committor_sparse(
    basis, weights, in_domain, guess, lag, test_basis=None
):
    """Estimate the forward committor using DGA with sparse basis sets.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) sparse matrix of float
        Sparse basis for estimating the committor. Must be zero outside
        of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Reweighting factor to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) sparse matrix of float, optional
        Sparse test basis against which to minimize the error. Must have
        the same dimension as the basis used to estimate the committor.
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
        a += x[:-lag].T @ sparse.diags(w[:-lag]) @ (y[iy] - y[:-lag])
        b -= x[:-lag].T @ sparse.diags(w[:-lag]) @ (g[iy] - g[:-lag])
    coeffs = sparse.linalg.spsolve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


def backward_committor_sparse(
    basis, weights, in_domain, guess, lag, test_basis=None
):
    """Estimate the backward committor using DGA with sparse basis sets.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) sparse matrix of float
        Sparse basis for estimating the committor. Must be zero outside
        of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Reweighting factor to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) sparse matrix of float, optional
        Sparse test basis against which to minimize the error. Must have
        the same dimension as the basis used to estimate the committor.
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
        a += x[lag:].T @ sparse.diags(w[:-lag]) @ (y[iy] - y[lag:])
        b -= x[lag:].T @ sparse.diags(w[:-lag]) @ (g[iy] - g[lag:])
    coeffs = sparse.linalg.spsolve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


def reweight(basis, lag, maxlag=None, guess=None, test_basis=None):
    """Estimate the reweighting factors to the invariant distribution.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray of float
        Basis for estimating the reweighting factors.
    lag : int
        Lag time in unit of frames.
    maxlag : int
        Number of frames at the end of each trajectory that are required
        to have zero weight. This is the maximum lag time the output
        weights can be used with by other methods.
    guess : list of (n_frames[i],) ndarray of float, optional
        Guess for the reweighting factors. The last maxlag frames of
        each trajectory must be zero.
        If None, use uniform weights (except for the last lag frames).
    test_basis : list of (n_frames[i], n_basis) ndarray of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the reweighting
        factors.
        If None, use the basis that is used to estimate the reweighting
        factors.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the reweighting factors at each frame of the
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
            w = np.ones(len(x))
            w[-maxlag:] = 0.0
            guess.append(w)
    a = 0.0
    b = 0.0
    for x, y, w in zip(test_basis, basis, guess):
        assert np.all(w[-maxlag:] == 0.0)
        a += ((x[lag:] - x[:-lag]).T * w[:-lag]) @ y[:-lag]
        b -= (x[lag:] - x[:-lag]).T @ w[:-lag]
    coeffs = linalg.solve(a, b)
    return [w * (y @ coeffs + 1.0) for y, w in zip(basis, guess)]


def reweight_sparse(basis, lag, maxlag=None, guess=None, test_basis=None):
    """Estimate the reweighting factors to the invariant distribution
    using sparse basis sets.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) sparse matrix of float
        Sparse basis for estimating the reweighting factors.
    lag : int
        Lag time in unit of frames.
    maxlag : int
        Number of frames at the end of each trajectory that are required
        to have zero weight. This is the maximum lag time the output
        weights can be used with by other methods.
    guess : list of (n_frames[i],) ndarray of float, optional
        Guess for the reweighting factors. The last maxlag frames of
        each trajectory must be zero.
        If None, use uniform weights (except for the last lag frames).
    test_basis : list of (n_frames[i], n_basis) sparse matrix of float, optional
        Sparse test basis against which to minimize the error. Must have
        the same dimension as the basis used to estimate the reweighting
        factors.
        If None, use the basis that is used to estimate the reweighting
        factors.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the reweighting factors at each frame of the
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
        a += (x[lag:] - x[:-lag]).T @ sparse.diags(w[:-lag]) @ y[:-lag]
        b -= (x[lag:] - x[:-lag]).T @ w[:-lag]
    coeffs = sparse.linalg.spsolve(a, b)
    return [w * (y @ coeffs + 1.0) for y, w in zip(basis, guess)]