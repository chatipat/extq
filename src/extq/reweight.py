import numpy as np
from scipy import linalg
from scipy import sparse


def reweight(basis, lag, guess=None, test_basis=None):
    """Estimate the reweighting factors to the invariant distribution.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray of float
        Basis for estimating the reweighting factors.
    lag : int
        Lag time in unit of frames.
    guess : list of (n_frames[i],) ndarray of float, optional
        Guess for the reweighting factors. The last lag frames of each
        trajectory must be zero.
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
    if test_basis is None:
        test_basis = basis
    if guess is None:
        guess = []
        for x in basis:
            w = np.ones(len(x))
            w[-lag:] = 0.0
            guess.append(w)
    a = 0.0
    b = 0.0
    for x, y, w in zip(test_basis, basis, guess):
        assert np.all(w[-lag:] == 0.0)
        a += ((x[lag:] - x[:-lag]).T * w[:-lag]) @ y[:-lag]
        b -= (x[lag:] - x[:-lag]).T @ w[:-lag]
    coeffs = linalg.solve(a, b)
    return [w * (y @ coeffs + 1.0) for y, w in zip(basis, guess)]


def reweight_sparse(basis, lag, guess=None, test_basis=None):
    """Estimate the reweighting factors to the invariant distribution
    using sparse basis sets.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) sparse matrix of float
        Sparse basis for estimating the reweighting factors.
    lag : int
        Lag time in unit of frames.
    guess : list of (n_frames[i],) ndarray of float, optional
        Guess for the reweighting factors. The last lag frames of each
        trajectory must be zero.
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
    if test_basis is None:
        test_basis = basis
    if guess is None:
        guess = []
        for x in basis:
            w = np.ones(x.shape[0])
            w[-lag:] = 0.0
            guess.append(w)
    a = 0.0
    b = 0.0
    for x, y, w in zip(test_basis, basis, guess):
        assert np.all(w[-lag:] == 0.0)
        a += (x[lag:] - x[:-lag]).T @ sparse.diags(w[:-lag]) @ y[:-lag]
        b -= (x[lag:] - x[:-lag]).T @ w[:-lag]
    coeffs = sparse.linalg.spsolve(a, b)
    return [w * (y @ coeffs + 1.0) for y, w in zip(basis, guess)]
