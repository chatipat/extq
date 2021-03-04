import numpy as np
from scipy import linalg

from .extended import moving_matmul


def forward_extended_committor(
    basis,
    weights,
    transitions,
    guess,
    lag,
    test_basis=None,
    check=True,
):
    """Estimate the forward extended committor using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_indices, n_basis) ndarray of float
        Basis for estimating the extended committor. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_frames[i]-1, n_indices, n_indices) ndarray
        Possible transitions of the index process between adjacent
        frames. Note that indices 0 and n_indices-1 are special. Index 0
        indicates the reactant, and must have no transitions to it from
        any other index. Index n_indices-1 indicates the product, and
        must not have any transitions from it to any other index. Also,
        both indices 0 and n_indices-1 must have a single transition to
        itself.
    guess : list of (n_frames[i], n_indices) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_indices, n_basis) ndarray of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.
    check : bool, optional
        If True, throw an error if inputs are invalid. Disabling this
        can increase performance.

    Returns
    -------
    list of (n_frames[i], n_indices) ndarray of float
        Estimated forward extended committor at each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, g in zip(test_basis, basis, weights, transitions, guess):
        assert np.all(w[-lag:] == 0.0)

        if check:
            # check that transitions are valid
            assert np.all(m[:, 0, 0] == 1)
            assert np.all(m[:, 1:, 0] == 0)
            assert np.all(m[:, -1, -1] == 1)
            assert np.all(m[:, -1, :-1] == 0)

            # check that the bases and guess obey boundary conditions
            assert np.all(x[:, 0] == 0.0)
            assert np.all(x[:, -1] == 0.0)
            assert np.all(y[:, 0] == 0.0)
            assert np.all(y[:, -1] == 0.0)
            assert np.all(g[:, 0] == 0.0)
            assert np.all(g[:, -1] == 1.0)

        n_indices = m.shape[1]

        m = np.array(m, dtype=w.dtype, order="C")
        wm = moving_matmul(np.copy(m, order="C"), lag)
        wm *= w[:-lag, None, None]
        for i in range(1, n_indices - 1):
            for j in range(1, n_indices - 1):
                mask = wm[:, i, j] != 0.0
                if np.any(mask):
                    wmij = wm[mask, i, j]
                    xij = x[:-lag, i][mask]
                    yij = y[lag:, j][mask]
                    gij = g[lag:, j][mask]
                    a += (xij.T * wmij) @ yij
                    b -= (xij.T * wmij) @ gij
            mask = wm[:, i, -1] != 0.0
            if np.any(mask):
                wmij = wm[mask, i, -1]
                xij = x[:-lag, i][mask]
                b -= xij.T @ wmij

        mask = w[:-lag] != 0.0
        if np.any(mask):
            wi = w[:-lag][mask]
            for i in range(1, n_indices - 1):
                xi = x[:-lag, i][mask]
                yi = y[:-lag, i][mask]
                gi = g[:-lag, i][mask]
                a -= (xi.T * wi) @ yi
                b += (xi.T * wi) @ gi

    coeffs = linalg.solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


def backward_extended_committor(
    basis,
    weights,
    transitions,
    guess,
    lag,
    test_basis=None,
    check=True,
):
    """Estimate the backward extended committor using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_indices, n_basis) ndarray of float
        Basis for estimating the extended committor. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_frames[i]-1, n_indices, n_indices) ndarray
        Possible transitions of the index process between adjacent
        frames. Note that indices 0 and n_indices-1 are special. Index 0
        indicates the reactant, and must have no transitions to it from
        any other index. Index n_indices-1 indicates the product, and
        must not have any transitions from it to any other index. Also,
        both indices 0 and n_indices-1 must have a single transition to
        itself.
    guess : list of (n_frames[i], n_indices) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_indices, n_basis) ndarray of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.
    check : bool, optional
        If True, throw an error if inputs are invalid. Disabling this
        can increase performance.

    Returns
    -------
    list of (n_frames[i], n_indices) ndarray of float
        Estimated backward extended committor at each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, g in zip(test_basis, basis, weights, transitions, guess):
        assert np.all(w[-lag:] == 0.0)

        if check:
            # check that transitions are valid
            assert np.all(m[:, 0, 0] == 1)
            assert np.all(m[:, 1:, 0] == 0)
            assert np.all(m[:, -1, -1] == 1)
            assert np.all(m[:, -1, :-1] == 0)

            # check that the bases and guess obey boundary conditions
            assert np.all(x[:, 0] == 0.0)
            assert np.all(x[:, -1] == 0.0)
            assert np.all(y[:, 0] == 0.0)
            assert np.all(y[:, -1] == 0.0)
            assert np.all(g[:, 0] == 1.0)
            assert np.all(g[:, -1] == 0.0)

        n_indices = m.shape[1]

        m = np.array(m, dtype=w.dtype, order="C")
        wm = moving_matmul(m, lag)
        wm *= w[:-lag, None, None]
        for i in range(1, n_indices - 1):
            for j in range(1, n_indices - 1):
                mask = wm[:, j, i] != 0.0
                if np.any(mask):
                    wmij = wm[mask, j, i]
                    xij = x[lag:, i][mask]
                    yij = y[:-lag, j][mask]
                    gij = g[:-lag, j][mask]
                    a += (xij.T * wmij) @ yij
                    b -= (xij.T * wmij) @ gij
            mask = wm[:, 0, i] != 0.0
            if np.any(mask):
                wmij = wm[mask, 0, i]
                xij = x[lag:, i][mask]
                b -= xij.T @ wmij

        mask = w[:-lag] != 0.0
        if np.any(mask):
            wi = w[:-lag][mask]
            for i in range(1, n_indices - 1):
                xi = x[lag:, i][mask]
                yi = y[lag:, i][mask]
                gi = g[lag:, i][mask]
                a -= (xi.T * wi) @ yi
                b += (xi.T * wi) @ gi

    coeffs = linalg.solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]
