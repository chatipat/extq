import numpy as np
import scipy.sparse
import scipy.sparse.linalg
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
    basis : list of (n_indices, n_frames[i], n_basis) ndarray of float
        Basis for estimating the extended committor. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames. Note that indices 0 and n_indices-1 are special. Index 0
        indicates the reactant, and must have no transitions to it from
        any other index. Index n_indices-1 indicates the product, and
        must not have any transitions from it to any other index. Also,
        both indices 0 and n_indices-1 must have a single transition to
        itself.
    guess : list of (n_indices, n_frames[i]) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_indices, n_frames[i], n_basis) ndarray of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.
    check : bool, optional
        If True, throw an error if inputs are invalid. Disabling this
        can increase performance.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimated forward extended committor at each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, g in zip(test_basis, basis, weights, transitions, guess):
        assert np.all(w[-lag:] == 0.0)

        if check:
            _check_transitions(m)

            # check that the bases and guess obey boundary conditions
            assert np.all(x[0] == 0.0)
            assert np.all(x[-1] == 0.0)
            assert np.all(y[0] == 0.0)
            assert np.all(y[-1] == 0.0)
            assert np.all(g[0] == 0.0)
            assert np.all(g[-1] == 1.0)

        n_indices = m.shape[0]

        m = np.moveaxis(m, -1, 0)
        m = np.array(m, dtype=w.dtype, order="C")
        m = moving_matmul(m, lag)
        m = np.moveaxis(m, 0, -1)

        wm = w[:-lag] * m
        for i in range(1, n_indices - 1):
            for j in range(1, n_indices - 1):
                mask = wm[i, j] != 0.0
                if np.any(mask):
                    wmij = wm[i, j, mask]
                    xij = x[i, :-lag][mask]
                    yij = y[j, lag:][mask]
                    gij = g[j, lag:][mask]
                    a += (xij.T * wmij) @ yij
                    b -= (xij.T * wmij) @ gij
            mask = wm[i, -1] != 0.0
            if np.any(mask):
                wmij = wm[i, -1, mask]
                xij = x[i, :-lag][mask]
                b -= xij.T @ wmij

        mask = w[:-lag] != 0.0
        if np.any(mask):
            wi = w[:-lag][mask]
            for i in range(1, n_indices - 1):
                xi = x[i, :-lag][mask]
                yi = y[i, :-lag][mask]
                gi = g[i, :-lag][mask]
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
    basis : list of (n_indices, n_frames[i], n_basis) ndarray of float
        Basis for estimating the extended committor. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames. Note that indices 0 and n_indices-1 are special. Index 0
        indicates the reactant, and must have no transitions to it from
        any other index. Index n_indices-1 indicates the product, and
        must not have any transitions from it to any other index. Also,
        both indices 0 and n_indices-1 must have a single transition to
        itself.
    guess : list of (n_indices, n_frames[i]) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_indices, n_frames[i], n_basis) ndarray of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.
    check : bool, optional
        If True, throw an error if inputs are invalid. Disabling this
        can increase performance.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimated backward extended committor at each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, g in zip(test_basis, basis, weights, transitions, guess):
        assert np.all(w[-lag:] == 0.0)

        if check:
            _check_transitions(m)

            # check that the bases and guess obey boundary conditions
            assert np.all(x[0] == 0.0)
            assert np.all(x[-1] == 0.0)
            assert np.all(y[0] == 0.0)
            assert np.all(y[-1] == 0.0)
            assert np.all(g[0] == 1.0)
            assert np.all(g[-1] == 0.0)

        n_indices = m.shape[0]

        m = np.moveaxis(m, -1, 0)
        m = np.array(m, dtype=w.dtype, order="C")
        m = moving_matmul(m, lag)
        m = np.moveaxis(m, 0, -1)

        wm = w[:-lag] * m
        for i in range(1, n_indices - 1):
            for j in range(1, n_indices - 1):
                mask = wm[j, i] != 0.0
                if np.any(mask):
                    wmij = wm[j, i, mask]
                    xij = x[i, lag:][mask]
                    yij = y[j, :-lag][mask]
                    gij = g[j, :-lag][mask]
                    a += (xij.T * wmij) @ yij
                    b -= (xij.T * wmij) @ gij
            mask = wm[0, i] != 0.0
            if np.any(mask):
                wmij = wm[0, i, mask]
                xij = x[i, lag:][mask]
                b -= xij.T @ wmij

        mask = w[:-lag] != 0.0
        if np.any(mask):
            wi = w[:-lag][mask]
            for i in range(1, n_indices - 1):
                xi = x[i, lag:][mask]
                yi = y[i, lag:][mask]
                gi = g[i, lag:][mask]
                a -= (xi.T * wi) @ yi
                b += (xi.T * wi) @ gi

    coeffs = linalg.solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


def forward_extended_committor_sparse(
    basis,
    weights,
    transitions,
    guess,
    lag,
    test_basis=None,
    check=True,
):
    """Estimate the forward extended committor with a sparse basis set.

    Parameters
    ----------
    basis : list of list of (n_frames[i], n_basis) sparse matrix of float
        Sparse basis for estimating the extended committor. Must be zero
        outside of the domain. The outer list is over trajectories; the
        inner list is over indices.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames. Note that indices 0 and n_indices-1 are special. Index 0
        indicates the reactant, and must have no transitions to it from
        any other index. Index n_indices-1 indicates the product, and
        must not have any transitions from it to any other index. Also,
        both indices 0 and n_indices-1 must have a single transition to
        itself.
    guess : list of (n_indices, n_frames[i]) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of list of (n_frames[i], n_basis) sparse matrix of float, optional
        Sparse test basis against which to minimize the error. Must have
        the same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.
    check : bool, optional
        If True, throw an error if inputs are invalid. Disabling this
        can increase performance.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimated forward extended committor at each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, g in zip(test_basis, basis, weights, transitions, guess):
        assert np.all(w[-lag:] == 0.0)

        if check:
            _check_transitions(m)

            # check that the bases and guess obey boundary conditions
            assert x[0].count_nonzero() == 0
            assert x[-1].count_nonzero() == 0
            assert y[0].count_nonzero() == 0
            assert y[-1].count_nonzero() == 0
            assert np.all(g[0] == 0.0)
            assert np.all(g[-1] == 1.0)

        n_indices = m.shape[0]

        m = np.moveaxis(m, -1, 0)
        m = np.array(m, dtype=w.dtype, order="C")
        m = moving_matmul(m, lag)
        m = np.moveaxis(m, 0, -1)

        wm = w[:-lag] * m
        for i in range(1, n_indices - 1):
            for j in range(1, n_indices - 1):
                a += x[i][:-lag].T @ scipy.sparse.diags(wm[i, j]) @ y[j][lag:]
                b -= x[i][:-lag].T @ scipy.sparse.diags(wm[i, j]) @ g[j][lag:]
            b -= x[i][:-lag].T @ wm[i, -1]  # since g[-1] = 1.0

        for i in range(1, n_indices - 1):
            a -= x[i][:-lag].T @ scipy.sparse.diags(w[:-lag]) @ y[i][:-lag]
            b += x[i][:-lag].T @ scipy.sparse.diags(w[:-lag]) @ g[i][:-lag]

    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return [
        np.array([yi @ coeffs + gi for yi, gi in zip(y, g)])
        for y, g in zip(basis, guess)
    ]


def backward_extended_committor_sparse(
    basis,
    weights,
    transitions,
    guess,
    lag,
    test_basis=None,
    check=True,
):
    """Estimate the backward extended committor with a sparse basis set.

    Parameters
    ----------
    basis : list of list of (n_frames[i], n_basis) sparse matrix of float
        Sparse basis for estimating the extended committor. Must be zero
        outside of the domain. The outer list is over trajectories; the
        inner list is over indices.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames. Note that indices 0 and n_indices-1 are special. Index 0
        indicates the reactant, and must have no transitions to it from
        any other index. Index n_indices-1 indicates the product, and
        must not have any transitions from it to any other index. Also,
        both indices 0 and n_indices-1 must have a single transition to
        itself.
    guess : list of (n_indices, n_frames[i]) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of list of (n_frames[i], n_basis) sparse matrix of float, optional
        Sparse test basis against which to minimize the error. Must have
        the same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.
    check : bool, optional
        If True, throw an error if inputs are invalid. Disabling this
        can increase performance.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimated backward extended committor at each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, g in zip(test_basis, basis, weights, transitions, guess):
        assert np.all(w[-lag:] == 0.0)

        if check:
            _check_transitions(m)

            # check that the bases and guess obey boundary conditions
            assert x[0].count_nonzero() == 0
            assert x[-1].count_nonzero() == 0
            assert y[0].count_nonzero() == 0
            assert y[-1].count_nonzero() == 0
            assert np.all(g[0] == 1.0)
            assert np.all(g[-1] == 0.0)

        n_indices = m.shape[0]

        m = np.moveaxis(m, -1, 0)
        m = np.array(m, dtype=w.dtype, order="C")
        m = moving_matmul(m, lag)
        m = np.moveaxis(m, 0, -1)

        wm = w[:-lag] * m
        for i in range(1, n_indices - 1):
            for j in range(1, n_indices - 1):
                a += x[i][lag:].T @ scipy.sparse.diags(wm[j, i]) @ y[j][:-lag]
                b -= x[i][lag:].T @ scipy.sparse.diags(wm[j, i]) @ g[j][:-lag]
            b -= x[i][lag:].T @ wm[0, i]  # since g[0] == 1.0

        for i in range(1, n_indices - 1):
            a -= x[i][lag:].T @ scipy.sparse.diags(w[:-lag]) @ y[i][lag:]
            b += x[i][lag:].T @ scipy.sparse.diags(w[:-lag]) @ g[i][lag:]

    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return [
        np.array([yi @ coeffs + gi for yi, gi in zip(y, g)])
        for y, g in zip(basis, guess)
    ]


def _check_transitions(m):
    """Check that transitions are valid."""
    assert m.ndim == 3
    assert m.shape[0] == m.shape[1]
    assert np.all(m[0, 0] == 1)
    assert np.all(m[1:, 0] == 0)
    assert np.all(m[-1, -1] == 1)
    assert np.all(m[-1, :-1] == 0)
