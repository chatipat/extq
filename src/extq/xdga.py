import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy import linalg

from .extended import backward_modified_transitions
from .extended import forward_modified_transitions
from .extended import moving_matmul


def forward_extended_committor(
    basis,
    weights,
    transitions,
    in_domain,
    guess,
    lag,
    test_basis=None,
):
    """Estimate the forward extended committor using DGA.

    Parameters
    ----------
    basis : list of (n_domain_indices, n_frames[i], n_basis) ndarray of float
        Basis for estimating the extended committor. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames.
    in_domain : list of (n_indices, n_frames[i]) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : list of (n_indices, n_frames[i]) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_domain_indices, n_frames[i], n_basis) ndarray of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimated forward extended committor at each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, d, g in zip(
        test_basis, basis, weights, transitions, in_domain, guess
    ):
        assert np.all(w[-lag:] == 0.0)

        ni, _, nt = m.shape
        dtype = np.result_type(x, y, w, m, g)
        f = np.zeros((ni, ni, nt), dtype=dtype)
        m = forward_modified_transitions(m, d, f, g, lag, dtype=dtype)

        for i in range(ni):
            wx = w[:-lag, None] * x[i, :-lag]

            yi = 0.0
            gi = 0.0

            for j in range(ni):
                yi += m[i, j, :, None] * y[j, lag:]
                gi += m[i, j] * g[j, lag:]
            gi += m[i, ni]  # integral and boundary conditions

            yi -= y[i, :-lag]
            gi -= g[i, :-lag]

            a += wx.T @ yi
            b -= wx.T @ gi

    coeffs = linalg.solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


def backward_extended_committor(
    basis,
    weights,
    transitions,
    in_domain,
    guess,
    lag,
    test_basis=None,
):
    """Estimate the backward extended committor using DGA.

    Parameters
    ----------
    basis : list of (n_domain_indices, n_frames[i], n_basis) ndarray of float
        Basis for estimating the extended committor. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames.
    in_domain : list of (n_indices, n_frames[i]) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : list of (n_indices, n_frames[i]) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_domain_indices, n_frames[i], n_basis) ndarray of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimated backward extended committor at each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, d, g in zip(
        test_basis, basis, weights, transitions, in_domain, guess
    ):
        assert np.all(w[-lag:] == 0.0)

        ni, _, nt = m.shape
        dtype = np.result_type(x, y, w, m, g)
        f = np.zeros((ni, ni, nt), dtype=dtype)
        m = backward_modified_transitions(m, d, f, g, lag, dtype=dtype)

        for i in range(ni):
            wx = w[:-lag, None] * x[i, lag:]

            yi = 0.0
            gi = 0.0

            for j in range(ni):
                yi += m[j, i, :, None] * y[j, :-lag]
                gi += m[j, i] * g[j, :-lag]
            gi += m[ni, i]  # integral and boundary conditions

            yi -= y[i, lag:]
            gi -= g[i, lag:]

            a += wx.T @ yi
            b -= wx.T @ gi

    coeffs = linalg.solve(a, b)
    return [y @ coeffs + g for y, g in zip(basis, guess)]


def forward_extended_committor_sparse(
    basis,
    weights,
    transitions,
    in_domain,
    guess,
    lag,
    test_basis=None,
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
        frames.
    in_domain : list of (n_indices, n_frames[i]) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : list of (n_indices, n_frames[i]) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of list of (n_frames[i], n_basis) sparse matrix of float, optional
        Sparse test basis against which to minimize the error. Must have
        the same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimated forward extended committor at each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, d, g in zip(
        test_basis, basis, weights, transitions, in_domain, guess
    ):
        assert np.all(w[-lag:] == 0.0)

        ni, _, nt = m.shape
        dtype = np.result_type(*x, *y, w, m, g)
        f = np.zeros((ni, ni, nt), dtype=dtype)
        m = forward_modified_transitions(m, d, f, g, lag, dtype=dtype)

        for i in range(ni):
            wx = scipy.sparse.diags(w[:-lag]) @ x[i][:-lag]

            yi = 0.0
            gi = 0.0

            for j in range(ni):
                yi += scipy.sparse.diags(m[i, j]) @ y[j][lag:]
                gi += scipy.sparse.diags(m[i, j]) @ g[j][lag:]
            gi += m[i, ni]  # integral and boundary conditions

            yi -= y[i][:-lag]
            gi -= g[i][:-lag]

            a += wx.T @ yi
            b -= wx.T @ gi

    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return [
        np.array([yi @ coeffs + gi for yi, gi in zip(y, g)])
        for y, g in zip(basis, guess)
    ]


def backward_extended_committor_sparse(
    basis,
    weights,
    transitions,
    in_domain,
    guess,
    lag,
    test_basis=None,
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
        frames.
    in_domain : list of (n_indices, n_frames[i]) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : list of (n_indices, n_frames[i]) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of list of (n_frames[i], n_basis) sparse matrix of float, optional
        Sparse test basis against which to minimize the error. Must have
        the same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimated backward extended committor at each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, d, g in zip(
        test_basis, basis, weights, transitions, in_domain, guess
    ):
        assert np.all(w[-lag:] == 0.0)

        ni, _, nt = m.shape
        dtype = np.result_type(*x, *y, w, m, g)
        f = np.zeros((ni, ni, nt), dtype=dtype)
        m = backward_modified_transitions(m, d, f, g, lag, dtype=dtype)

        for i in range(ni):
            wx = scipy.sparse.diags(w[:-lag]) @ x[i][lag:]

            yi = 0.0
            gi = 0.0

            for j in range(ni):
                yi += scipy.sparse.diags(m[j, i]) @ y[j][:-lag]
                gi += scipy.sparse.diags(m[j, i]) @ g[j][:-lag]
            gi += m[ni, i]  # integral and boundary conditions

            yi -= y[i][lag:]
            gi -= g[i][lag:]

            a += wx.T @ yi
            b -= wx.T @ gi

    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return [
        np.array([yi @ coeffs + gi for yi, gi in zip(y, g)])
        for y, g in zip(basis, guess)
    ]
