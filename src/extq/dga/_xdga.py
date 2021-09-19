import numpy as np
import scipy.sparse

from ..extended import backward_modified_transitions
from ..extended import forward_modified_transitions
from ._utils import extended_transform
from ._utils import solve


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
    basis : list of list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the extended committor. Must be zero
        outside of the domain. The outer list is over trajectories;
        the inner list is over indices.
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
    test_basis : list of list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
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
        ai, bi = _forward_helper(x, y, w, m, d, 0.0, g, lag)
        a += ai
        b += bi

    coeffs = solve(a, b)
    return extended_transform(coeffs, basis, guess)


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
    basis : list of list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the extended committor. Must be zero
        outside of the domain. The outer list is over trajectories;
        the inner list is over indices.
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
    test_basis : list of list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
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
        ai, bi = _backward_helper(x, y, w, m, d, 0.0, g, lag)
        a += ai
        b += bi

    coeffs = solve(a, b)
    return extended_transform(coeffs, basis, guess)


def forward_extended_mfpt(
    basis,
    weights,
    transitions,
    in_domain,
    guess,
    lag,
    test_basis=None,
):
    """Estimate the forward mean first passage time using DGA.

    Parameters
    ----------
    basis : list of list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the mean first passage time . Must be zero
        outside of the domain. The outer list is over trajectories;
        the inner list is over indices.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames.
    in_domain : list of (n_indices, n_frames[i]) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : list of (n_indices, n_frames[i]) ndarray of float
        Guess for the mean first passage time . Must obey boundary
        conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimated forward mean first passage time at each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, d, g in zip(
        test_basis, basis, weights, transitions, in_domain, guess
    ):
        ai, bi = _forward_helper(x, y, w, m, d, 1.0, g, lag)
        a += ai
        b += bi

    coeffs = solve(a, b)
    return extended_transform(coeffs, basis, guess)


def backward_extended_mfpt(
    basis,
    weights,
    transitions,
    in_domain,
    guess,
    lag,
    test_basis=None,
):
    """Estimate the backward mean first passage time using DGA.

    Parameters
    ----------
    basis : list of list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the mean first passage time . Must be zero
        outside of the domain. The outer list is over trajectories;
        the inner list is over indices.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames.
    in_domain : list of (n_indices, n_frames[i]) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : list of (n_indices, n_frames[i]) ndarray of float
        Guess for the mean first passage time . Must obey boundary
        conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimated backward mean first passage time at each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, d, g in zip(
        test_basis, basis, weights, transitions, in_domain, guess
    ):
        ai, bi = _backward_helper(x, y, w, m, d, 1.0, g, lag)
        a += ai
        b += bi

    coeffs = solve(a, b)
    return extended_transform(coeffs, basis, guess)


def forward_extended_feynman_kac(
    basis,
    weights,
    transitions,
    in_domain,
    function,
    guess,
    lag,
    test_basis=None,
):
    """Solve the forward Feynman-Kac formula using DGA.

    Parameters
    ----------
    basis : list of list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the solution to the Feynman-Kac formula.
        Must be zero outside of the domain. The outer list is over
        trajectories; the inner list is over indices.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames.
    in_domain : list of (n_indices, n_frames[i]) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    function : list of (n_indices, n_frames[i]-1) ndarray of float
        Function to integrate. Note that this is defined over
        transitions, not frames.
    guess : list of (n_indices, n_frames[i]) ndarray of float
        Guess for the solution. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimate of the solution of the forward Feynman-Kac formulat at
        each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, d, f, g in zip(
        test_basis, basis, weights, transitions, in_domain, function, guess
    ):
        ai, bi = _forward_helper(x, y, w, m, d, f, g, lag)
        a += ai
        b += bi

    coeffs = solve(a, b)
    return extended_transform(coeffs, basis, guess)


def backward_extended_feynman_kac(
    basis,
    weights,
    transitions,
    in_domain,
    function,
    guess,
    lag,
    test_basis=None,
):
    """Solve the backward Feynman-Kac formula using DGA.

    Parameters
    ----------
    basis : list of list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the solution to the Feynman-Kac formula.
        Must be zero outside of the domain. The outer list is over
        trajectories; the inner list is over indices.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames.
    in_domain : list of (n_indices, n_frames[i]) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    function : list of (n_indices, n_frames[i]-1) ndarray of float
        Function to integrate. Note that this is defined over
        transitions, not frames.
    guess : list of (n_indices, n_frames[i]) ndarray of float
        Guess for the solution. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimate of the solution of the backward Feynman-Kac formulat at
        each frame.

    """
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, m, d, f, g in zip(
        test_basis, basis, weights, transitions, in_domain, function, guess
    ):
        ai, bi = _backward_helper(x, y, w, m, d, f, g, lag)
        a += ai
        b += bi

    coeffs = solve(a, b)
    return extended_transform(coeffs, basis, guess)


def _forward_helper(x, y, w, m, d, f, g, lag):
    assert np.all(w[-lag:] == 0.0)

    ni, _, nt = m.shape
    dtype = np.result_type(*x, *y, w, m, g)
    if np.ndim(f) == 0:
        f = np.full((ni, ni, nt), f, dtype=dtype)
    m = forward_modified_transitions(m, d, f, g, lag, dtype=dtype)

    a = 0.0
    b = 0.0

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

    return a, b


def _backward_helper(x, y, w, m, d, f, g, lag):
    assert np.all(w[-lag:] == 0.0)

    ni, _, nt = m.shape
    dtype = np.result_type(*x, *y, w, m, g)
    if np.ndim(f) == 0:
        f = np.full((ni, ni, nt), f, dtype=dtype)
    m = backward_modified_transitions(m, d, f, g, lag, dtype=dtype)

    a = 0.0
    b = 0.0

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

    return a, b
