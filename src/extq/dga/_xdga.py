import numba as nb
import numpy as np
from more_itertools import zip_equal

from .. import linalg
from ..moving_semigroup import moving_matmul

__all__ = [
    "forward_extended_committor",
    "forward_extended_mfpt",
    "forward_extended_feynman_kac",
    "backward_extended_committor",
    "backward_extended_mfpt",
    "backward_extended_feynman_kac",
]


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
    return forward_extended_feynman_kac(
        basis,
        weights,
        transitions,
        in_domain,
        np.zeros(len(weights)),
        guess,
        lag,
        test_basis=test_basis,
    )


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
    return forward_extended_feynman_kac(
        basis,
        weights,
        transitions,
        in_domain,
        np.ones(len(weights)),
        guess,
        lag,
        test_basis=test_basis,
    )


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
    assert lag > 0
    if test_basis is None:
        test_basis = basis
    n_indices = None
    n_basis = None
    a = 0.0
    b = 0.0
    for x, y, w, m, d, f, g in zip_equal(
        test_basis, basis, weights, transitions, in_domain, function, guess
    ):
        n_frames = x[0].shape[0]
        n_indices = len(x) if n_indices is None else n_indices
        n_basis = x[0].shape[1] if n_basis is None else n_basis
        f = np.broadcast_to(f, (n_indices, n_indices, n_frames - 1))

        assert len(x) == n_indices
        assert (xi.shape == (n_frames, n_basis) for xi in x)
        assert len(y) == n_indices
        assert (yi.shape == (n_frames, n_basis) for yi in y)
        assert w.shape == (n_frames,)
        assert m.shape == (n_indices, n_indices, n_frames - 1)
        assert d.shape == (n_indices, n_frames)
        assert f.shape == (n_indices, n_indices, n_frames - 1)
        assert g.shape == (n_indices, n_frames)

        assert np.all(w[max(0, n_frames - lag) :] == 0.0)
        if n_frames <= lag:
            continue

        m = _forward_transitions_helper(m, d, f, g, lag)
        m = np.moveaxis(m, 0, -1)

        for i in range(n_indices):
            wx = linalg.scale_rows(w[:-lag], x[i][:-lag])

            yi = 0.0
            gi = 0.0

            for j in range(n_indices):
                yi += linalg.scale_rows(m[i, j], y[j][lag:])
                gi += linalg.scale_rows(m[i, j], g[j][lag:])
            gi += m[i, -1]  # integral and boundary conditions

            yi -= y[i][:-lag]
            gi -= g[i][:-lag]

            a += wx.T @ yi
            b -= wx.T @ gi

    coeffs = linalg.solve(a, b)
    return transform(coeffs, basis, guess)


@nb.njit
def _forward_transitions_helper(m, d, f, g, lag):
    n_indices = d.shape[0]
    n_frames = d.shape[1]
    r = np.zeros((n_frames - 1, n_indices + 1, n_indices + 1))
    for n in range(n_frames - 1):
        for i in range(n_indices):
            if d[i, n]:
                for j in range(n_indices):
                    r[n, i, j] = m[i, j, n]
                    r[n, i, -1] += m[i, j, n] * f[i, j, n]  # integral
            else:
                r[n, i, -1] = g[i, n]  # boundary conditions
        r[n, -1, -1] = 1.0
    r = moving_matmul(r, lag)
    return r


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
    return backward_extended_feynman_kac(
        basis,
        weights,
        transitions,
        in_domain,
        np.zeros(len(weights)),
        guess,
        lag,
        test_basis=test_basis,
    )


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
    return backward_extended_feynman_kac(
        basis,
        weights,
        transitions,
        in_domain,
        np.ones(len(weights)),
        guess,
        lag,
        test_basis=test_basis,
    )


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
    assert lag > 0
    if test_basis is None:
        test_basis = basis
    n_indices = None
    n_basis = None
    a = 0.0
    b = 0.0
    for x, y, w, m, d, f, g in zip_equal(
        test_basis, basis, weights, transitions, in_domain, function, guess
    ):
        n_frames = x[0].shape[0]
        n_indices = len(x) if n_indices is None else n_indices
        n_basis = x[0].shape[1] if n_basis is None else n_basis
        f = np.broadcast_to(f, (n_indices, n_indices, n_frames - 1))

        assert len(x) == n_indices
        assert (xi.shape == (n_frames, n_basis) for xi in x)
        assert len(y) == n_indices
        assert (yi.shape == (n_frames, n_basis) for yi in y)
        assert w.shape == (n_frames,)
        assert m.shape == (n_indices, n_indices, n_frames - 1)
        assert d.shape == (n_indices, n_frames)
        assert f.shape == (n_indices, n_indices, n_frames - 1)
        assert g.shape == (n_indices, n_frames)

        assert np.all(w[max(0, n_frames - lag) :] == 0.0)
        if n_frames <= lag:
            continue

        m = _backward_transitions_helper(m, d, f, g, lag)
        m = np.moveaxis(m, 0, -1)

        for i in range(n_indices):
            wx = linalg.scale_rows(w[:-lag], x[i][lag:])

            yi = 0.0
            gi = 0.0

            for j in range(n_indices):
                yi += linalg.scale_rows(m[j, i], y[j][:-lag])
                gi += linalg.scale_rows(m[j, i], g[j][:-lag])
            gi += m[-1, i]  # integral and boundary conditions

            yi -= y[i][lag:]
            gi -= g[i][lag:]

            a += wx.T @ yi
            b -= wx.T @ gi

    coeffs = linalg.solve(a, b)
    return transform(coeffs, basis, guess)


@nb.njit
def _backward_transitions_helper(m, d, f, g, lag):
    n_indices = d.shape[0]
    n_frames = d.shape[1]
    r = np.zeros((n_frames - 1, n_indices + 1, n_indices + 1))
    for n in range(n_frames - 1):
        for j in range(n_indices):
            if d[j, n + 1]:
                for i in range(n_indices):
                    r[n, i, j] = m[i, j, n]
                    r[n, -1, j] += m[i, j, n] * f[i, j, n]  # integral
            else:
                r[n, -1, j] = g[j, n + 1]  # boundary conditions
        r[n, -1, -1] = 1.0
    r = moving_matmul(r, lag)
    return r


def transform(coeffs, basis, guess):
    return [
        np.array([yi @ coeffs + gi for yi, gi in zip_equal(y, g)])
        for y, g in zip_equal(basis, guess)
    ]
