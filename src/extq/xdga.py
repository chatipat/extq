import numpy as np
from more_itertools import zip_equal

from . import linalg
from .moving_semigroup import moving_matmul

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
    basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float
        Basis for estimating the extended committor. Must be zero
        outside of the domain.
    weights : (n_trajs,) array_like of (n_frames[traj],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Possible transitions of the index process between adjacent
        frames.
    in_domain : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.

    Returns
    -------
    (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Estimated forward extended committor at each frame.

    """
    return forward_extended_feynman_kac(
        basis,
        weights,
        transitions,
        in_domain,
        0.0,
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
    basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float
        Basis for estimating the mean first passage time . Must be zero
        outside of the domain.
    weights : (n_trajs,) array_like of (n_frames[traj],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Possible transitions of the index process between adjacent
        frames.
    in_domain : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Guess for the mean first passage time . Must obey boundary
        conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.

    Returns
    -------
    (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Estimated forward mean first passage time at each frame.

    """
    return forward_extended_feynman_kac(
        basis,
        weights,
        transitions,
        in_domain,
        1.0,
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
    basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float
        Basis for estimating the solution to the Feynman-Kac formula.
        Must be zero outside of the domain.
    weights : (n_trajs,) array_like of (n_frames[traj],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Possible transitions of the index process between adjacent
        frames.
    in_domain : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    function : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Function to integrate. Note that this is defined over
        transitions, not frames.
    guess : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Guess for the solution. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.

    Returns
    -------
    (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Estimate of the solution of the forward Feynman-Kac formulat at
        each frame.

    """
    assert lag > 0
    if test_basis is None:
        test_basis = basis
    function = _broadcast_integrand(function, transitions)

    n_indices = None
    n_basis = None
    a = 0.0
    b = 0.0
    for x, y, w, k, d, f, g in zip_equal(
        _adapt_basis(test_basis),
        _adapt_basis(basis),
        weights,
        _adapt_steps(transitions),
        _adapt_frames(in_domain),
        _adapt_steps(function),
        _adapt_frames(guess),
    ):
        n_frames = x[0].shape[0]
        n_indices = len(x) if n_indices is None else n_indices
        n_basis = x[0].shape[1] if n_basis is None else n_basis

        assert len(x) == n_indices
        assert all(xi.shape == (n_frames, n_basis) for xi in x)
        assert len(y) == n_indices
        assert all(yi.shape == (n_frames, n_basis) for yi in y)
        assert w.shape == (n_frames,)
        assert k.shape == (n_indices, n_indices, n_frames - 1)
        assert d.shape == (n_indices, n_frames)
        assert f.shape == (n_indices, n_indices, n_frames - 1)
        assert g.shape == (n_indices, n_frames)

        iw = np.flatnonzero(w)  # start of window
        if len(iw) == 0:
            continue
        ix = iw  # initial time
        iy = ix + lag  # final time
        assert iy[-1] < n_frames  # all frames < n_frames
        ik = ix  # kernel index

        m = np.zeros((n_frames - 1, n_indices + 1, n_indices + 1))
        m = np.moveaxis(m, 0, -1)
        m[:-1, :-1] = np.where(d[:, None, :-1], k, 0)
        m[:-1, -1] = np.where(d[:, :-1], np.sum(k * f, axis=1), g[:, :-1])
        m[-1, -1] = 1
        m = np.moveaxis(moving_matmul(np.moveaxis(m, -1, 0), lag), 0, -1)

        for i in range(n_indices):
            wx = linalg.scale_rows(w[iw], x[i][ix])

            yi = 0.0
            gi = 0.0

            for j in range(n_indices):
                yi += linalg.scale_rows(m[i, j][ik], y[j][iy])
                gi += linalg.scale_rows(m[i, j][ik], g[j][iy])
            gi += m[i, -1][ik]  # integral and boundary conditions

            yi -= y[i][ix]
            gi -= g[i][ix]

            a += wx.T @ yi
            b -= wx.T @ gi

    coeffs = linalg.solve(a, b)
    return transform(coeffs, basis, guess)


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
    basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float
        Basis for estimating the extended committor. Must be zero
        outside of the domain.
    weights : (n_trajs,) array_like of (n_frames[traj],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Possible transitions of the index process between adjacent
        frames.
    in_domain : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the extended
        committor. If None, use the basis that is used to estimate the
        extended committor.

    Returns
    -------
    (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Estimated backward extended committor at each frame.

    """
    return backward_extended_feynman_kac(
        basis,
        weights,
        transitions,
        in_domain,
        0.0,
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
    basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float
        Basis for estimating the mean first passage time . Must be zero
        outside of the domain.
    weights : (n_trajs,) array_like of (n_frames[traj],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Possible transitions of the index process between adjacent
        frames.
    in_domain : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    guess : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Guess for the mean first passage time . Must obey boundary
        conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.

    Returns
    -------
    (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Estimated backward mean first passage time at each frame.

    """
    return backward_extended_feynman_kac(
        basis,
        weights,
        transitions,
        in_domain,
        1.0,
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
    basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float
        Basis for estimating the solution to the Feynman-Kac formula.
        Must be zero outside of the domain.
    weights : (n_trajs,) array_like of (n_frames[traj],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Possible transitions of the index process between adjacent
        frames.
    in_domain : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    function : (n_indices, n_indices, n_trajs) array_like of (n_frames[traj] - 1,) ndarray of float
        Function to integrate. Note that this is defined over steps, not
        frames.
    guess : (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Guess for the solution. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : (n_indices, n_trajs) array_like of (n_frames[traj], n_basis) {ndarray, sparray} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.

    Returns
    -------
    (n_indices, n_trajs) array_like of (n_frames[traj],) ndarray of float
        Estimate of the solution of the backward Feynman-Kac formula at
        each frame.

    """
    assert lag > 0
    if test_basis is None:
        test_basis = basis
    function = _broadcast_integrand(function, transitions)

    n_indices = None
    n_basis = None
    a = 0.0
    b = 0.0
    for x, y, w, k, d, f, g in zip_equal(
        _adapt_basis(test_basis),
        _adapt_basis(basis),
        weights,
        _adapt_steps(transitions),
        _adapt_frames(in_domain),
        _adapt_steps(function),
        _adapt_frames(guess),
    ):
        n_frames = x[0].shape[0]
        n_indices = len(x) if n_indices is None else n_indices
        n_basis = x[0].shape[1] if n_basis is None else n_basis

        assert len(x) == n_indices
        assert all(xi.shape == (n_frames, n_basis) for xi in x)
        assert len(y) == n_indices
        assert all(yi.shape == (n_frames, n_basis) for yi in y)
        assert w.shape == (n_frames,)
        assert k.shape == (n_indices, n_indices, n_frames - 1)
        assert d.shape == (n_indices, n_frames)
        assert f.shape == (n_indices, n_indices, n_frames - 1)
        assert g.shape == (n_indices, n_frames)

        iw = np.flatnonzero(w)  # start of window
        if len(iw) == 0:
            continue
        ix = iw + lag  # initial time
        assert ix[-1] < n_frames  # all frames < n_frames
        iy = ix - lag  # final time
        ik = iy  # kernel index

        m = np.zeros((n_frames - 1, n_indices + 1, n_indices + 1))
        m = np.moveaxis(m, 0, -1)
        m[:-1, :-1] = np.where(d[None, :, 1:], k, 0)
        m[-1, :-1] = np.where(d[:, 1:], np.sum(k * f, axis=0), g[:, 1:])
        m[-1, -1] = 1
        m = np.moveaxis(moving_matmul(np.moveaxis(m, -1, 0), lag), 0, -1)

        for i in range(n_indices):
            wx = linalg.scale_rows(w[iw], x[i][ix])

            yi = 0.0
            gi = 0.0

            for j in range(n_indices):
                yi += linalg.scale_rows(m[j, i][ik], y[j][iy])
                gi += linalg.scale_rows(m[j, i][ik], g[j][iy])
            gi += m[-1, i][ik]  # integral and boundary conditions

            yi -= y[i][ix]
            gi -= g[i][ix]

            a += wx.T @ yi
            b -= wx.T @ gi

    coeffs = linalg.solve(a, b)
    return transform(coeffs, basis, guess)


def transform(coeffs, basis, guess):
    return [
        [yi @ coeffs + gi for yi, gi in zip_equal(basis_i, guess_i)]
        for basis_i, guess_i in zip_equal(basis, guess)
    ]


def _broadcast_integrand(f, transitions):
    if not np.iterable(f):
        f = [
            [
                [np.broadcast_to(f, kij) for kij in transitions_ij]
                for transitions_ij in transitions_i
            ]
            for transitions_i in transitions
        ]
    return f


def _adapt_basis(a):
    return np.moveaxis(_objarray(a, 2), 1, 0).tolist()


def _adapt_frames(a):
    return map(np.array, np.moveaxis(_objarray(a, 2), 1, 0).tolist())


def _adapt_steps(a):
    return map(np.array, np.moveaxis(_objarray(a, 3), 2, 0).tolist())


def _objarray(a, ndim):
    shape = _shape(a, ndim)
    out = np.full(shape, None)
    for index in np.ndindex(shape):
        x = a
        for i in index:
            x = x[i]
        out[index] = x
    return out


def _shape(a, ndim):
    assert ndim >= 0
    if ndim == 0:
        return ()
    shapes = [_shape(ai, ndim - 1) for ai in a]
    assert all(shape == shapes[0] for shape in shapes)
    return (len(a), *shapes[0])
