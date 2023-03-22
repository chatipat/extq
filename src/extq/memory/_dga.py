"""DGA with memory estimators for statistics."""


import numpy as np
import scipy.sparse
from more_itertools import zip_equal

from .. import linalg
from ..integral import integral_coeffs, integral_windows
from ._transitions import (
    backward_feynman_kac_transitions,
    forward_feynman_kac_transitions,
)

__all__ = [
    "constant_intermediate",
    "reweight",
    "reweight_matrices",
    "reweight_transform",
    "reweight_intermediate",
    "forward_committor",
    "forward_mfpt",
    "forward_feynman_kac",
    "forward_feynman_kac_matrices",
    "forward_feynman_kac_transform",
    "forward_feynman_kac_intermediate",
    "backward_committor",
    "backward_mfpt",
    "backward_feynman_kac",
    "backward_feynman_kac_matrices",
    "backward_feynman_kac_transform",
    "backward_feynman_kac_intermediate",
    "integral",
    "pointwise_integral",
]


def constant_intermediate(trajs, mem=0):
    """
    Returns an intermediate representation of the constant function of
    ones for use in further calculations.

    Parameters
    ----------
    trajs : sequence of (n_frames, ...) array-like
        Sequence of trajectories. Only the length of each trajectory is
        used.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.

    Returns
    -------
    c_output : object
        Intermediate representation of the constant function of ones.

    """
    out = []
    for traj in trajs:
        n_frames = traj.shape[0]
        k = np.ones((n_frames - 1, 1, 1))
        m = np.ones((n_frames, 1))

        u = np.zeros((n_frames, 1, mem + 2))
        u[:, 0, 0] = 1.0
        u[:, 0, 1] = -1.0

        out.append((k, m, u))
    return out


def reweight(basis, weights, lag, mem=0, test_basis=None, output="proj"):
    """
    Estimate the invariant distribution using DGA with memory.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the invariant distribution. The span of
        `basis` must *not* contain the constant function.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    lag : int
        Maximum lag time in units of frames.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.
    output : {'proj', 'coef', 'full'}, optional
        'proj':
            Return the projected estimate (default).
        'coef':
            Return DGA coefficients.
        'full':
            Return an intermediate representation for use in further
            calculation.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the invariant distribution.

    """
    a, b = reweight_matrices(
        basis, weights, lag, mem=mem, test_basis=test_basis
    )
    coef = _dga_mem(a, b, mem)
    if output == "proj":
        return reweight_transform(coef, basis, weights)
    elif output == "coef":
        return coef
    elif output == "full":
        return reweight_intermediate(coef, basis, weights)
    else:
        raise ValueError(f"output must be 'proj', 'coef', or 'full'")


def reweight_matrices(basis, weights, lag, mem=0, test_basis=None):
    """
    Compute matrices for estimating the invariant distribution.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the invariant distribution. The span of
        `basis` must *not* contain the constant function.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    lag : int
        Maximum lag time in units of frames.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    a : dict of {int : (n_basis, n_basis) ndarray of float}
        Homogeneous terms.
    b : dict of {int : (n_basis, mem + 2) ndarray of float}
        Nonhomogeneous terms.

    """
    if test_basis is None:
        test_basis = basis

    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    a0 = np.zeros((n_basis, n_basis))
    a = {t: np.zeros((n_basis, n_basis)) for t in range(1, mem + 2)}
    b = {t: np.zeros(n_basis) for t in range(1, mem + 2)}

    for x, y, w in zip_equal(test_basis, basis, weights):
        n_frames = len(w)
        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)

        if n_frames <= lag:
            assert np.all(w == 0.0)
            continue
        end = n_frames - lag
        assert np.all(w[end:] == 0.0)

        a0 += _build(w[:end], x[:end], y[:end])
        for n in range(1, mem + 2):
            t = n * dlag
            a[n] += _build(w[:end], x[t : end + t], y[:end])
            b[n] += w[:end] @ (x[t : end + t] - x[:end])

    # biorthonormalize x with respect to y
    solve = linalg.factorized(a0)
    a = {t: solve(a[t]) for t in range(1, mem + 2)}
    b = {t: solve(b[t]) for t in range(1, mem + 2)}

    # right multiply by [[1]] matrix and its memory
    # because full correlation matrix is [[a, b], [0, 1]]
    c = np.concatenate([[1.0, -1.0], np.zeros(mem)])
    b = {t: np.outer(b[t], c) for t in range(1, mem + 2)}

    return a, b


def reweight_transform(coef, basis, weights):
    """
    Returns the projected invariant distribution.

    Parameters
    ----------
    coef : (n_basis, mem + 2) ndarray of float
        DGA coefficients.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the invariant distribution. The span of
        `basis` must *not* contain the constant function.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the projected invariant distribution.

    """
    return [w * (y @ coef[:, 0] + 1.0) for y, w in zip_equal(basis, weights)]


def reweight_intermediate(coef, basis, weights):
    """
    Returns an intermediate representation of the invariant distribution
    for use in further calculations.

    Parameters
    ----------
    coef : (n_basis, mem + 2) ndarray of float
        DGA coefficients.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the invariant distribution. The span of
        `basis` must *not* contain the constant function.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.

    Returns
    -------
    w_output : object
        Intermediate representation of the invariant distribution.

    """
    n_basis, n_lags = coef.shape
    c = np.zeros(n_lags)
    c[0] = 1.0
    c[1] = -1.0
    out = []
    for x, w in zip_equal(basis, weights):
        n_frames = len(w)

        assert x.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)

        k = np.ones((n_frames - 1, 1, 1))

        m = np.ones((n_frames, 1))

        u = np.empty((n_frames, 1, n_lags))
        u[:, 0] = w[:, None] * (x @ coef + c)

        out.append((k, m, u))
    return out


def forward_committor(
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem=0,
    test_basis=None,
    output="proj",
):
    """
    Estimate the forward committor using DGA with memory.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the committor. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.
    output : {'proj', 'coef', 'full'}, optional
        'proj':
            Return the projected estimate (default).
        'coef':
            Return DGA coefficients.
        'full':
            Return an intermediate representation for use in further
            calculation.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the committor.

    """
    return forward_feynman_kac(
        basis,
        weights,
        in_domain,
        np.zeros(len(weights)),
        guess,
        lag,
        mem=mem,
        test_basis=test_basis,
        output=output,
    )


def forward_mfpt(
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem=0,
    test_basis=None,
    output="proj",
):
    """
    Estimate the forward mean first passage time (MFPT) using DGA with
    memory.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the MFPT. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the MFPT. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.
    output : {'proj', 'coef', 'full'}, optional
        'proj':
            Return the projected estimate (default).
        'coef':
            Return DGA coefficients.
        'full':
            Return an intermediate representation for use in further
            calculation.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the MFPT.

    """
    return forward_feynman_kac(
        basis,
        weights,
        in_domain,
        np.ones(len(weights)),
        guess,
        lag,
        mem=mem,
        test_basis=test_basis,
        output=output,
    )


def forward_feynman_kac(
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem=0,
    test_basis=None,
    output="proj",
):
    """
    Estimate the solution to a forward Feynman-Kac problem using DGA
    with memory.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    function : sequence of (n_frames[i] - 1,) ndarray of float
        Function to integrate. This is defined over *transitions*, not
        frames.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.
    output : {'proj', 'coef', 'full'}, optional
        'proj':
            Return the projected estimate (default).
        'coef':
            Return DGA coefficients.
        'full':
            Return an intermediate representation for use in further
            calculation.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the solution.

    """
    a, b = forward_feynman_kac_matrices(
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem=mem,
        test_basis=test_basis,
    )
    coef = _dga_mem(a, b, mem)
    if output == "proj":
        return forward_feynman_kac_transform(coef, basis, in_domain, guess)
    elif output == "coef":
        return coef
    elif output == "full":
        return forward_feynman_kac_intermediate(
            coef, basis, in_domain, function, guess
        )
    else:
        raise ValueError(f"output must be 'proj', 'coef', or 'full'")


def forward_feynman_kac_matrices(
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem=0,
    test_basis=None,
):
    """
    Compute matrices for solving a forward Feynman-Kac problem.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    function : sequence of (n_frames[i] - 1,) ndarray of float
        Function to integrate. This is defined over *transitions*, not
        frames.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    a : dict of {int : (n_basis, n_basis) ndarray of float}
        Homogeneous terms.
    b : dict of {int : (n_basis, mem + 2) ndarray of float}
        Nonhomogeneous terms.

    """
    if test_basis is None:
        test_basis = basis

    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    a0 = np.zeros((n_basis, n_basis))
    a = {t: np.zeros((n_basis, n_basis)) for t in range(1, mem + 2)}
    b = {t: np.zeros(n_basis) for t in range(1, mem + 2)}

    for x, y, w, d, f, g in zip_equal(
        test_basis, basis, weights, in_domain, function, guess
    ):
        n_frames = len(w)
        f = np.broadcast_to(f, n_frames - 1)
        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)

        if n_frames <= lag:
            assert np.all(w == 0.0)
            continue
        end = n_frames - lag
        assert np.all(w[end:] == 0.0)

        wx = linalg.scale_rows(w, x)

        a0 += _build(d[:end], wx[:end], y[:end])
        for n in range(1, mem + 2):
            t = n * dlag
            k = forward_feynman_kac_transitions(d, f, g, t)
            a[n] += _build(k[:end, 0, 0], wx[:end], y[t : end + t])
            b[n] += k[:end, 0, 1] @ wx[:end]

    # biorthonormalize x with respect to y
    solve = linalg.factorized(a0)
    a = {t: solve(a[t]) for t in range(1, mem + 2)}
    b = {t: solve(b[t]) for t in range(1, mem + 2)}

    # right multiply by [[1]] matrix and its memory
    # because full correlation matrix is [[a, b], [0, 1]]
    c = np.concatenate([[1.0, -1.0], np.zeros(mem)])
    b = {t: np.outer(b[t], c) for t in range(1, mem + 2)}

    return a, b


def forward_feynman_kac_transform(coef, basis, in_domain, guess):
    """
    Returns the projected solution of a forward Feynman-Kac problem.

    Parameters
    ----------
    coef : (n_basis, mem + 2) ndarray of float
        DGA coefficients.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the projected solution.

    """
    return [
        d * (y @ coef[:, 0]) + g
        for y, d, g in zip_equal(basis, in_domain, guess)
    ]


def forward_feynman_kac_intermediate(coef, basis, in_domain, function, guess):
    """
    Returns an intermediate representation of the solution of a forward
    Feynman-Kac problem for use in further calculations.

    Parameters
    ----------
    coef : (n_basis, mem + 2) ndarray of float
        DGA coefficients.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    function : sequence of (n_frames[i] - 1,) ndarray of float
        Function to integrate. This is defined over *transitions*, not
        frames.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.

    Returns
    -------
    f_output : object
        Intermediate representation of the solution.

    """
    n_basis, n_lags = coef.shape
    c = np.zeros(n_lags)
    c[0] = 1.0
    c[1] = -1.0
    out = []
    for y, d, f, g in zip_equal(basis, in_domain, function, guess):
        n_frames = len(d)
        f = np.broadcast_to(f, n_frames - 1)

        assert y.shape == (n_frames, n_basis)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)

        k = forward_feynman_kac_transitions(d, f, g, 1)

        m = np.zeros((n_frames, 2))
        m[:, 0] = d
        m[:, 1] = g

        u = np.empty((n_frames, 2, n_lags))
        u[:, 0] = y @ coef
        u[:, 1] = c

        out.append((k, m, u))
    return out


def backward_committor(
    w_output,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem=0,
    test_basis=None,
    output="proj",
):
    """
    Estimate the backward committor using DGA with memory.

    Parameters
    ----------
    w_output : object
        Intermediate representation from `reweight`.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the committor. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the committor.
        Must have the same dimension as `basis`. If `None`, use `basis`.
    output : {'proj', 'coef', 'full'}, optional
        'proj':
            Return the projected estimate (default).
        'coef':
            Return DGA coefficients.
        'full':
            Return an intermediate representation for use in further
            calculation.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the product of the committor and the invariant
        distribution.

    """
    return backward_feynman_kac(
        w_output,
        basis,
        weights,
        in_domain,
        np.zeros(len(weights)),
        guess,
        lag,
        mem=mem,
        test_basis=test_basis,
        output=output,
    )


def backward_mfpt(
    w_output,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem=0,
    test_basis=None,
    output="proj",
):
    """
    Estimate the backward mean first passage time (MFPT) using DGA
    with memory.

    Parameters
    ----------
    w_output : object
        Intermediate representation from `reweight`.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the MFPT. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the MFPT. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the MFPT. Must
        have the same dimension as `basis`. If `None`, use `basis`.
    output : {'proj', 'coef', 'full'}, optional
        'proj':
            Return the projected estimate (default).
        'coef':
            Return DGA coefficients.
        'full':
            Return an intermediate representation for use in further
            calculation.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the product of the MFPT and the invariant
        distribution.

    """
    return backward_feynman_kac(
        w_output,
        basis,
        weights,
        in_domain,
        np.ones(len(weights)),
        guess,
        lag,
        mem=mem,
        test_basis=test_basis,
        output=output,
    )


def backward_feynman_kac(
    w_output,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem=0,
    test_basis=None,
    output="proj",
):
    """
    Estimate the solution to a backward Feynman-Kac problem using DGA
    with memory.

    Parameters
    ----------
    w_output : object
        Intermediate representation from `reweight`.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    function : sequence of (n_frames[i] - 1,) ndarray of float
        Function to integrate. This is defined over *transitions*, not
        frames.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the solution.
        Must have the same dimension as `basis`. If `None`, use `basis`.
    output : {'proj', 'coef', 'full'}, optional
        'proj':
            Return the projected estimate (default).
        'coef':
            Return DGA coefficients.
        'full':
            Return an intermediate representation for use in further
            calculation.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the solution. Note that this is the product of the
        Feynman-Kac statistic and the invariant distribution.

    """
    a, b = backward_feynman_kac_matrices(
        w_output,
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem=mem,
        test_basis=test_basis,
    )
    coef = _dga_mem(a, b, mem)
    if output == "proj":
        return backward_feynman_kac_transform(
            coef, w_output, basis, weights, in_domain, guess
        )
    elif output == "coef":
        return coef
    elif output == "full":
        return backward_feynman_kac_intermediate(
            coef, w_output, basis, weights, in_domain, function, guess
        )
    else:
        raise ValueError(f"output must be 'proj', 'coef', or 'full'")


def backward_feynman_kac_matrices(
    w_output,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem=0,
    test_basis=None,
):
    """
    Compute matrices for solving a backward Feynman-Kac problem.

    Parameters
    ----------
    w_output : object
        Intermediate representation from `reweight`.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    function : sequence of (n_frames[i] - 1,) ndarray of float
        Function to integrate. This is defined over *transitions*, not
        frames.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the solution.
        Must have the same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    a : dict of {int : (n_basis, n_basis) ndarray of float}
        Homogeneous terms.
    b : dict of {int : (n_basis, mem + 2) ndarray of float}
        Nonhomogeneous terms.

    """
    if test_basis is None:
        test_basis = basis

    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    a0 = np.zeros((n_basis, n_basis))
    a = {t: np.zeros((n_basis, n_basis)) for t in range(1, mem + 2)}
    b = {t: np.zeros((n_basis, mem + 2)) for t in range(1, mem + 2)}

    for (_, _, u_w), x, y, w, d, f, g in zip_equal(
        w_output, test_basis, basis, weights, in_domain, function, guess
    ):
        n_frames = len(w)
        f = np.broadcast_to(f, n_frames - 1)
        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)
        assert u_w.shape == (n_frames, 1, mem + 2)

        if n_frames <= lag:
            assert np.all(w == 0.0) and np.all(u_w == 0.0)
            continue
        end = n_frames - lag
        assert np.all(w[end:] == 0.0) and np.all(u_w[end:] == 0.0)

        wy = linalg.scale_rows(w, y)

        a0 += _build(d[:end], x[:end], wy[:end])
        for n in range(1, mem + 2):
            t = n * dlag
            k = backward_feynman_kac_transitions(d, f, g, t)
            a[n] += _build(k[:end, 0, 0], x[t : end + t], wy[:end])
            b[n] += _build(k[:end, 1, 0], x[t : end + t], u_w[:end, 0])

    # biorthonormalize x with respect to y
    solve = linalg.factorized(a0)
    a = {t: solve(a[t]) for t in range(1, mem + 2)}
    b = {t: solve(b[t]) for t in range(1, mem + 2)}

    return a, b


def backward_feynman_kac_transform(
    coef, w_output, basis, weights, in_domain, guess
):
    """
    Returns the projected solution of a backward Feynman-Kac problem.

    Parameters
    ----------
    coef : (n_basis, mem + 2) ndarray of float
        DGA coefficients.
    w_output : object
        Intermediate representation from `reweight`.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the projected solution. Note that this is the
        product of the Feynman-Kac statistic and the invariant
        distribution.

    """
    return [
        d * w * (y @ coef[:, 0]) + g * u_w[:, 0, 0]
        for (_, _, u_w), y, w, d, g in zip_equal(
            w_output, basis, weights, in_domain, guess
        )
    ]


def backward_feynman_kac_intermediate(
    coef, w_output, basis, weights, in_domain, function, guess
):
    """
    Returns an intermediate representation of the solution of a backward
    Feynman-Kac problem for use in further calculations.

    Parameters
    ----------
    coef : (n_basis, mem + 2) ndarray of float
        DGA coefficients.
    w_output : object
        Intermediate representation from `reweight`.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    function : sequence of (n_frames[i] - 1,) ndarray of float
        Function to integrate. This is defined over *transitions*, not
        frames.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.

    Returns
    -------
    b_output : object
        Intermediate representation of the solution to a backward
        Feynman-Kac problem.

    """
    n_basis, n_lags = coef.shape
    out = []
    for (_, _, u_w), y, w, d, f, g in zip_equal(
        w_output, basis, weights, in_domain, function, guess
    ):
        n_frames = len(d)
        f = np.broadcast_to(f, n_frames - 1)

        assert u_w.shape == (n_frames, 1, n_lags)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)

        k = backward_feynman_kac_transitions(d, f, g, 1)

        m = np.zeros((n_frames, 2))
        m[:, 0] = d
        m[:, 1] = g

        u = np.empty((n_frames, 2, n_lags))
        u[:, 0] = w * (y @ coef)
        u[:, 1] = u_w[:, 0]

        out.append((k, m, u))
    return out


def integral(b_output, f_output, values, obslag, lag, mem=0):
    """
    Calculate an integral-type statistic.

    Parameters
    ----------
    b_output : object
        Intermediate representation from a backward-in-time statistic.
    f_output : object
        Intermediate representation from a forward-in-time statistic.
    values : list of (n_frames[i]-obslag,) ndarray of float
        Value of the observable at each frame or step.
    obslag : int
        Lag time of the observable. This function currently supports
        0 and 1.
    lag : int
        Total lag time.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.

    Returns
    -------
    float
        Value of the integral.

    """
    assert obslag in [0, 1]
    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)

    out = 0.0
    for (k_b, m_b, u_b), (k_f, m_f, u_f), v in zip_equal(
        b_output, f_output, values
    ):
        if obslag == 0:
            k_v = m_b[:, :, None] * v[:, None, None] * m_f[:, None, :]
            k_v = 0.5 * (k_b @ k_v[1:] + k_v[:-1] @ k_f)
        else:
            k_v = m_b[:-1, :, None] * v[:, None, None] * m_f[1:, None, :]

        for n in range(1, mem + 2):
            t = n * dlag
            k = integral_windows(k_b, k_f, k_v, 1, t)
            a = np.sum(np.transpose(u_b, (0, 2, 1)) @ k @ u_f, axis=0)
            for k in range(mem - n + 2):
                for l in range(mem - n - k + 2):
                    out += a[k, l]
    return out / dlag


def pointwise_integral(b_output, f_output, obslag, lag, mem=0):
    """
    Calculate pointwise coefficients for integral-type statistics.

    Parameters
    ----------
    b_output : object
        Intermediate representation from a backward-in-time statistic.
    f_output : object
        Intermediate representation from a forward-in-time statistic.
    obslag : int
        Lag time of the observable. This function currently supports
        0 and 1.
    lag : int
        Total lag time.
    mem : int, optional
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. By
        default, use `mem=0`, which corresponds to not using memory.

    Returns
    -------
    list of (n_frames[i]-obslag,) ndarray of float
        Pointwise coefficients for integral-type statistics.

    """
    assert obslag in [0, 1]
    assert lag > 0
    assert mem >= 0
    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    out = []
    for (k_b, m_b, u_b), (k_f, m_f, u_f) in zip_equal(b_output, f_output):
        nf, ni, nk = u_b.shape
        _, nj, nl = u_f.shape
        assert m_b.shape == (nf, ni)
        assert m_f.shape == (nf, nj)
        assert u_b.shape == (nf, ni, nk)
        assert u_f.shape == (nf, nj, nl)
        assert k_b.shape == (nf - 1, ni, ni)
        assert k_f.shape == (nf - 1, nj, nj)
        assert nf > lag
        end = nf - lag
        a = np.zeros((nf - 1, ni, nj))
        u_f_sum = np.cumsum(u_f, axis=-1)
        for n in range(mem + 1):
            s = (n + 1) * dlag
            kend = end + s - 1
            # u[t,i,j] = sum_{k+l <= mem-n} u_b[t,i,k] * u_f[t+s,j,l]
            u = np.zeros((end, ni, nj))
            for k in range(min(mem - n, nk - 1) + 1):
                u += (
                    u_b[:end, :, None, k]
                    * u_f_sum[s : end + s, None, :, min(mem - n - k, nl - 1)]
                )
            a[:kend] += integral_coeffs(u, k_b[:kend], k_f[:kend], 1, s)
        if obslag == 0:
            c = np.zeros((nf, ni, nj))
            c[:-1] += a @ np.swapaxes(k_f, 1, 2)
            c[1:] += np.swapaxes(k_b, 1, 2) @ a
            a = 0.5 * c
            a = np.sum(m_b[:, :, None] * a * m_f[:, None, :], axis=(1, 2))
        else:
            a = np.sum(m_b[:-1, :, None] * a * m_f[1:, None, :], axis=(1, 2))
        out.append(a / dlag)
    return out


def _build(w, x, y):
    """Build dense correlation matrices."""
    out = x.T @ linalg.scale_rows(w, y)
    if scipy.sparse.issparse(out):
        out = out.toarray()
    return out


def _dga_mem(a, b, mem):
    """Solve for DGA coefficients."""
    n_basis = a[1].shape[0]
    am = {}
    bm = {}
    lhs = np.identity(n_basis)
    rhs = np.zeros(n_basis)
    for t in range(1, mem + 2):
        assert a[t].shape == (n_basis, n_basis)
        assert b[t].shape == (n_basis, mem + 2)
        am[t] = -(a[t] + sum(a[t - s] @ am[s] for s in range(1, t)))
        bm[t] = -(
            b[t][:, 0]
            + sum(a[t - s] @ bm[s] + b[t - s][:, s] for s in range(1, t))
        )
        lhs += am[t]
        rhs -= bm[t]
    coef = np.empty((n_basis, mem + 2))
    coef[:, 0] = linalg.solve(lhs, rhs)
    for t in range(1, mem + 2):
        coef[:, t] = am[t] @ coef[:, 0] + bm[t]
    return coef
