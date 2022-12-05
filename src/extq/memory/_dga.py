"""DGA with memory estimators for statistics."""

import numpy as np

from .. import linalg
from . import _matrix
from . import _memory

__all__ = [
    "reweight",
    "forward_committor",
    "forward_mfpt",
    "forward_feynman_kac",
    "backward_committor",
    "backward_mfpt",
    "backward_feynman_kac",
    "reweight_integral",
    "forward_committor_integral",
    "forward_mfpt_integral",
    "forward_feynman_kac_integral",
    "backward_committor_integral",
    "backward_mfpt_integral",
    "backward_feynman_kac_integral",
    "tpt_integral",
    "integral",
    "reweight_solve",
    "reweight_transform",
    "forecast_solve",
    "forecast_transform",
    "aftcast_solve",
    "aftcast_transform",
    "integral_solve",
    "forward_coeffs",
    "backward_coeffs",
]


def reweight(basis, weights, lag, mem=0, test=None):
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
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the invariant distribution.

    """
    mats = [
        _matrix.reweight_matrix(basis, weights, t, test=test)
        for t in _memlags(lag, mem)
    ]
    return reweight_solve(mats, basis, weights)


def forward_committor(basis, weights, in_domain, guess, lag, mem=0, test=None):
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
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.

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
        test=test,
    )


def forward_mfpt(basis, weights, in_domain, guess, lag, mem=0, test=None):
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
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.

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
        test=test,
    )


def forward_feynman_kac(
    basis, weights, in_domain, function, guess, lag, mem=0, test=None
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
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the solution.

    """
    mats = [
        _matrix.forward_feynman_kac_matrix(
            basis, weights, in_domain, function, guess, t, test=test
        )
        for t in _memlags(lag, mem)
    ]
    return forecast_solve(mats, basis, in_domain, guess)


def backward_committor(
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    """
    Estimate the backward committor using DGA with memory.

    Parameters
    ----------
    w_basis : sequence of (n_frames[i], n_w_basis) {ndarray, sparse matrix} of float
        Basis for estimating the invariant distribution. The span of
        `w_basis` must *not* contain the constant function.
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
    w_test : sequence of (n_frames[i], n_w_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the invariant
        distribution. Must have the same dimension as `w_basis`. If
        `None`, use `w_basis`.
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the committor.
        Must have the same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the committor.

    """
    return backward_feynman_kac(
        w_basis,
        basis,
        weights,
        in_domain,
        np.zeros(len(weights)),
        guess,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )


def backward_mfpt(
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    """
    Estimate the backward mean first passage time (MFPT) using DGA
    with memory.

    Parameters
    ----------
    w_basis : sequence of (n_frames[i], n_w_basis) {ndarray, sparse matrix} of float
        Basis for estimating the invariant distribution. The span of
        `w_basis` must *not* contain the constant function.
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
    w_test : sequence of (n_frames[i], n_w_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the invariant
        distribution. Must have the same dimension as `w_basis`. If
        `None`, use `w_basis`.
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the MFPT. Must
        have the same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the MFPT.

    """
    return backward_feynman_kac(
        w_basis,
        basis,
        weights,
        in_domain,
        np.ones(len(weights)),
        guess,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )


def backward_feynman_kac(
    w_basis,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    """
    Estimate the solution to a backward Feynman-Kac problem using DGA
    with memory.

    Parameters
    ----------
    w_basis : sequence of (n_frames[i], n_w_basis) {ndarray, sparse matrix} of float
        Basis for estimating the invariant distribution. The span of
        `w_basis` must *not* contain the constant function.
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
    w_test : sequence of (n_frames[i], n_w_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the invariant
        distribution. Must have the same dimension as `w_basis`. If
        `None`, use `w_basis`.
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the solution.
        Must have the same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the solution.

    """
    mats = [
        _matrix.backward_feynman_kac_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            function,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
        for t in _memlags(lag, mem)
    ]
    return aftcast_solve(mats, w_basis, basis, in_domain, guess)


def reweight_integral(basis, weights, values, lag, mem=0, test=None):
    b_mats = [
        _matrix.reweight_matrix(basis, weights, t, test=test)
        for t in _memlags(lag, mem)
    ]
    f_mats = [_matrix.constant_matrix(weights, t) for t in _memlags(lag, mem)]
    v_mats = [
        _matrix.reweight_integral_matrix(basis, weights, values, t)
        for t in _memlags(lag, mem)
    ]
    return integral_solve(b_mats, f_mats, v_mats, lag, mem)


def forward_committor_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    return forward_feynman_kac_integral(
        w_basis,
        basis,
        weights,
        in_domain,
        values,
        np.zeros(len(weights)),
        guess,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )


def forward_mfpt_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    return forward_feynman_kac_integral(
        w_basis,
        basis,
        weights,
        in_domain,
        values,
        np.ones(len(weights)),
        guess,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )


def forward_feynman_kac_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    function,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    b_mats = [
        _matrix.reweight_matrix(w_basis, weights, t, test=w_test)
        for t in _memlags(lag, mem)
    ]
    f_mats = [
        _matrix.forward_feynman_kac_matrix(
            basis, weights, in_domain, function, guess, t, test=test
        )
        for t in _memlags(lag, mem)
    ]
    v_mats = [
        _matrix.forward_feynman_kac_integral_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            values,
            function,
            guess,
            t,
        )
        for t in _memlags(lag, mem)
    ]
    return integral_solve(b_mats, f_mats, v_mats, lag, mem)


def backward_committor_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    return backward_feynman_kac_integral(
        w_basis,
        basis,
        weights,
        in_domain,
        values,
        np.zeros(len(weights)),
        guess,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )


def backward_mfpt_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    return backward_feynman_kac_integral(
        w_basis,
        basis,
        weights,
        in_domain,
        values,
        np.ones(len(weights)),
        guess,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )


def backward_feynman_kac_integral(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    function,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    b_mats = [
        _matrix.backward_feynman_kac_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            function,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
        for t in _memlags(lag, mem)
    ]
    f_mats = [_matrix.constant_matrix(weights, t) for t in _memlags(lag, mem)]
    v_mats = [
        _matrix.backward_feynman_kac_integral_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            values,
            function,
            guess,
            t,
        )
        for t in _memlags(lag, mem)
    ]
    return integral_solve(b_mats, f_mats, v_mats, lag, mem)


def tpt_integral(
    w_basis,
    b_basis,
    f_basis,
    weights,
    in_domain,
    values,
    b_guess,
    f_guess,
    lag,
    mem=0,
    w_test=None,
    b_test=None,
    f_test=None,
):
    return integral(
        w_basis,
        b_basis,
        f_basis,
        weights,
        in_domain,
        in_domain,
        values,
        np.zeros(len(weights)),
        np.zeros(len(weights)),
        b_guess,
        f_guess,
        lag,
        mem=mem,
        w_test=w_test,
        b_test=b_test,
        f_test=f_test,
    )


def integral(
    w_basis,
    b_basis,
    f_basis,
    weights,
    b_domain,
    f_domain,
    values,
    b_function,
    f_function,
    b_guess,
    f_guess,
    lag,
    mem=0,
    w_test=None,
    b_test=None,
    f_test=None,
):
    b_mats = [
        _matrix.backward_feynman_kac_matrix(
            w_basis,
            b_basis,
            weights,
            b_domain,
            b_function,
            b_guess,
            t,
            w_test=w_test,
            test=b_test,
        )
        for t in _memlags(lag, mem)
    ]
    f_mats = [
        _matrix.forward_feynman_kac_matrix(
            f_basis, weights, f_domain, f_function, f_guess, t, test=f_test
        )
        for t in _memlags(lag, mem)
    ]
    v_mats = [
        _matrix.integral_matrix(
            w_basis,
            b_basis,
            f_basis,
            weights,
            b_domain,
            f_domain,
            values,
            b_function,
            f_function,
            b_guess,
            f_guess,
            t,
        )
        for t in _memlags(lag, mem)
    ]
    return integral_solve(b_mats, f_mats, v_mats, lag, mem)


def reweight_solve(mats, basis, weights):
    """
    Compute the invariant distribution using correlation matrices.

    Parameters
    ----------
    mats : sequence of (1 + n_basis, 1 + n_basis) {ndarray, sparse matrix} of float
        Sequence of correlation matrices at equally-spaced lag times,
        starting at a lag time of zero.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for the projected invariant distribution. The span of
        `basis` must *not* contain the constant function.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Projected invariant distribution.

    """
    coeffs = backward_coeffs(mats)
    return reweight_transform(coeffs, basis, weights)


def reweight_transform(coeffs, basis, weights):
    """
    Compute the invariant distribution using projection coefficients.

    Parameters
    ----------
    coeffs : (1 + n_basis,) ndarray of float
        Projection coefficients.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for the projected invariant distribution. The span of
        `basis` must *not* contain the constant function.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Projected invariant distribution.

    """
    result = []
    for x_w, w in zip(basis, weights):
        result.append(w * (coeffs[0] + x_w @ coeffs[1:]))
    return result


def forecast_solve(mats, basis, in_domain, guess):
    """
    Compute the projected forecast using correlation matrices.

    Parameters
    ----------
    mats : sequence of (n_basis + 1, n_basis + 1) {ndarray, sparse matrix} of float
        Sequence of correlation matrices at equally-spaced lag times,
        starting at a lag time of zero.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for the projected forecast. Must be zero outside of the
        domain.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the projected forecast. Must satisfy boundary
        conditions.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Projected forecast.

    """
    coeffs = forward_coeffs(mats)
    return forecast_transform(coeffs, basis, in_domain, guess)


def forecast_transform(coeffs, basis, in_domain, guess):
    """
    Compute the projected forecast using projection coefficients.

    Parameters
    ----------
    coeffs : (n_basis + 1,) ndarray of float
        Projection coefficients.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for the projected forecast. Must be zero outside of the
        domain.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the projected forecast. Must satisfy boundary
        conditions.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Projected forecast.

    """
    result = []
    for y_f, d_f, g_f in zip(basis, in_domain, guess):
        result.append(g_f + np.where(d_f, y_f @ coeffs[:-1], 0.0) / coeffs[-1])
    return result


def aftcast_solve(mats, w_basis, basis, in_domain, guess):
    """
    Compute the projected aftcast using correlation matrices.

    Parameters
    ----------
    mats : sequence of (1 + n_w_basis + n_basis, 1 + n_w_basis + n_basis) {ndarray, sparse matrix} of float
        Sequence of correlation matrices at equally-spaced lag times,
        starting at a lag time of zero.
    w_basis : sequence of (n_frames[i], n_w_basis) {ndarray, sparse matrix} of float
        Basis for the projected invariant distribution. The span of
        `w_basis` must *not* contain the constant function.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for the projected aftcast. Must be zero outside of the
        domain.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the projected aftcast. Must satisfy boundary
        conditions.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Projected aftcast.

    """
    coeffs = backward_coeffs(mats)
    return aftcast_transform(coeffs, w_basis, basis, in_domain, guess)


def aftcast_transform(coeffs, w_basis, basis, in_domain, guess):
    """
    Compute the projected aftcast using projection coefficients.

    Parameters
    ----------
    coeffs : (1 + n_w_basis + n_basis,) ndarray of float
        Projection coefficients.
    w_basis : sequence of (n_frames[i], n_w_basis) {ndarray, sparse matrix} of float
        Basis for the projected invariant distribution. The span of
        `w_basis` must *not* contain the constant function.
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for the projected aftcast. Must be zero outside of the
        domain.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the projected aftcast. Must satisfy boundary
        conditions.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Projected aftcast.

    """
    result = []
    for x_w, x_b, d_b, g_b in zip(w_basis, basis, in_domain, guess):
        n = x_w.shape[1] + 1
        com = coeffs[0] + x_w @ coeffs[1:n]
        result.append(g_b + np.where(d_b, x_b @ coeffs[n:], 0.0) / com)
    return result


def integral_solve(b_mats, f_mats, v_mats, lag, mem):
    """
    Compute an ergodic average from correlation matrices.

    Parameters
    ----------
    b_mats : sequence of (n_b_basis, n_b_basis) {ndarray, sparse matrix} of float
        Sequence of correlation matrices for the backward-in-time
        statistic at lag times `numpy.linspace(0, lag, mem+2)`.
    f_mats : sequence of (n_f_basis, n_f_basis) {ndarray, sparse matrix} of float
        Sequence of correlation matrices for the forward-in-time
        statistic at lag times `numpy.linspace(0, lag, mem+2)`.
    v_mats : sequence of (n_b_basis, n_f_basis) {ndarray, sparse matrix} of float
        Sequence of correlation matrices for the integral at lag times
        `numpy.linspace(0, lag, mem+2)`.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32.
        `mem=0` corresponds to not using memory.

    Returns
    -------
    float
        Ergodic average.

    """
    assert lag % (mem + 1) == 0
    assert len(b_mats) == mem + 2
    assert len(f_mats) == mem + 2
    assert len(v_mats) == mem + 2
    n_b, n_f = v_mats[0].shape

    mats = []
    for b_mat, f_mat, v_mat in zip(b_mats, f_mats, v_mats):
        mat = np.zeros((n_b + n_f, n_b + n_f))
        mat[:n_b, :n_b] = b_mat
        mat[n_b:, n_b:] = f_mat
        mat[:n_b, n_b:] = v_mat
        mats.append(mat)

    mems = _memory.memory(mats)
    b_mems = [m[:n_b, :n_b] for m in mems]
    f_mems = [m[n_b:, n_b:] for m in mems]
    v_mems = [m[:n_b, n_b:] for m in mems]

    b_coeffs = backward_coeffs(b_mats, b_mems)
    f_coeffs = forward_coeffs(f_mats, f_mems)
    gen = v_mats[1] - v_mats[0] + sum(v_mems)
    return (b_coeffs @ gen @ f_coeffs) / (lag // (mem + 1))


def forward_coeffs(mats, mems=None):
    """
    Solve a forward-in-time problem for projection coefficients.

    Parameters
    ----------
    mats : sequence of (n_basis, n_basis) {ndarray, sparse matrix} of float
        Sequence of correlation matrices at equally-spaced lag times,
        starting at a lag time of zero.
    mems : sequence of (n_basis, n_basis) {ndarray, sparse matrix} of float, optional
        Sequence of memory matrices at equally-spaced lag times (with
        the same spacing as `mats`), starting at a lag time of zero.
        Must satisfy ``len(mems) = len(mats) - 2``. If `None`, compute
        `mems` from `mats`.

    Returns
    -------
    (n_basis,) ndarray of float
        Projection coefficients.

    """
    if mems is None:
        mems = _memory.memory(mats)
    assert len(mats) == len(mems) + 2
    gen = mats[1] - mats[0] + sum(mems)
    return np.concatenate([linalg.solve(gen[:-1, :-1], -gen[:-1, -1]), [1.0]])


def backward_coeffs(mats, mems=None):
    """
    Solve a backward-in-time problem for projection coefficients.

    Parameters
    ----------
    mats : sequence of (n_basis, n_basis) {ndarray, sparse matrix} of float
        Sequence of correlation matrices at equally-spaced lag times,
        starting at a lag time of zero.
    mems : sequence of (n_basis, n_basis) {ndarray, sparse matrix} of float, optional
        Sequence of memory matrices at equally-spaced lag times (with
        the same spacing as `mats`), starting at a lag time of zero.
        Must satisfy ``len(mems) = len(mats) - 2``. If `None`, compute
        `mems` from `mats`.

    Returns
    -------
    (n_basis,) ndarray of float
        Projection coefficients.

    """
    if mems is None:
        mems = _memory.memory(mats)
    assert len(mats) == len(mems) + 2
    gen = linalg.solve(mats[0], mats[1] - mats[0] + sum(mems))
    return linalg.solve(
        mats[0].T,
        np.concatenate([[1.0], linalg.solve(gen.T[1:, 1:], -gen.T[1:, 0])]),
    )


def _memlags(lag, mem):
    """
    Return the lag times at which to evaluate correlation matrices.

    This function acts similarly to `numpy.linspace(0, lag, mem+2)`.

    Parameters
    ----------
    lag : int
        Maximum lag time.
    mem : int
        Number of memory matrices, which are evaluated at equally spaced
        times between time 0 and time `lag`. `mem+1` must evenly divide
        `lag`. For example, with a `lag=32`, `mem=3` and `mem=7` are
        fine since 7+1=8 and 3+1=4 evenly divide 32.

    Returns
    -------
    ndarray of int
        Lag times at which to evaluate correlation matrices.

    """
    assert lag % (mem + 1) == 0
    return np.arange(0, lag + 1, lag // (mem + 1))
