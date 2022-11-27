"""DGA with memory estimators for statistics."""

import numpy as np

from .. import linalg
from . import _matrix
from . import _memory


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
    mats = []
    for t in _memlags(lag, mem):
        mats.append(_matrix.reweight_matrix(basis, weights, t, test=test))
    return _reweight(mats, basis, weights)


def _reweight(mats, basis, weights):
    """
    Compute the invariant distribution using correlation matrices.

    Parameters
    ----------
    mats : sequence of (1 + n_basis, 1 + n_basis) ndarray of float
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
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
    coeffs = reweight_solve(gen)
    return reweight_transform(coeffs, basis, weights)


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
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_committor_matrix(
                basis, weights, in_domain, guess, t, test=test
            )
        )
    return _forward(mats, basis, in_domain, guess)


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
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_mfpt_matrix(
                basis, weights, in_domain, guess, t, test=test
            )
        )
    return _forward(mats, basis, in_domain, guess)


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
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_feynman_kac_matrix(
                basis, weights, in_domain, function, guess, t, test=test
            )
        )
    return _forward(mats, basis, in_domain, guess)


def _forward(mats, basis, in_domain, guess):
    """
    Compute the projected forecast using correlation matrices.

    Parameters
    ----------
    mats : sequence of (n_basis + 1, n_basis + 1) ndarray of float
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
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
    coeffs = forward_solve(gen)
    return forward_transform(coeffs, basis, in_domain, guess)


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
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.backward_committor_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _backward(mats, w_basis, basis, in_domain, guess)


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
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.backward_mfpt_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _backward(mats, w_basis, basis, in_domain, guess)


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
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
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
        )
    return _backward(mats, w_basis, basis, in_domain, guess)


def _backward(mats, w_basis, basis, in_domain, guess):
    """
    Compute the projected aftcast using correlation matrices.

    Parameters
    ----------
    mats : sequence of (1 + n_w_basis + n_basis, 1 + n_w_basis + n_basis) ndarray of float
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
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
    coeffs = backward_solve(gen)
    return backward_transform(coeffs, w_basis, basis, in_domain, guess)


def reweight_integral(basis, weights, values, lag, mem=0, test=None):
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.reweight_integral_matrix(
                basis, weights, values, t, test=test
            )
        )
    return _integral(mats, lag, mem)


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
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_committor_integral_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                values,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _integral(mats, lag, mem)


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
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_mfpt_integral_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                values,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _integral(mats, lag, mem)


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
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.forward_feynman_kac_integral_matrix(
                w_basis,
                basis,
                weights,
                in_domain,
                values,
                function,
                guess,
                t,
                w_test=w_test,
                test=test,
            )
        )
    return _integral(mats, lag, mem)


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
    mats = []
    for t in _memlags(lag, mem):
        _matrix.backward_committor_integral_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            values,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
    return _integral(mats, lag, mem)


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
    mats = []
    for t in _memlags(lag, mem):
        _matrix.backward_mfpt_integral_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            values,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
    return _integral(mats, lag, mem)


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
    mats = []
    for t in _memlags(lag, mem):
        _matrix.backward_feynman_kac_integral_matrix(
            w_basis,
            basis,
            weights,
            in_domain,
            values,
            function,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
    return _integral(mats, lag, mem)


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
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
            _matrix.tpt_integral_matrix(
                w_basis,
                b_basis,
                f_basis,
                weights,
                in_domain,
                values,
                b_guess,
                f_guess,
                t,
                w_test=w_test,
                b_test=b_test,
                f_test=f_test,
            )
        )
    return _integral(mats, lag, mem)


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
    mats = []
    for t in _memlags(lag, mem):
        mats.append(
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
                w_test=w_test,
                b_test=b_test,
                f_test=f_test,
            )
        )
    return _integral(mats, lag, mem)


def _integral(mats, lag, mem):
    """
    Compute an ergodic average from correlation matrices.

    Parameters
    ----------
    mats : sequence of (n_basis, n_basis) {ndarray, sparse matrix} of float
        Sequence of correlation matrices at lag times
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
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
    return integral_solve(gen, mats[0]) / (lag // (mem + 1))


def reweight_solve(gen):
    """
    Solve a reweighting problem for projection coefficients.

    Parameters
    ----------
    gen : (n_basis, n_basis) {ndarray, sparse matrix} of float
        Memory-corrected generator matrix.

    Returns
    -------
    (n_basis,) ndarray of float
        Projection coefficients.

    """
    return np.concatenate([[1.0], linalg.solve(gen.T[1:, 1:], -gen.T[1:, 0])])


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


def forward_solve(gen):
    """
    Solve a forward-in-time problem for projection coefficients.

    Parameters
    ----------
    gen : (n_basis, n_basis) {ndarray, sparse matrix} of float
        Memory-corrected generator matrix.

    Returns
    -------
    (n_basis,) ndarray of float
        Projection coefficients.

    """
    return np.concatenate([linalg.solve(gen[:-1, :-1], -gen[:-1, -1]), [1.0]])


def forward_transform(coeffs, basis, in_domain, guess):
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


def backward_solve(gen):
    """
    Solve a backward-in-time problem for projection coefficients.

    Parameters
    ----------
    gen : (n_basis, n_basis) {ndarray, sparse matrix} of float
        Memory-corrected generator matrix.

    Returns
    -------
    (n_basis,) ndarray of float
        Projection coefficients.

    """
    return np.concatenate([[1.0], linalg.solve(gen.T[1:, 1:], -gen.T[1:, 0])])


def backward_transform(coeffs, w_basis, basis, in_domain, guess):
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


def integral_solve(gen, eye):
    """
    Solve a problem for an ergodic average.

    Parameters
    ----------
    gen : (n_basis, n_basis) {ndarray, sparse matrix} of float
        Memory-corrected generator matrix.
    eye : (n_basis, n_basis) {ndarray, sparse matrix} of float
        Zero lag time correlation matrix.

    Returns
    -------
    float
        Ergodic average.

    """
    mat = linalg.solve(eye, gen)
    forward_coeffs = np.concatenate(
        [linalg.solve(mat[1:-1, 1:-1], -mat[1:-1, -1]), [1.0]]
    )
    backward_coeffs = np.concatenate(
        [[1.0], linalg.solve(mat.T[1:-1, 1:-1], -mat.T[1:-1, 0])]
    )
    return backward_coeffs @ mat[:-1, 1:] @ forward_coeffs


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
