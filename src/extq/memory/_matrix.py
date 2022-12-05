"""Correlation matrices for DGA."""

import numpy as np

from . import _kernel
from ._tlcc import wtlcc_dense as _build

__all__ = [
    "constant_matrix",
    "reweight_matrix",
    "forward_committor_matrix",
    "forward_mfpt_matrix",
    "forward_feynman_kac_matrix",
    "backward_committor_matrix",
    "backward_mfpt_matrix",
    "backward_feynman_kac_matrix",
    "reweight_integral_matrix",
    "forward_committor_integral_matrix",
    "forward_mfpt_integral_matrix",
    "forward_feynman_kac_integral_matrix",
    "backward_committor_integral_matrix",
    "backward_mfpt_integral_matrix",
    "backward_feynman_kac_integral_matrix",
    "tpt_integral_matrix",
    "integral_matrix",
]


def constant_matrix(weights, lag):
    """
    Compute the correlation matrix for the constant function.

    Parameters
    ----------
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    lag : int
        Lag time in units of frames.

    Returns
    -------
    (1, 1) ndarray of float
        Correlation matrix.

    """
    mat = None
    for w in weights:
        mat = _constant_matrix(w, lag, mat)
    return _bmat(mat)


def reweight_matrix(basis, weights, lag, test=None):
    """
    Compute the correlation matrix for the invariant distribution.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the invariant distribution. The span of
        `basis` must *not* contain the constant function.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    lag : int
        Lag time in units of frames.
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    (1 + n_basis, 1 + n_basis) ndarray of float
        Correlation matrix.

    """
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, w in zip(basis, test, weights):
        mat = _reweight_matrix(x_w, y_w, w, lag, mat)
    return _bmat(mat)


def forward_committor_matrix(basis, weights, in_domain, guess, lag, test=None):
    """
    Compute the correlation matrix for the forward committor.

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
        Lag time in units of frames.
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    (n_basis + 1, n_basis + 1) ndarray of float
        Correlation matrix.

    """
    return forward_feynman_kac_matrix(
        basis,
        weights,
        in_domain,
        np.zeros(len(weights)),
        guess,
        lag,
        test=test,
    )


def forward_mfpt_matrix(basis, weights, in_domain, guess, lag, test=None):
    """
    Compute the correlation matrix for the forward mean first passage
    time (MFPT).

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
        Lag time in units of frames.
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    (n_basis + 1, n_basis + 1) ndarray of float
        Correlation matrix.

    """
    return forward_feynman_kac_matrix(
        basis,
        weights,
        in_domain,
        np.ones(len(weights)),
        guess,
        lag,
        test=test,
    )


def forward_feynman_kac_matrix(
    basis, weights, in_domain, function, guess, lag, test=None
):
    """
    Compute the correlation matrix for the forward Feynman-Kac problem.

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
        Lag time in units of frames.
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    (n_basis + 1, n_basis + 1) ndarray of float
        Correlation matrix.

    """
    if test is None:
        test = basis
    mat = None
    for x_f, y_f, w, in_d, f, g in zip(
        test, basis, weights, in_domain, function, guess
    ):
        mat = _forward_matrix(x_f, y_f, w, in_d, f, g, lag, mat)
    return _bmat(mat)


def backward_committor_matrix(
    w_basis, basis, weights, in_domain, guess, lag, w_test=None, test=None
):
    """
    Compute the correlation matrix for the backward committor.

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
        Lag time in units of frames.
    w_test : sequence of (n_frames[i], n_w_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the invariant
        distribution. Must have the same dimension as `w_basis`. If
        `None`, use `w_basis`.
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the committor.
        Must have the same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    (1 + n_w_basis + n_basis, 1 + n_w_basis + n_basis) ndarray of float
        Correlation matrix.

    """
    return backward_feynman_kac_matrix(
        w_basis,
        basis,
        weights,
        in_domain,
        np.zeros(len(weights)),
        guess,
        lag,
        w_test=w_test,
        test=test,
    )


def backward_mfpt_matrix(
    w_basis, basis, weights, in_domain, guess, lag, w_test=None, test=None
):
    """
    Compute the correlation matrix for the backward mean first passage
    time (MFPT).

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
        Lag time in units of frames.
    w_test : sequence of (n_frames[i], n_w_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the invariant
        distribution. Must have the same dimension as `w_basis`. If
        `None`, use `w_basis`.
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the MFPT. Must
        have the same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    (1 + n_w_basis + n_basis, 1 + n_w_basis + n_basis) ndarray of float
        Correlation matrix.

    """
    return backward_feynman_kac_matrix(
        w_basis,
        basis,
        weights,
        in_domain,
        np.ones(len(weights)),
        guess,
        lag,
        w_test=w_test,
        test=test,
    )


def backward_feynman_kac_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    w_test=None,
    test=None,
):
    """
    Compute the correlation matrix for the backward Feynman-Kac problem.

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
        Lag time in units of frames.
    w_test : sequence of (n_frames[i], n_w_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the invariant
        distribution. Must have the same dimension as `w_basis`. If
        `None`, use `w_basis`.
    test : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the solution.
        Must have the same dimension as `basis`. If `None`, use `basis`.

    Returns
    -------
    (1 + n_w_basis + n_basis, 1 + n_w_basis + n_basis) ndarray of float
        Correlation matrix.

    """
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, f, g in zip(
        w_basis, w_test, basis, test, weights, in_domain, function, guess
    ):
        mat = _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, f, g, lag, mat)
    return _bmat(mat)


def reweight_integral_matrix(basis, weights, values, lag):
    mat = None
    for x_w, w, v in zip(basis, weights, values):
        mat = _reweight_integral_matrix(x_w, w, v, lag, mat)
    return _bmat(mat)


def forward_committor_integral_matrix(
    w_basis, basis, weights, in_domain, values, guess, lag
):
    return forward_feynman_kac_integral_matrix(
        w_basis,
        basis,
        weights,
        in_domain,
        values,
        np.zeros(len(weights)),
        guess,
        lag,
    )


def forward_mfpt_integral_matrix(
    w_basis, basis, weights, in_domain, values, guess, lag
):
    return forward_feynman_kac_integral_matrix(
        w_basis,
        basis,
        weights,
        in_domain,
        values,
        np.ones(len(weights)),
        guess,
        lag,
    )


def forward_feynman_kac_integral_matrix(
    w_basis, basis, weights, in_domain, values, function, guess, lag
):
    mat = None
    for x_w, y_f, w, in_d, v, f, g in zip(
        w_basis, basis, weights, in_domain, values, function, guess
    ):
        mat = _forward_integral_matrix(x_w, y_f, w, in_d, v, f, g, lag, mat)
    return _bmat(mat)


def backward_committor_integral_matrix(
    w_basis, basis, weights, in_domain, values, guess, lag
):
    return backward_feynman_kac_integral_matrix(
        w_basis,
        basis,
        weights,
        in_domain,
        values,
        np.zeros(len(weights)),
        guess,
        lag,
    )


def backward_mfpt_integral_matrix(
    w_basis, basis, weights, in_domain, values, guess, lag
):
    return backward_feynman_kac_integral_matrix(
        w_basis,
        basis,
        weights,
        in_domain,
        values,
        np.ones(len(weights)),
        guess,
        lag,
    )


def backward_feynman_kac_integral_matrix(
    w_basis, basis, weights, in_domain, values, function, guess, lag
):
    mat = None
    for x_w, x_b, w, in_d, v, f, g in zip(
        w_basis, basis, weights, in_domain, function, values, guess
    ):
        mat = _backward_integral_matrix(x_w, x_b, w, in_d, v, f, g, lag, mat)
    return _bmat(mat)


def tpt_integral_matrix(
    w_basis,
    b_basis,
    f_basis,
    weights,
    in_domain,
    values,
    b_guess,
    f_guess,
    lag,
):
    return integral_matrix(
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
    )


def integral_matrix(
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
):
    mat = None
    for (x_w, x_b, y_f, w, d_b, d_f, v, f_b, f_f, g_b, g_f,) in zip(
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
    ):
        mat = _integral_matrix(
            x_w, x_b, y_f, w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag, mat
        )
    return _bmat(mat)


def _constant_matrix(w, lag, mat):
    """
    Compute the correlation matrix for the constant function from a
    single trajectory

    Parameters
    ----------
    w : (n_frames,) ndarray of float
        Weight of each frame. The last `lag` frames must be zero.
    lag : int
        Lag time in units of frames.
    mat : (1, 1) ndarray of object or None
        Matrix in which to store the result. If `None`, create a new
        matrix.

    Returns
    -------
    (1, 1) ndarray of object
        Correlation matrix.

    """
    m = _kernel.reweight_kernel(w, lag)
    if mat is None:
        mat = np.full((1, 1), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    # fmt: on
    return mat


def _reweight_matrix(x_w, y_w, w, lag, mat):
    """
    Compute the correlation matrix for the invariant distribution from a
    single trajectory

    Parameters
    ----------
    x_w : (n_frames, n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the invariant distribution. The span of
        `basis` must *not* contain the constant function.
    y_w : (n_frames, n_basis) {ndarray, sparse matrix} of float
        Test basis against which to minimize the error.
    w : (n_frames,) ndarray of float
        Weight of each frame. The last `lag` frames must be zero.
    lag : int
        Lag time in units of frames.
    mat : (2, 2) ndarray of object or None
        Matrix in which to store the result. If `None`, create a new
        matrix.

    Returns
    -------
    (2, 2) ndarray of object
        Correlation matrix.

    """
    m = _kernel.reweight_kernel(w, lag)
    if mat is None:
        mat = np.full((2, 2), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    # fmt: on
    return mat


def _forward_matrix(x_f, y_f, w, d_f, f_f, g_f, lag, mat):
    """
    Compute the correlation matrix for forecasts.

    Parameters
    ----------
    x_f : (n_frames, n_basis) {ndarray, sparse matrix} of float
        Test basis against which to minimize the error.
    y_f : (n_frames, n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    w : (n_frames,) ndarray of float
        Weight of each frame. The last `lag` frames must be zero.
    d_f : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    f_f : (n_frames - 1,) ndarray of float
        Function to integrate. This is defined over *transitions*, not
        frames.
    g_f : (n_frames,) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Lag time in units of frames.
    mat : (2, 2) ndarray of object or None
        Matrix in which to store the result. If `None`, create a new
        matrix.

    Returns
    -------
    (2, 2) ndarray of object
        Correlation matrix.

    """
    f_f = np.broadcast_to(f_f, len(w) - 1)
    m = _kernel.forward_kernel(w, d_f, f_f, g_f, lag)
    if mat is None:
        mat = np.full((2, 2), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], x_f , y_f , mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 1], x_f , None, mat[0, 1], lag)
    mat[1, 1] = _build(m[1, 1], None, None, mat[1, 1], lag)
    # fmt: on
    return mat


def _backward_matrix(x_w, y_w, x_b, y_b, w, d_b, f_b, g_b, lag, mat):
    """
    Compute the correlation matrix for aftcasts.

    Parameters
    ----------
    x_w : (n_frames, n_w_basis) {ndarray, sparse matrix} of float
        Basis for estimating the invariant distribution. The span of
        `basis` must *not* contain the constant function.
    y_w : (n_frames, n_w_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the invariant
        distribution.
    x_b : (n_frames, n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    y_b : (n_frames, n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error of the solution.
    w : (n_frames,) ndarray of float
        Weight of each frame. The last `lag` frames must be zero.
    d_b : (n_frames,) ndarray of bool
        Whether each frame is in the domain.
    f_b : (n_frames - 1,) ndarray of float
        Function to integrate. This is defined over *transitions*, not
        frames.
    g_b : (n_frames,) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Lag time in units of frames.
    mat : (3, 3) ndarray of object or None
        Matrix in which to store the result. If `None`, create a new
        matrix.

    Returns
    -------
    (3, 3) ndarray of object
        Correlation matrix.

    """
    f_b = np.broadcast_to(f_b, len(w) - 1)
    m = _kernel.backward_kernel(w, d_b, f_b, g_b, lag)
    if mat is None:
        mat = np.full((3, 3), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[0, 2] = _build(m[0, 1], None, y_b , mat[0, 2], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    mat[1, 2] = _build(m[0, 1], x_w , y_b , mat[1, 2], lag)
    mat[2, 2] = _build(m[1, 1], x_b , y_b , mat[2, 2], lag)
    # fmt: on
    return mat


def _reweight_integral_matrix(x_w, w, v, lag, mat):
    m = _kernel.reweight_integral_kernel(w, v, lag)
    if mat is None:
        mat = np.full((2, 1), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    # fmt: on
    return mat


def _forward_integral_matrix(x_w, y_f, w, d_f, v, f_f, g_f, lag, mat):
    f_f = np.broadcast_to(f_f, len(w) - 1)
    m = _kernel.forward_integral_kernel(w, d_f, v, f_f, g_f, lag)
    if mat is None:
        mat = np.full((2, 2), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], None, y_f , mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 1], None, None, mat[0, 1], lag)
    mat[1, 0] = _build(m[0, 0], x_w , y_f , mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 1], x_w , None, mat[1, 1], lag)
    # fmt: on
    return mat


def _backward_integral_matrix(x_w, x_b, w, d_b, v, f_b, g_b, lag, mat):
    f_b = np.broadcast_to(f_b, len(w) - 1)
    m = _kernel.backward_integral_kernel(w, d_b, v, f_b, g_b, lag)
    if mat is None:
        mat = np.full((3, 1), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[2, 0] = _build(m[1, 0], x_b , None, mat[2, 0], lag)
    # fmt: on
    return mat


def _integral_matrix(
    x_w, x_b, y_f, w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag, mat
):
    f_b = np.broadcast_to(f_b, len(w) - 1)
    f_f = np.broadcast_to(f_f, len(w) - 1)
    m = _kernel.integral_kernel(w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag)
    if mat is None:
        mat = np.full((3, 2), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], None, y_f , mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 1], None, None, mat[0, 1], lag)
    mat[1, 0] = _build(m[0, 0], x_w , y_f , mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 1], x_w , None, mat[1, 1], lag)
    mat[2, 0] = _build(m[1, 0], x_b , y_f , mat[2, 0], lag)
    mat[2, 1] = _build(m[1, 1], x_b , None, mat[2, 1], lag)
    # fmt: on
    return mat


def _bmat(blocks):
    """
    Build a matrix from blocks.

    Parameters
    ----------
    blocks : 2D ndarray of {(n_rows[i], n_cols[j]) ndarray, None}
        Array of blocks, which must have conformable dimensions. An
        entry `None` is interpreted as a matrix of zeros.

    Returns
    -------
    (sum(n_rows), sum(n_cols)) ndarray of float
        Matrix built from the blocks.

    """
    s0, s1 = _bshape(blocks)
    si = np.cumsum(np.concatenate([[0], s0]))
    sj = np.cumsum(np.concatenate([[0], s1]))
    mat = np.zeros((si[-1], sj[-1]))
    for i in range(len(s0)):
        for j in range(len(s1)):
            if blocks[i, j] is not None:
                mat[si[i] : si[i + 1], sj[j] : sj[j + 1]] = blocks[i, j]
    return mat


def _bshape(blocks):
    """
    Obtain the dimensions of the blocks in an array of blocks.

    Parameters
    ----------
    blocks : 2D ndarray of {(n_rows[i], n_cols[j]) ndarray, None}
        Array of blocks, which must have conformable dimensions. An
        entry `None` is interpreted as a matrix of zeros.

    Returns
    -------
    n_rows : tuple of int
        Row dimensions of the blocks. `blocks[i, j]` has `n_rows[i]`
        rows.
    n_cols : tuple of int
        Column dimensions of the blocks. `blocks[i, j]` has n_cols[j]`
        columns.

    """
    br, bc = blocks.shape
    rows = [None] * br
    cols = [None] * bc
    for i in range(br):
        for j in range(bc):
            if blocks[i, j] is not None:
                r, c = blocks[i, j].shape
                if rows[i] is None:
                    rows[i] = r
                if cols[j] is None:
                    cols[j] = c
                assert (rows[i], cols[j]) == (r, c)
    for r in rows:
        assert r is not None
    for c in cols:
        assert c is not None
    return (tuple(rows), tuple(cols))
