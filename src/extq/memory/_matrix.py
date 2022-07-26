import numpy as np

from . import _kernel
from ._tlcc import wtlcc_dense as _build


def reweight_matrix(basis, weights, lag, test=None):
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, w in zip(basis, test, weights):
        mat = _reweight_matrix(x_w, y_w, w, lag, mat)
    return _bmat(mat)


def forward_committor_matrix(basis, weights, in_domain, guess, lag, test=None):
    """Compute correlation matrix for forward committors.

    Parameters
    ----------
    basis : array-like (n_trajs,) of ndarray (n_frames, n_basis) of float
        Basis functions; must satisfy boundary conditions
    weights, guess : array-like (n_trajs,) of ndarray (n_frames,) of float
        Weight, guess for committor at each frame in trajectories
    in_domain : array-like (n_trajs,) of ndarray (n_frames,) of bool
        Whether each frame is in the domain (A U B)^c
    test : optional
        Test basis functions; if not specified, will use the same as the right basis

    Returns
    -------
    ndarray (n_basis + 1, n_basis + 1) of float
    """
    if test is None:
        test = basis
    mat = None
    for x_f, y_f, w, in_d, g in zip(test, basis, weights, in_domain, guess):
        mat = _forward_matrix(x_f, y_f, w, in_d, 0.0, g, lag, mat)
    return _bmat(mat)


def forward_mfpt_matrix(basis, weights, in_domain, guess, lag, test=None):
    """Compute correlation matrix for (forward) mean first passage time."""

    if test is None:
        test = basis
    mat = None
    for x_f, y_f, w, in_d, g in zip(test, basis, weights, in_domain, guess):
        mat = _forward_matrix(x_f, y_f, w, in_d, 1.0, g, lag, mat)
    return _bmat(mat)


def forward_feynman_kac_matrix(
    basis, weights, in_domain, function, guess, lag, test=None
):
    """Compute correlation matrix for forward Feynman-Kac problem.

    Parameters
    ----------
    basis : array-like (n_trajs,) of ndarray (n_frames, n_basis) of float
        Basis functions; must satisfy boundary conditions
    weights, function, guess : array-like (n_trajs,) of ndarray (n_frames,) of float
        Weight, function to integrate, guess for committor at each frame in trajectories
    in_domain : array-like (n_trajs,) of ndarray (n_frames,) of bool
        Whether each frame is in the domain (A U B)^c
    test : optional
        Test basis functions; if not specified, will use the same as the right basis

    Returns
    -------
    ndarray (2 * n_basis + 1, 2 * n_basis + 1) of float
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
    """Compute correlation matrix for backward committors.

    Parameters
    ----------
    w_basis, basis : array-like (n_trajs,) of ndarray (n_frames, n_basis) of float
        Basis functions to estimate reweighting factor, without boundary
        conditions; basis functions to estimate committor, must satisfy
        boundary conditions
    weights, guess : array-like (n_trajs,) of ndarray (n_frames,) of float
        Weight, guess for committor at each frame in trajectories
    in_domain : array-like (n_trajs,) of ndarray (n_frames,) of bool
        Whether each frame is in the domain (A U B)^c
    test : optional
        Test basis functions; if not specified, will use the same as the right basis

    Returns
    -------
    ndarray (2 * n_basis + 1, 2 * n_basis + 1) of float
    """

    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, g_b in zip(
        w_basis, w_test, basis, test, weights, in_domain, guess
    ):
        mat = _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, 0.0, g_b, lag, mat)
    return _bmat(mat)


def backward_mfpt_matrix(
    w_basis, basis, weights, in_domain, guess, lag, w_test=None, test=None
):
    """Compute correlation matrix for (backward) mean last passage time."""
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, g_b in zip(
        w_basis, w_test, basis, test, weights, in_domain, guess
    ):
        mat = _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, 1.0, g_b, lag, mat)
    return _bmat(mat)


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
    """Parameters
    ----------
    w_basis, basis : array-like (n_trajs,) of ndarray (n_frames, n_basis) of float
        Basis functions to estimate reweighting factor, without boundary
        conditions; basis functions to estimate committor, must satisfy
        boundary conditions
    weights, function, guess : array-like (n_trajs,) of ndarray (n_frames,) of float
        Weight, function to integrate, guess for committor at each frame in trajectories
    in_domain : array-like (n_trajs,) of ndarray (n_frames,) of bool
        Whether each frame is in the domain (A U B)^c
    test : optional
        Test basis functions; if not specified, will use the same as the right basis

    Returns
    -------
    ndarray (2 * n_basis + 1, 2 * n_basis + 1) of float
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


def reweight_integral_matrix(basis, weights, values, lag, test=None):
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, w, v in zip(basis, test, weights, values):
        mat = _reweight_integral_matrix(x_w, y_w, w, v, lag, mat)
    return _bmat(mat)


def forward_committor_integral_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_f, y_f, w, in_d, v, g in zip(
        w_basis, w_test, test, basis, weights, in_domain, values, guess
    ):
        mat = _forward_integral_matrix(
            x_w, y_w, x_f, y_f, w, in_d, v, 0.0, g, lag, mat
        )
    return _bmat(mat)


def forward_mfpt_integral_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_f, y_f, w, in_d, v, g in zip(
        w_basis, w_test, test, basis, weights, in_domain, values, guess
    ):
        mat = _forward_integral_matrix(
            x_w, y_w, x_f, y_f, w, in_d, v, 1.0, g, lag, mat
        )
    return _bmat(mat)


def forward_feynman_kac_integral_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    function,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_f, y_f, w, in_d, v, f, g in zip(
        w_basis,
        w_test,
        test,
        basis,
        weights,
        in_domain,
        values,
        function,
        guess,
    ):
        mat = _forward_integral_matrix(
            x_w, y_w, x_f, y_f, w, in_d, v, f, g, lag, mat
        )
    return _bmat(mat)


def backward_committor_integral_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, v, g in zip(
        w_basis,
        w_test,
        basis,
        test,
        weights,
        in_domain,
        values,
        guess,
    ):
        mat = _backward_integral_matrix(
            x_w, y_w, x_b, y_b, w, in_d, v, 0.0, g, lag, mat
        )
    return _bmat(mat)


def backward_mfpt_integral_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, v, g in zip(
        w_basis,
        w_test,
        basis,
        test,
        weights,
        in_domain,
        values,
        guess,
    ):
        mat = _backward_integral_matrix(
            x_w, y_w, x_b, y_b, w, in_d, v, 1.0, g, lag, mat
        )
    return _bmat(mat)


def backward_feynman_kac_integral_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    function,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, v, f, g in zip(
        w_basis,
        w_test,
        basis,
        test,
        weights,
        in_domain,
        function,
        values,
        guess,
    ):
        mat = _backward_integral_matrix(
            x_w, y_w, x_b, y_b, w, in_d, v, f, g, lag, mat
        )
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
    w_test=None,
    b_test=None,
    f_test=None,
):
    if w_test is None:
        w_test = w_basis
    if b_test is None:
        b_test = b_basis
    if f_test is None:
        f_test = f_basis
    mat = None
    for x_w, y_w, x_b, y_b, x_f, y_f, w, in_d, v, g_b, g_f in zip(
        w_basis,
        w_test,
        b_basis,
        b_test,
        f_test,
        f_basis,
        weights,
        in_domain,
        values,
        b_guess,
        f_guess,
    ):
        mat = _integral_matrix(
            x_w,
            y_w,
            x_b,
            y_b,
            x_f,
            y_f,
            w,
            in_d,
            in_d,
            v,
            0.0,
            0.0,
            g_b,
            g_f,
            lag,
            mat,
        )
    return _bmat(mat)


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
    w_test=None,
    b_test=None,
    f_test=None,
):
    if w_test is None:
        w_test = w_basis
    if b_test is None:
        b_test = b_basis
    if f_test is None:
        f_test = f_basis
    mat = None
    for (
        x_w,
        y_w,
        x_b,
        y_b,
        x_f,
        y_f,
        w,
        d_b,
        d_f,
        v,
        f_b,
        f_f,
        g_b,
        g_f,
    ) in zip(
        w_basis,
        w_test,
        b_basis,
        b_test,
        f_test,
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
            x_w,
            y_w,
            x_b,
            y_b,
            x_f,
            y_f,
            w,
            d_b,
            d_f,
            v,
            f_b,
            f_f,
            g_b,
            g_f,
            lag,
            mat,
        )
    return _bmat(mat)


def _reweight_matrix(x_w, y_w, w, lag, mat):
    """Compute the correlation matrix for reweighting.

    Parameters
    ----------
    x_w, y_w : ndarray (n_frames, n_basis) of float
        Left and right basis functions.
    w : ndarray (n_frames,) of float
        Initial weights for each frame
    lag : int
    mat : ndarray (2, 2) or None
        Matrix in which to store result
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
    """Compute the correlation matrix for forward-in-time statistics.

    Parameters
    ----------
    x_f, y_f : ndarray (n_frames, n_basis) of float
        Left and right basis functions
    w : ndarray (n_frames,) of float
        Initial weights for each frame
    d_f : ndarray (n_frames,) of bool
        Whether each frame is in the domain
    f_f, g_f : ndarray (n_frames,) of float
        Function to integrate until entrance into domain, guess for statistic.
    lag : int
    mat : ndarray (2, 2) or None
        Matrix in which to store result
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
    """Compute the correlation matrix for backward-in-time statistics.

    Parameters
    ----------
    x_w, y_w : ndarray (n_frames, n_basis) of float
        Left and right basis functions for weights (do not satisfy boundary
        conditions)
    x_b, y_b : ndarray (n_frames, n_basis) of float
        Left and right basis functions for statistic (must satisfy boundary
        conditions)
    w : ndarray (n_frames,) of float
        Initial weights for each frame
    d_b : ndarray (n_frames,) of bool
        Whether each frame is in the domain
    f_b, g_b : ndarray (n_frames,) of float
        Function to integrate until entrance into domain, guess for statistic.
    lag : int
    mat : ndarray (3, 3) or None
        Matrix in which to store result
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


def _reweight_integral_matrix(x_w, y_w, w, v, lag, mat):
    m = _kernel.reweight_integral_kernel(w, v, lag)
    if mat is None:
        mat = np.full((3, 3), None)
    # fmt: off
    # upper left
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    # upper right
    mat[0, 2] = _build(m[0, 1], None, None, mat[0, 2], lag)
    mat[1, 2] = _build(m[0, 1], x_w , None, mat[1, 2], lag)
    # lower right
    mat[2, 2] = _build(m[1, 1], None, None, mat[2, 2], lag)
    # fmt: on
    return mat


def _forward_integral_matrix(
    x_w, y_w, x_f, y_f, w, d_f, v, f_f, g_f, lag, mat
):
    f_f = np.broadcast_to(f_f, len(w) - 1)
    m = _kernel.forward_integral_kernel(w, d_f, v, f_f, g_f, lag)
    if mat is None:
        mat = np.full((4, 4), None)
    # fmt: off
    # upper left
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    # upper right
    mat[0, 2] = _build(m[0, 1], None, y_f , mat[0, 2], lag)
    mat[0, 3] = _build(m[0, 2], None, None, mat[0, 3], lag)
    mat[1, 2] = _build(m[0, 1], x_w , y_f , mat[1, 2], lag)
    mat[1, 3] = _build(m[0, 2], x_w , None, mat[1, 3], lag)
    # lower right
    mat[2, 2] = _build(m[1, 1], x_f , y_f , mat[2, 2], lag)
    mat[2, 3] = _build(m[1, 2], x_f , None, mat[2, 3], lag)
    mat[3, 3] = _build(m[2, 2], None, None, mat[3, 3], lag)
    # fmt: on
    return mat


def _backward_integral_matrix(
    x_w, y_w, x_b, y_b, w, d_b, v, f_b, g_b, lag, mat
):
    f_b = np.broadcast_to(f_b, len(w) - 1)
    m = _kernel.backward_integral_kernel(w, d_b, v, f_b, g_b, lag)
    if mat is None:
        mat = np.full((4, 4), None)
    # fmt: off
    # upper left
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[0, 2] = _build(m[0, 1], None, y_b , mat[0, 2], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    mat[1, 2] = _build(m[0, 1], x_w , y_b , mat[1, 2], lag)
    mat[2, 2] = _build(m[1, 1], x_b , y_b , mat[2, 2], lag)
    # upper right
    mat[0, 3] = _build(m[0, 2], None, None, mat[0, 3], lag)
    mat[1, 3] = _build(m[0, 2], x_w , None, mat[1, 3], lag)
    mat[2, 3] = _build(m[1, 2], x_b , None, mat[2, 3], lag)
    # lower right
    mat[3, 3] = _build(m[2, 2], None, None, mat[3, 3], lag)
    # fmt: on
    return mat


def _integral_matrix(
    x_w, y_w, x_b, y_b, x_f, y_f, w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag, mat
):
    f_b = np.broadcast_to(f_b, len(w) - 1)
    f_f = np.broadcast_to(f_f, len(w) - 1)
    m = _kernel.integral_kernel(w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag)
    if mat is None:
        mat = np.full((5, 5), None)
    # fmt: off
    # upper left
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[0, 2] = _build(m[0, 1], None, y_b , mat[0, 2], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    mat[1, 2] = _build(m[0, 1], x_w , y_b , mat[1, 2], lag)
    mat[2, 2] = _build(m[1, 1], x_b , y_b , mat[2, 2], lag)
    # upper right
    mat[0, 3] = _build(m[0, 2], None, y_f , mat[0, 3], lag)
    mat[0, 4] = _build(m[0, 3], None, None, mat[0, 4], lag)
    mat[1, 3] = _build(m[0, 2], x_w , y_f , mat[1, 3], lag)
    mat[1, 4] = _build(m[0, 3], x_w , None, mat[1, 4], lag)
    mat[2, 3] = _build(m[1, 2], x_b , y_f , mat[2, 3], lag)
    mat[2, 4] = _build(m[1, 3], x_b , None, mat[2, 4], lag)
    # lower right
    mat[3, 3] = _build(m[2, 2], x_f , y_f , mat[3, 3], lag)
    mat[3, 4] = _build(m[2, 3], x_f , None, mat[3, 4], lag)
    mat[4, 4] = _build(m[3, 3], None, None, mat[4, 4], lag)
    # fmt: on
    return mat


def _bmat(blocks):
    """Instantiate blocks into a full matrix.

    Parameters
    ----------
    blocks : array-like of 2-D ndarray

    Returns
    -------
    mat : 2-D ndarray
        Full matrix
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
    """Obtain shapes of block matrix

    Returns
    -------
    (tuple of int, tuple of int)
        Dimensions of blocks, where ith index of first tuple and the
        jth index of the second tuple correspond to the row and column
        dimension of the (i, j)th block
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
