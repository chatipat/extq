import operator

import numpy as np

from .. import linalg
from ._kernel import backward_integral_kernel
from ._kernel import backward_kernel
from ._kernel import forward_integral_kernel
from ._kernel import forward_kernel
from ._kernel import integral_kernel
from ._kernel import reweight_integral_kernel
from ._kernel import reweight_kernel
from ._tlcc import wtlcc_dense as _build
from ._utils import bmatmul
from ._utils import bshape
from ._utils import from_blocks
from ._utils import to_blockvec


def reweight_matrix(basis, weights, lag, test=None):
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, w in zip(basis, test, weights):
        mat = _reweight_matrix(x_w, y_w, w, lag, mat)
    return mat


def reweight_solve(bgen):
    return _solve_backward(bgen)


def reweight_transform(coeffs, basis, weights):
    result = []
    for x_w, w in zip(basis, weights):
        result.append(_reweight_transform(coeffs, x_w, w))
    return result


def forward_committor_matrix(basis, weights, in_domain, guess, lag, test=None):
    if test is None:
        test = basis
    mat = None
    for x_f, y_f, w, in_d, g in zip(test, basis, weights, in_domain, guess):
        mat = _forward_matrix(x_f, y_f, w, in_d, 0.0, g, lag, mat)
    return mat


def forward_mfpt_matrix(basis, weights, in_domain, guess, lag, test=None):
    if test is None:
        test = basis
    mat = None
    for x_f, y_f, w, in_d, g in zip(test, basis, weights, in_domain, guess):
        mat = _forward_matrix(x_f, y_f, w, in_d, 1.0, g, lag, mat)
    return mat


def forward_feynman_kac_matrix(
    basis, weights, in_domain, function, guess, lag, test=None
):
    if test is None:
        test = basis
    mat = None
    for x_f, y_f, w, in_d, f, g in zip(
        test, basis, weights, in_domain, function, guess
    ):
        mat = _forward_matrix(x_f, y_f, w, in_d, f, g, lag, mat)
    return mat


def forward_solve(bgen):
    return _solve_forward(bgen)


def forward_transform(coeffs, basis, in_domain, guess):
    result = []
    for y_f, in_d, g in zip(basis, in_domain, guess):
        result.append(_forward_transform(coeffs, y_f, in_d, g))
    return result


def backward_committor_matrix(
    w_basis, basis, weights, in_domain, guess, lag, w_test=None, test=None
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, g_b in zip(
        w_basis, w_test, basis, test, weights, in_domain, guess
    ):
        mat = _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, 0.0, g_b, lag, mat)
    return mat


def backward_mfpt_matrix(
    w_basis, basis, weights, in_domain, guess, lag, w_test=None, test=None
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, g_b in zip(
        w_basis, w_test, basis, test, weights, in_domain, guess
    ):
        mat = _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, 1.0, g_b, lag, mat)
    return mat


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
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, x_b, y_b, w, in_d, f, g in zip(
        w_basis, w_test, basis, test, weights, in_domain, function, guess
    ):
        mat = _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, f, g, lag, mat)
    return mat


def backward_solve(bgen):
    return _solve_backward(bgen)


def backward_transform(coeffs, w_basis, basis, in_domain, guess):
    result = []
    for x_w, x_b, in_d, g in zip(w_basis, basis, in_domain, guess):
        result.append(_backward_transform(coeffs, x_w, x_b, in_d, g))
    return result


def reweight_integral_matrix(basis, weights, values, lag, test=None):
    if test is None:
        test = basis
    mat = None
    for x_w, y_w, w, v in zip(basis, test, weights, values):
        mat = _reweight_integral_matrix(x_w, y_w, w, v, lag, mat)
    return mat


def reweight_integral_solve(bgen):
    assert bgen.shape == (3, 3)
    return _solve_observable(bgen[2:, 2:], bgen[:2, :2], bgen[:2, 2:])


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
    return mat


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
    return mat


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
    return mat


def forward_integral_solve(bgen):
    assert bgen.shape == (4, 4)
    return _solve_observable(bgen[2:, 2:], bgen[:2, :2], bgen[:2, 2:])


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
    return mat


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
    return mat


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
    return mat


def backward_integral_solve(bgen):
    assert bgen.shape == (4, 4)
    return _solve_observable(bgen[3:, 3:], bgen[:3, :3], bgen[:3, 3:])


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
    return mat


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
    return mat


def integral_solve(bgen):
    assert bgen.shape == (5, 5)
    return _solve_observable(bgen[3:, 3:], bgen[:3, :3], bgen[:3, 3:])


def _reweight_matrix(x_w, y_w, w, lag, mat):
    m = reweight_kernel(w, lag)
    if mat is None:
        mat = np.full((2, 2), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], None, None, mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 0], None, y_w , mat[0, 1], lag)
    mat[1, 0] = _build(m[0, 0], x_w , None, mat[1, 0], lag)
    mat[1, 1] = _build(m[0, 0], x_w , y_w , mat[1, 1], lag)
    # fmt: on
    return mat


def _reweight_transform(coeffs, x_w, w):
    assert coeffs.shape == (2,)
    return w * (coeffs[0] + x_w @ coeffs[1])


def _forward_matrix(x_f, y_f, w, d_f, f_f, g_f, lag, mat):
    f_f = np.broadcast_to(f_f, len(w) - 1)
    m = forward_kernel(w, d_f, f_f, g_f, lag)
    if mat is None:
        mat = np.full((2, 2), None)
    # fmt: off
    mat[0, 0] = _build(m[0, 0], x_f , y_f , mat[0, 0], lag)
    mat[0, 1] = _build(m[0, 1], x_f , None, mat[0, 1], lag)
    mat[1, 1] = _build(m[1, 1], None, None, mat[1, 1], lag)
    # fmt: on
    return mat


def _forward_transform(coeffs, y_f, d_f, g_f):
    assert coeffs.shape == (2,)
    return g_f + np.where(d_f, y_f @ coeffs[0], 0.0) / coeffs[1]


def _backward_matrix(x_w, y_w, x_b, y_b, w, d_b, f_b, g_b, lag, mat):
    f_b = np.broadcast_to(f_b, len(w) - 1)
    m = backward_kernel(w, d_b, f_b, g_b, lag)
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


def _backward_transform(coeffs, x_w, x_b, d_b, g_b):
    assert coeffs.shape == (3,)
    com = coeffs[0] + x_w @ coeffs[1]
    return g_b + np.where(d_b, x_b @ coeffs[2], 0.0) / com


def _reweight_integral_matrix(x_w, y_w, w, v, lag, mat):
    m = reweight_integral_kernel(w, v, lag)
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
    m = forward_integral_kernel(w, d_f, v, f_f, g_f, lag)
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
    m = backward_integral_kernel(w, d_b, v, f_b, g_b, lag)
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
    m = integral_kernel(w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag)
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


def _solve_forward(bgen):
    shape = bshape(bgen)
    gen = from_blocks(bgen)
    coeffs = np.concatenate(
        [linalg.solve(gen[:-1, :-1], -gen[:-1, -1]), [1.0]]
    )
    return to_blockvec(coeffs, shape[1])


def _solve_backward(bgen):
    shape = bshape(bgen)
    gen = from_blocks(bgen)
    coeffs = np.concatenate(
        [[1.0], linalg.solve(gen.T[1:, 1:], -gen.T[1:, 0])]
    )
    return to_blockvec(coeffs, shape[0])


def _solve_observable(bgen_lr, bgen_ul, bgen_ur):
    forward_coeffs = _solve_forward(bgen_lr)
    backward_coeffs = _solve_backward(bgen_ul)
    result = bmatmul(
        operator.matmul,
        backward_coeffs[None, :],
        bmatmul(operator.matmul, bgen_ur, forward_coeffs[:, None]),
    )
    assert result.shape == (1, 1)
    return result[0, 0]
