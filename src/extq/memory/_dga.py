import operator

import numpy as np
import scipy.sparse

from .. import linalg
from ..moving_matmul import moving_matmul
from ._utils import badd
from ._utils import bmap
from ._utils import bmatmul
from ._utils import bshape
from ._utils import from_blocks
from ._utils import to_blockvec


def reweight_matrix(basis, weights, lag, test=None):
    if test is None:
        test = basis
    result = None
    for x_w, y_w, w in zip(basis, test, weights):
        result = _add(_reweight_matrix(x_w, y_w, w, lag), result)
    return result


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
    result = None
    for x_f, y_f, w, in_d, g in zip(test, basis, weights, in_domain, guess):
        result = _add(_forward_matrix(x_f, y_f, w, in_d, 0.0, g, lag), result)
    return result


def forward_mfpt_matrix(basis, weights, in_domain, guess, lag, test=None):
    if test is None:
        test = basis
    result = None
    for x_f, y_f, w, in_d, g in zip(test, basis, weights, in_domain, guess):
        result = _add(_forward_matrix(x_f, y_f, w, in_d, 1.0, g, lag), result)
    return result


def forward_feynman_kac_matrix(
    basis, weights, in_domain, function, guess, lag, test=None
):
    if test is None:
        test = basis
    result = None
    for x_f, y_f, w, in_d, f, g in zip(
        test, basis, weights, in_domain, function, guess
    ):
        result = _add(_forward_matrix(x_f, y_f, w, in_d, f, g, lag), result)
    return result


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
    result = None
    for x_w, y_w, x_b, y_b, w, in_d, g_b in zip(
        w_basis, w_test, basis, test, weights, in_domain, guess
    ):
        result = _add(
            _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, 0.0, g_b, lag),
            result,
        )
    return result


def backward_mfpt_matrix(
    w_basis, basis, weights, in_domain, guess, lag, w_test=None, test=None
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    result = None
    for x_w, y_w, x_b, y_b, w, in_d, g_b in zip(
        w_basis, w_test, basis, test, weights, in_domain, guess
    ):
        result = _add(
            _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, 1.0, g_b, lag),
            result,
        )
    return result


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
    result = None
    for x_w, y_w, x_b, y_b, w, in_d, f, g in zip(
        w_basis,
        w_test,
        basis,
        test,
        weights,
        in_domain,
        function,
        guess,
    ):
        result = _add(
            _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, f, g, lag), result
        )
    return result


def backward_solve(bgen):
    return _solve_backward(bgen)


def backward_transform(coeffs, w_basis, basis, weights, in_domain, guess):
    result = []
    for x_w, x_b, w, in_d, g in zip(w_basis, basis, weights, in_domain, guess):
        result.append(_backward_transform(coeffs, x_w, x_b, w, in_d, g))
    return result


def reweight_integral_matrix(basis, weights, values, lag, test=None):
    if test is None:
        test = basis
    result = None
    for x_w, y_w, w, v in zip(basis, test, weights, values):
        result = _add(_reweight_integral_matrix(x_w, y_w, w, v, lag), result)
    return result


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
    result = None
    for x_w, y_w, x_f, y_f, w, in_d, v, g in zip(
        w_basis, w_test, test, basis, weights, in_domain, values, guess
    ):
        result = _add(
            _forward_integral_matrix(
                x_w, y_w, x_f, y_f, w, in_d, v, 0.0, g, lag
            ),
            result,
        )
    return result


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
    result = None
    for x_w, y_w, x_f, y_f, w, in_d, v, g in zip(
        w_basis, w_test, test, basis, weights, in_domain, values, guess
    ):
        result = _add(
            _forward_integral_matrix(
                x_w, y_w, x_f, y_f, w, in_d, v, 1.0, g, lag
            ),
            result,
        )
    return result


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
    result = None
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
        result = _add(
            _forward_integral_matrix(
                x_w, y_w, x_f, y_f, w, in_d, v, f, g, lag
            ),
            result,
        )
    return result


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
    result = None
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
        result = _add(
            _backward_integral_matrix(
                x_w, y_w, x_b, y_b, w, in_d, v, 0.0, g, lag
            ),
            result,
        )
    return result


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
    result = None
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
        result = _add(
            _backward_integral_matrix(
                x_w, y_w, x_b, y_b, w, in_d, v, 1.0, g, lag
            ),
            result,
        )
    return result


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
    result = None
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
        result = _add(
            _backward_integral_matrix(
                x_w, y_w, x_b, y_b, w, in_d, v, f, g, lag
            ),
            result,
        )
    return result


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
    result = None
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
        result = _add(
            _integral_matrix(
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
            ),
            result,
        )
    return result


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
    result = None
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
        result = _add(
            _integral_matrix(
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
            ),
            result,
        )
    return result


def integral_solve(bgen):
    assert bgen.shape == (5, 5)
    return _solve_observable(bgen[3:, 3:], bgen[:3, :3], bgen[:3, 3:])


def _reweight_matrix(x_w, y_w, w, lag):
    c = np.ones((len(w), 1))
    o = np.ones(len(w))

    m = np.full((1, 1), None)
    if lag == 0:
        m[0, 0] = np.ones(len(w))
    else:
        m[0, 0] = np.ones(len(w) - 1)
        m = moving_matmul(m, lag)

    u = [[o]]
    v = [[o]]
    x = [[c, x_w]]
    y = [[c, y_w]]

    return _build(x, y, w, m, u, v, lag)


def _reweight_transform(coeffs, x_w, w):
    c = np.ones((len(w), 1))
    o = np.ones(len(w))

    u = [[o]]
    x = [[c, x_w]]

    wu = bmap(lambda a: w * a, _blocks(u))
    result = bmatmul(
        operator.mul,
        wu,
        bmatmul(operator.matmul, _blocks(x), coeffs[:, None]),
    )
    assert result.shape == (1, 1)
    return result[0, 0]


def _forward_matrix(x_f, y_f, w, d_f, f_f, g_f, lag):
    c = np.ones((len(w), 1))
    d = np.where(d_f, 1.0, 0.0)
    b = np.where(d_f, 0.0, g_f)
    g = np.where(d_f, g_f, 0.0)
    o = np.ones(len(w))

    m = np.full((2, 2), None)
    if lag == 0:
        m[0, 0] = d
        m[0, 1] = np.zeros(len(w))
        m[1, 1] = np.ones(len(w))
    else:
        m[0, 0] = d[:-1] * d[1:]
        m[0, 1] = d[:-1] * (b[1:] + f_f)
        m[1, 1] = np.ones(len(w) - 1)
        m = moving_matmul(m, lag)

    u = [[d, None], [None, o]]
    v = [[d, g], [None, o]]
    x = [[x_f, None], [None, c]]
    y = [[y_f, None], [None, c]]

    return _build(x, y, w, m, u, v, lag)


def _forward_transform(coeffs, y_f, d_f, g_f):
    c = np.ones((len(g_f), 1))
    d = np.where(d_f, 1.0, 0.0)
    o = np.ones(len(g_f))

    v = [[d, g_f], [None, o]]
    y = [[y_f, None], [None, c]]

    v = _blocks(v)
    result = bmatmul(
        operator.mul,
        v,
        bmatmul(operator.matmul, _blocks(y), coeffs[:, None]),
    )
    assert result.shape == (2, 1)
    assert np.all(result[1, 0] == 1.0)
    return result[0, 0]


def _backward_matrix(x_w, y_w, x_b, y_b, w, d_b, f_b, g_b, lag):
    c = np.ones((len(w), 1))
    d = np.where(d_b, 1.0, 0.0)
    a = np.where(d_b, 0.0, g_b)
    g = np.where(d_b, g_b, 0.0)
    o = np.ones(len(w))

    m = np.full((2, 2), None)
    if lag == 0:
        m[0, 0] = np.ones(len(w))
        m[0, 1] = np.zeros(len(w))
        m[1, 1] = d
    else:
        m[0, 0] = np.ones(len(w) - 1)
        m[0, 1] = (a[:-1] + f_b) * d[1:]
        m[1, 1] = d[:-1] * d[1:]
        m = moving_matmul(m, lag)

    u = [[o, None], [g, d]]
    v = [[o, None], [None, d]]
    x = [[c, x_w, None], [None, None, x_b]]
    y = [[c, y_w, None], [None, None, y_b]]

    return _build(x, y, w, m, u, v, lag)


def _backward_transform(coeffs, x_w, x_b, w, d_b, g_b):
    c = np.ones((len(w), 1))
    d = np.where(d_b, 1.0, 0.0)
    o = np.ones(len(w))

    u = [[o, None], [g_b, d]]
    x = [[c, x_w, None], [None, None, x_b]]

    wu = bmap(lambda a: w * a, _blocks(u))
    result = bmatmul(
        operator.mul,
        wu,
        bmatmul(operator.matmul, _blocks(x), coeffs[:, None]),
    )
    assert result.shape == (2, 1)
    return result[1, 0] / result[0, 0]


def _reweight_integral_matrix(x_w, y_w, w, v, lag):
    c = np.ones((len(w), 1))
    o = np.ones(len(w))

    m = np.full((2, 2), None)
    if lag == 0:
        m[0, 0] = np.ones(len(w))
        m[0, 1] = np.zeros(len(w))
        m[1, 1] = np.ones(len(w))
    else:
        m[0, 0] = np.ones(len(w) - 1)
        m[0, 1] = v
        m[1, 1] = np.ones(len(w) - 1)
        m = moving_matmul(m, lag)

    u = [[o, None], [None, o]]
    v = [[o, None], [None, o]]
    x = [[c, x_w, None], [None, None, c]]
    y = [[c, y_w, None], [None, None, c]]

    return _build(x, y, w, m, u, v, lag)


def _forward_integral_matrix(x_w, y_w, x_f, y_f, w, d_f, v, f_f, g_f, lag):
    c = np.ones((len(w), 1))
    d = np.where(d_f, 1.0, 0.0)
    b = np.where(d_f, 0.0, g_f)
    g = np.where(d_f, g_f, 0.0)
    o = np.ones(len(w))

    m = np.full((3, 3), None)
    if lag == 0:
        m[0, 0] = np.ones(len(w))
        m[0, 1] = np.zeros(len(w))
        m[0, 2] = np.zeros(len(w))
        m[1, 1] = d
        m[1, 2] = np.zeros(len(w))
        m[2, 2] = np.ones(len(w))
    else:
        m[0, 0] = np.ones(len(w) - 1)
        m[0, 1] = v * d[1:]
        m[0, 2] = v * b[1:]
        m[1, 1] = d[:-1] * d[1:]
        m[1, 2] = d[:-1] * (b[1:] + f_f)
        m[2, 2] = np.ones(len(w) - 1)
        m = moving_matmul(m, lag)

    u = [[o, None, None], [None, d, None], [None, None, o]]
    v = [[o, None, None], [None, d, g], [None, None, o]]
    x = [[c, x_w, None, None], [None, None, x_f, None], [None, None, None, c]]
    y = [[c, y_w, None, None], [None, None, y_f, None], [None, None, None, c]]

    return _build(x, y, w, m, u, v, lag)


def _backward_integral_matrix(x_w, y_w, x_b, y_b, w, d_b, v, f_b, g_b, lag):
    c = np.ones((len(w), 1))
    d = np.where(d_b, 1.0, 0.0)
    a = np.where(d_b, 0.0, g_b)
    g = np.where(d_b, g_b, 0.0)
    o = np.ones(len(w))

    m = np.full((3, 3), None)
    if lag == 0:
        m[0, 0] = np.ones(len(w))
        m[0, 1] = np.zeros(len(w))
        m[0, 2] = np.zeros(len(w))
        m[1, 1] = d
        m[1, 2] = np.zeros(len(w))
        m[2, 2] = np.ones(len(w))
    else:
        m[0, 0] = np.ones(len(w) - 1)
        m[0, 1] = (a[:-1] + f_b) * d[1:]
        m[0, 2] = a[:-1] * v
        m[1, 1] = d[:-1] * d[1:]
        m[1, 2] = d[:-1] * v
        m[2, 2] = np.ones(len(w) - 1)
        m = moving_matmul(m, lag)

    u = [[o, None, None], [g, d, None], [None, None, o]]
    v = [[o, None, None], [None, d, None], [None, None, o]]
    x = [[c, x_w, None, None], [None, None, x_b, None], [None, None, None, c]]
    y = [[c, y_w, None, None], [None, None, y_b, None], [None, None, None, c]]

    return _build(x, y, w, m, u, v, lag)


def _integral_matrix(
    x_w, y_w, x_b, y_b, x_f, y_f, w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag
):
    c = np.ones((len(w), 1))
    db = np.where(d_b, 1.0, 0.0)
    df = np.where(d_f, 1.0, 0.0)
    a = np.where(d_b, 0.0, g_b)
    b = np.where(d_f, 0.0, g_f)
    gb = np.where(d_b, g_b, 0.0)
    gf = np.where(d_f, g_f, 0.0)
    o = np.ones(len(w))

    m = np.full((4, 4), None)
    if lag == 0:
        m[0, 0] = np.ones(len(w))
        m[0, 1] = np.zeros(len(w))
        m[0, 2] = np.zeros(len(w))
        m[0, 3] = np.zeros(len(w))
        m[1, 1] = db
        m[1, 2] = np.zeros(len(w))
        m[1, 3] = np.zeros(len(w))
        m[2, 2] = df
        m[2, 3] = np.zeros(len(w))
        m[3, 3] = np.ones(len(w))
    else:
        m[0, 0] = np.ones(len(w) - 1)
        m[0, 1] = (a[:-1] + f_b) * db[1:]
        m[0, 2] = a[:-1] * v * df[1:]
        m[0, 3] = a[:-1] * v * b[1:]
        m[1, 1] = db[:-1] * db[1:]
        m[1, 2] = db[:-1] * v * df[1:]
        m[1, 3] = db[:-1] * v * b[1:]
        m[2, 2] = df[:-1] * df[1:]
        m[2, 3] = df[:-1] * (b[1:] + f_f)
        m[3, 3] = np.ones(len(w) - 1)
        m = moving_matmul(m, lag)

    u = [
        [o, None, None, None],
        [gb, db, None, None],
        [None, None, df, None],
        [None, None, None, o],
    ]
    v = [
        [o, None, None, None],
        [None, db, None, None],
        [None, None, df, gf],
        [None, None, None, o],
    ]
    x = [
        [c, x_w, None, None, None],
        [None, None, x_b, None, None],
        [None, None, None, x_f, None],
        [None, None, None, None, c],
    ]
    y = [
        [c, y_w, None, None, None],
        [None, None, y_b, None, None],
        [None, None, None, y_f, None],
        [None, None, None, None, c],
    ]

    return _build(x, y, w, m, u, v, lag)


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


def _build(x, y, w, m, u, v, lag):
    assert lag >= 0
    assert np.all(w[len(w) - lag :] == 0.0)
    last = -lag if lag > 0 else None
    m = bmap(lambda a: w[:last] * a, _blocks(m))
    u = bmap(lambda a: a[:last], _blocks(u)).T
    v = bmap(lambda a: a[lag:], _blocks(v))
    umv = bmatmul(operator.mul, u, bmatmul(operator.mul, m, v))
    x = bmap(lambda a: a[:last].T, _blocks(x)).T
    y = bmap(lambda a: a[lag:], _blocks(y))
    umvy = bmatmul(_scale, umv, y)
    xumvy = bmatmul(operator.matmul, x, umvy)
    return xumvy


def _blocks(blocks):
    result = np.empty((len(blocks), len(blocks[0])), dtype=object)
    result[:] = blocks
    return result


def _add(mat, acc=None):
    if acc is None:
        return mat
    else:
        return badd(mat, acc)


def _scale(w, x):
    if scipy.sparse.issparse(x):
        x = x.tocsr()
        return scipy.sparse.csr_matrix(
            (x.data * np.repeat(w, np.diff(x.indptr)), x.indices, x.indptr),
            shape=x.shape,
        )
    else:
        return x * w[:, None]
