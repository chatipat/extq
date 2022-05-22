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
from ._utils import asblocks
from ._utils import badd
from ._utils import bconcatenate_lag
from ._utils import bmap
from ._utils import bmatmul
from ._utils import bshape
from ._utils import from_blocks
from ._utils import to_blockvec


def reweight_matrix(basis, weights, lag, test=None, chunks=1):
    if test is None:
        test = basis
    terms = (
        _reweight_matrix(x_w, y_w, w, lag)
        for x_w, y_w, w in zip(basis, test, weights)
    )
    return _sum(terms, lag, chunks=chunks)


def reweight_solve(bgen):
    return _solve_backward(bgen)


def reweight_transform(coeffs, basis, weights):
    result = []
    for x_w, w in zip(basis, weights):
        result.append(_reweight_transform(coeffs, x_w, w))
    return result


def forward_committor_matrix(
    basis, weights, in_domain, guess, lag, test=None, chunks=1
):
    if test is None:
        test = basis
    terms = (
        _forward_matrix(x_f, y_f, w, in_d, 0.0, g, lag)
        for x_f, y_f, w, in_d, g in zip(test, basis, weights, in_domain, guess)
    )
    return _sum(terms, lag, chunks=chunks)


def forward_mfpt_matrix(
    basis, weights, in_domain, guess, lag, test=None, chunks=1
):
    if test is None:
        test = basis
    terms = (
        _forward_matrix(x_f, y_f, w, in_d, 1.0, g, lag)
        for x_f, y_f, w, in_d, g in zip(test, basis, weights, in_domain, guess)
    )
    return _sum(terms, lag, chunks=chunks)


def forward_feynman_kac_matrix(
    basis, weights, in_domain, function, guess, lag, test=None, chunks=1
):
    if test is None:
        test = basis
    terms = (
        _forward_matrix(x_f, y_f, w, in_d, f, g, lag)
        for x_f, y_f, w, in_d, f, g in zip(
            test, basis, weights, in_domain, function, guess
        )
    )
    return _sum(terms, lag, chunks=chunks)


def forward_solve(bgen):
    return _solve_forward(bgen)


def forward_transform(coeffs, basis, in_domain, guess):
    result = []
    for y_f, in_d, g in zip(basis, in_domain, guess):
        result.append(_forward_transform(coeffs, y_f, in_d, g))
    return result


def backward_committor_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    w_test=None,
    test=None,
    chunks=1,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    terms = (
        _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, 0.0, g_b, lag)
        for x_w, y_w, x_b, y_b, w, in_d, g_b in zip(
            w_basis, w_test, basis, test, weights, in_domain, guess
        )
    )
    return _sum(terms, lag, chunks=chunks)


def backward_mfpt_matrix(
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    w_test=None,
    test=None,
    chunks=1,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    terms = (
        _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, 1.0, g_b, lag)
        for x_w, y_w, x_b, y_b, w, in_d, g_b in zip(
            w_basis, w_test, basis, test, weights, in_domain, guess
        )
    )
    return _sum(terms, lag, chunks=chunks)


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
    chunks=1,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    terms = (
        _backward_matrix(x_w, y_w, x_b, y_b, w, in_d, f, g, lag)
        for x_w, y_w, x_b, y_b, w, in_d, f, g in zip(
            w_basis,
            w_test,
            basis,
            test,
            weights,
            in_domain,
            function,
            guess,
        )
    )
    return _sum(terms, lag, chunks=chunks)


def backward_solve(bgen):
    return _solve_backward(bgen)


def backward_transform(coeffs, w_basis, basis, weights, in_domain, guess):
    result = []
    for x_w, x_b, w, in_d, g in zip(w_basis, basis, weights, in_domain, guess):
        result.append(_backward_transform(coeffs, x_w, x_b, w, in_d, g))
    return result


def reweight_integral_matrix(basis, weights, values, lag, test=None, chunks=1):
    if test is None:
        test = basis
    terms = (
        _reweight_integral_matrix(x_w, y_w, w, v, lag)
        for x_w, y_w, w, v in zip(basis, test, weights, values)
    )
    return _sum(terms, lag, chunks=chunks)


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
    chunks=1,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    terms = (
        _forward_integral_matrix(x_w, y_w, x_f, y_f, w, in_d, v, 0.0, g, lag)
        for x_w, y_w, x_f, y_f, w, in_d, v, g in zip(
            w_basis, w_test, test, basis, weights, in_domain, values, guess
        )
    )
    return _sum(terms, lag, chunks=chunks)


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
    chunks=1,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    terms = (
        _forward_integral_matrix(x_w, y_w, x_f, y_f, w, in_d, v, 1.0, g, lag)
        for x_w, y_w, x_f, y_f, w, in_d, v, g in zip(
            w_basis, w_test, test, basis, weights, in_domain, values, guess
        )
    )
    return _sum(terms, lag, chunks=chunks)


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
    chunks=1,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    terms = (
        _forward_integral_matrix(x_w, y_w, x_f, y_f, w, in_d, v, f, g, lag)
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
        )
    )
    return _sum(terms, lag, chunks=chunks)


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
    chunks=1,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    terms = (
        _backward_integral_matrix(x_w, y_w, x_b, y_b, w, in_d, v, 0.0, g, lag)
        for x_w, y_w, x_b, y_b, w, in_d, v, g in zip(
            w_basis,
            w_test,
            basis,
            test,
            weights,
            in_domain,
            values,
            guess,
        )
    )
    return _sum(terms, lag, chunks=chunks)


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
    chunks=1,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    terms = (
        _backward_integral_matrix(x_w, y_w, x_b, y_b, w, in_d, v, 1.0, g, lag)
        for x_w, y_w, x_b, y_b, w, in_d, v, g in zip(
            w_basis,
            w_test,
            basis,
            test,
            weights,
            in_domain,
            values,
            guess,
        )
    )
    return _sum(terms, lag, chunks=chunks)


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
    chunks=1,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    terms = (
        _backward_integral_matrix(x_w, y_w, x_b, y_b, w, in_d, v, f, g, lag)
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
        )
    )
    return _sum(terms, lag, chunks=chunks)


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
    chunks=1,
):
    if w_test is None:
        w_test = w_basis
    if b_test is None:
        b_test = b_basis
    if f_test is None:
        f_test = f_basis
    terms = (
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
        )
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
        )
    )
    return _sum(terms, lag, chunks=chunks)


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
    chunks=1,
):
    if w_test is None:
        w_test = w_basis
    if b_test is None:
        b_test = b_basis
    if f_test is None:
        f_test = f_basis
    terms = (
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
        )
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
        )
    )
    return _sum(terms, lag, chunks=chunks)


def integral_solve(bgen):
    assert bgen.shape == (5, 5)
    return _solve_observable(bgen[3:, 3:], bgen[:3, :3], bgen[:3, 3:])


def _reweight_matrix(x_w, y_w, w, lag):
    c = np.ones((len(w), 1))
    x = [[c, x_w]]
    y = [[c, y_w]]
    m = reweight_kernel(w, lag)
    return x, y, m


def _reweight_transform(coeffs, x_w, w):
    c = np.ones((len(w), 1))
    o = np.ones(len(w))

    u = [[o]]
    x = [[c, x_w]]

    wu = bmap(lambda a: w * a, asblocks(u))
    result = bmatmul(
        operator.mul,
        wu,
        bmatmul(operator.matmul, asblocks(x), coeffs[:, None]),
    )
    assert result.shape == (1, 1)
    return result[0, 0]


def _forward_matrix(x_f, y_f, w, d_f, f_f, g_f, lag):
    c = np.ones((len(w), 1))
    x = [[x_f, None], [None, c]]
    y = [[y_f, None], [None, c]]
    m = forward_kernel(w, d_f, f_f, g_f, lag)
    return x, y, m


def _forward_transform(coeffs, y_f, d_f, g_f):
    c = np.ones((len(g_f), 1))
    d = np.where(d_f, 1.0, 0.0)
    o = np.ones(len(g_f))

    v = [[d, g_f], [None, o]]
    y = [[y_f, None], [None, c]]

    v = asblocks(v)
    result = bmatmul(
        operator.mul,
        v,
        bmatmul(operator.matmul, asblocks(y), coeffs[:, None]),
    )
    assert result.shape == (2, 1)
    assert np.all(result[1, 0] == 1.0)
    return result[0, 0]


def _backward_matrix(x_w, y_w, x_b, y_b, w, d_b, f_b, g_b, lag):
    c = np.ones((len(w), 1))
    x = [[c, x_w, None], [None, None, x_b]]
    y = [[c, y_w, None], [None, None, y_b]]
    m = backward_kernel(w, d_b, f_b, g_b, lag)
    return x, y, m


def _backward_transform(coeffs, x_w, x_b, w, d_b, g_b):
    c = np.ones((len(w), 1))
    d = np.where(d_b, 1.0, 0.0)
    o = np.ones(len(w))

    u = [[o, None], [g_b, d]]
    x = [[c, x_w, None], [None, None, x_b]]

    wu = bmap(lambda a: w * a, asblocks(u))
    result = bmatmul(
        operator.mul,
        wu,
        bmatmul(operator.matmul, asblocks(x), coeffs[:, None]),
    )
    assert result.shape == (2, 1)
    return result[1, 0] / result[0, 0]


def _reweight_integral_matrix(x_w, y_w, w, v, lag):
    c = np.ones((len(w), 1))
    x = [[c, x_w, None], [None, None, c]]
    y = [[c, y_w, None], [None, None, c]]
    m = reweight_integral_kernel(w, v, lag)
    return x, y, m


def _forward_integral_matrix(x_w, y_w, x_f, y_f, w, d_f, v, f_f, g_f, lag):
    c = np.ones((len(w), 1))
    x = [[c, x_w, None, None], [None, None, x_f, None], [None, None, None, c]]
    y = [[c, y_w, None, None], [None, None, y_f, None], [None, None, None, c]]
    m = forward_integral_kernel(w, d_f, v, f_f, g_f, lag)
    return x, y, m


def _backward_integral_matrix(x_w, y_w, x_b, y_b, w, d_b, v, f_b, g_b, lag):
    c = np.ones((len(w), 1))
    x = [[c, x_w, None, None], [None, None, x_b, None], [None, None, None, c]]
    y = [[c, y_w, None, None], [None, None, y_b, None], [None, None, None, c]]
    m = backward_integral_kernel(w, d_b, v, f_b, g_b, lag)
    return x, y, m


def _integral_matrix(
    x_w, y_w, x_b, y_b, x_f, y_f, w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag
):
    c = np.ones((len(w), 1))
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
    m = integral_kernel(w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag)
    return x, y, m


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


def _sum(terms, lag, chunks=1):
    mat = None
    for c in _chunk(terms, chunks):
        xwy = _sum_chunks(c, lag)
        if mat is None:
            mat = xwy
        else:
            mat = badd(xwy, mat)
    return mat


def _sum_chunks(terms, lag):
    assert lag >= 0
    xlist = []
    ylist = []
    mlist = []
    for x, y, m in terms:
        xlist.append(x)
        ylist.append(y)
        mlist.append(m)
    x = bconcatenate_lag(xlist, 0, lag)
    y = bconcatenate_lag(ylist, lag, 0)
    m = bconcatenate_lag(mlist, 0, 0)
    x = bmap(lambda a: a.T, x).T
    my = bmatmul(linalg.scale_rows, m, y)
    return bmatmul(operator.matmul, x, my)


def _chunk(a, n=None):
    if n is None:
        yield list(a)
        return
    c = []
    for x in a:
        c.append(x)
        if len(c) == n:
            yield c
            c = []
    if len(c) > 0:
        yield c
