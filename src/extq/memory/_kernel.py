import operator

import numpy as np

from ..moving_matmul import moving_matmul
from ._utils import asblocks
from ._utils import bmap
from ._utils import bmatmul


def reweight_kernel(w, lag):
    o = np.ones(len(w))
    o1 = o[:-1]

    m = np.full((1, 1), None)
    if lag == 0:
        m[0, 0] = o
    else:
        m[0, 0] = o1
        m = moving_matmul(m, lag)

    p = [[o]]
    q = [[o]]

    return _assemble(w, m, p, q, lag)


def forward_kernel(w, d_f, f_f, g_f, lag):
    d = np.where(d_f, 1.0, 0.0)
    b = np.where(d_f, 0.0, g_f)
    g = np.where(d_f, g_f, 0.0)
    z = np.zeros(len(w))
    o = np.ones(len(w))
    o1 = o[:-1]

    m = np.full((2, 2), None)
    if lag == 0:
        m[0, 0] = d
        m[0, 1] = z
        m[1, 1] = o
    else:
        m[0, 0] = d[:-1] * d[1:]
        m[0, 1] = d[:-1] * (b[1:] + f_f)
        m[1, 1] = o1
        m = moving_matmul(m, lag)

    p = [[d, None], [None, o]]
    q = [[d, g], [None, o]]

    return _assemble(w, m, p, q, lag)


def backward_kernel(w, d_b, f_b, g_b, lag):
    d = np.where(d_b, 1.0, 0.0)
    a = np.where(d_b, 0.0, g_b)
    g = np.where(d_b, g_b, 0.0)
    z = np.zeros(len(w))
    o = np.ones(len(w))
    o1 = o[:-1]

    m = np.full((2, 2), None)
    if lag == 0:
        m[0, 0] = o
        m[0, 1] = z
        m[1, 1] = d
    else:
        m[0, 0] = o1
        m[0, 1] = (a[:-1] + f_b) * d[1:]
        m[1, 1] = d[:-1] * d[1:]
        m = moving_matmul(m, lag)

    p = [[o, None], [g, d]]
    q = [[o, None], [None, d]]

    return _assemble(w, m, p, q, lag)


def reweight_integral_kernel(w, v, lag):
    z = np.zeros(len(w))
    o = np.ones(len(w))
    o1 = o[:-1]

    m = np.full((2, 2), None)
    if lag == 0:
        m[0, 0] = o
        m[0, 1] = z
        m[1, 1] = o
    else:
        m[0, 0] = o1
        m[0, 1] = v
        m[1, 1] = o1
        m = moving_matmul(m, lag)

    p = [[o, None], [None, o]]
    q = [[o, None], [None, o]]

    return _assemble(w, m, p, q, lag)


def forward_integral_kernel(w, d_f, v, f_f, g_f, lag):
    d = np.where(d_f, 1.0, 0.0)
    b = np.where(d_f, 0.0, g_f)
    g = np.where(d_f, g_f, 0.0)
    z = np.zeros(len(w))
    o = np.ones(len(w))
    o1 = o[:-1]

    m = np.full((3, 3), None)
    if lag == 0:
        m[0, 0] = o
        m[0, 1] = z
        m[0, 2] = z
        m[1, 1] = d
        m[1, 2] = z
        m[2, 2] = o
    else:
        m[0, 0] = o1
        m[0, 1] = v * d[1:]
        m[0, 2] = v * b[1:]
        m[1, 1] = d[:-1] * d[1:]
        m[1, 2] = d[:-1] * (b[1:] + f_f)
        m[2, 2] = o1
        m = moving_matmul(m, lag)

    p = [[o, None, None], [None, d, None], [None, None, o]]
    q = [[o, None, None], [None, d, g], [None, None, o]]

    return _assemble(w, m, p, q, lag)


def backward_integral_kernel(w, d_b, v, f_b, g_b, lag):
    d = np.where(d_b, 1.0, 0.0)
    a = np.where(d_b, 0.0, g_b)
    g = np.where(d_b, g_b, 0.0)
    z = np.zeros(len(w))
    o = np.ones(len(w))
    o1 = o[:-1]

    m = np.full((3, 3), None)
    if lag == 0:
        m[0, 0] = o
        m[0, 1] = z
        m[0, 2] = z
        m[1, 1] = d
        m[1, 2] = z
        m[2, 2] = o
    else:
        m[0, 0] = o1
        m[0, 1] = (a[:-1] + f_b) * d[1:]
        m[0, 2] = a[:-1] * v
        m[1, 1] = d[:-1] * d[1:]
        m[1, 2] = d[:-1] * v
        m[2, 2] = o1
        m = moving_matmul(m, lag)

    p = [[o, None, None], [g, d, None], [None, None, o]]
    q = [[o, None, None], [None, d, None], [None, None, o]]

    return _assemble(w, m, p, q, lag)


def integral_kernel(w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag):
    db = np.where(d_b, 1.0, 0.0)
    df = np.where(d_f, 1.0, 0.0)
    a = np.where(d_b, 0.0, g_b)
    b = np.where(d_f, 0.0, g_f)
    gb = np.where(d_b, g_b, 0.0)
    gf = np.where(d_f, g_f, 0.0)
    z = np.zeros(len(w))
    o = np.ones(len(w))
    o1 = o[:-1]

    m = np.full((4, 4), None)
    if lag == 0:
        m[0, 0] = o
        m[0, 1] = z
        m[0, 2] = z
        m[0, 3] = z
        m[1, 1] = db
        m[1, 2] = z
        m[1, 3] = z
        m[2, 2] = df
        m[2, 3] = z
        m[3, 3] = o
    else:
        m[0, 0] = o1
        m[0, 1] = (a[:-1] + f_b) * db[1:]
        m[0, 2] = a[:-1] * v * df[1:]
        m[0, 3] = a[:-1] * v * b[1:]
        m[1, 1] = db[:-1] * db[1:]
        m[1, 2] = db[:-1] * v * df[1:]
        m[1, 3] = db[:-1] * v * b[1:]
        m[2, 2] = df[:-1] * df[1:]
        m[2, 3] = df[:-1] * (b[1:] + f_f)
        m[3, 3] = o1
        m = moving_matmul(m, lag)

    p = [
        [o, None, None, None],
        [gb, db, None, None],
        [None, None, df, None],
        [None, None, None, o],
    ]
    q = [
        [o, None, None, None],
        [None, db, None, None],
        [None, None, df, gf],
        [None, None, None, o],
    ]

    return _assemble(w, m, p, q, lag)


def _assemble(w, m, p, q, lag):
    assert lag >= 0
    last = -lag if lag > 0 else None
    m = bmap(lambda a: w[:last] * a, asblocks(m))
    p = bmap(lambda a: a[:last], asblocks(p)).T
    q = bmap(lambda a: a[lag:], asblocks(q))
    m = bmatmul(operator.mul, p, bmatmul(operator.mul, m, q))
    _check(w, m, lag)
    return m


def _check(w, m, lag):
    assert np.all(w[len(w) - lag :] == 0.0)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] is not None:
                assert m[i, j].shape == (len(w) - lag,)
