import numba as nb
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from .. import linalg


def forward_committor(basis, weights, in_domain, guess, lag, test_basis=None):
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, d, g, t in zip(
        test_basis, basis, weights, in_domain, guess, lag
    ):
        ai, bi = _forward(x, y, w, d, 0.0, g, t)
        a += ai
        b += bi
    coeffs = linalg.solve(a, b)
    return _transform(coeffs, basis, guess)


def forward_mfpt(basis, weights, in_domain, guess, lag, test_basis=None):
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, d, g, t in zip(
        test_basis, basis, weights, in_domain, guess, lag
    ):
        ai, bi = _forward(x, y, w, d, 1.0, g, t)
        a += ai
        b += bi
    coeffs = linalg.solve(a, b)
    return _transform(coeffs, basis, guess)


def forward_feynman_kac(
    basis, weights, in_domain, function, guess, lag, test_basis=None
):
    if test_basis is None:
        test_basis = basis
    a = 0.0
    b = 0.0
    for x, y, w, d, f, g, t in zip(
        test_basis, basis, weights, in_domain, function, guess, lag
    ):
        ai, bi = _forward(x, y, w, d, f, g, t)
        a += ai
        b += bi
    coeffs = linalg.solve(a, b)
    return _transform(coeffs, basis, guess)


def _forward(x, y, w, d, f, g, t):
    assert np.all(t >= 0)
    iy = np.minimum(np.arange(len(d)) + t, _forward_stop(d))
    assert np.all(iy >= 0) and np.all(iy < len(w))
    intf = np.insert(np.cumsum(np.broadcast_to(f, len(d) - 1)), 0, 0.0)
    integral = intf[iy] - intf
    wx = scipy.sparse.diags(w) @ x
    a = wx.T @ (y[iy] - y)
    b = -wx.T @ (g[iy] - g + integral)
    return a, b


@nb.njit
def _forward_stop(in_domain):
    n = len(in_domain)
    result = np.empty(n, dtype=np.int_)
    stop_time = n - 1
    for t in range(n - 1, -1, -1):
        if not in_domain[t]:
            stop_time = t
        result[t] = stop_time
    return result


def _transform(coeffs, basis, guess):
    return [y @ coeffs + g for y, g in zip(basis, guess)]
