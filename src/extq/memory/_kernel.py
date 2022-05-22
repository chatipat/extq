import numba as nb
import numpy as np

from ..moving_matmul import _mm2
from ..moving_matmul import _mm3
from ..moving_matmul import _mm4
from ..moving_semigroup import moving_semigroup


def reweight_kernel(w, lag):
    m = _reweight_kernel(w, lag)
    return _upper_blocks(m)


@nb.njit
def _reweight_kernel(w, lag):
    _check_lag(w, lag)
    m = np.zeros((1, 1, len(w) - lag))
    for t in range(len(w) - lag):
        m[0, 0, t] = w[t]
    return m


def forward_kernel(w, d_f, f_f, g_f, lag):
    f_f = np.broadcast_to(f_f, len(w) - 1)
    m = _forward_kernel(w, d_f, f_f, g_f, lag)
    return _upper_blocks(m)


@nb.njit
def _forward_kernel(w, d_f, f_f, g_f, lag):
    _check_lag(w, lag)
    assert d_f.ndim == 1 and len(d_f) == len(w)
    assert f_f.ndim == 1 and len(f_f) == len(w) - 1
    assert g_f.ndim == 1 and len(g_f) == len(w)
    if lag == 0:
        m = np.zeros((2, 2, len(w)))
        for t in range(len(w)):
            if d_f[t]:
                m[0, 0, t] = w[t]
                m[0, 1, t] = w[t] * g_f[t]
            m[1, 1, t] = w[t]
        return m
    else:
        k = np.zeros((len(w) - 1, 2, 2))
        for t in range(len(w) - 1):
            if d_f[t]:
                if d_f[t + 1]:
                    k[t, 0, 0] = 1.0
                    k[t, 0, 1] = f_f[t]
                else:
                    k[t, 0, 1] = g_f[t + 1] + f_f[t]
            k[t, 1, 1] = 1.0
        k = moving_semigroup(k, lag, _mm2)
        m = np.zeros((2, 2, len(w) - lag))
        for t in range(len(w) - lag):
            m[0, 0, t] = w[t] * k[t, 0, 0]
            m[0, 1, t] = w[t] * (k[t, 0, 1] + k[t, 0, 0] * g_f[t + lag])
            m[1, 1, t] = w[t]
        return m


def backward_kernel(w, d_b, f_b, g_b, lag):
    f_b = np.broadcast_to(f_b, len(w) - 1)
    m = _backward_kernel(w, d_b, f_b, g_b, lag)
    return _upper_blocks(m)


@nb.njit
def _backward_kernel(w, d_b, f_b, g_b, lag):
    _check_lag(w, lag)
    assert d_b.ndim == 1 and len(d_b) == len(w)
    assert f_b.ndim == 1 and len(f_b) == len(w) - 1
    assert g_b.ndim == 1 and len(g_b) == len(w)
    if lag == 0:
        m = np.zeros((2, 2, len(w)))
        for t in range(len(w)):
            m[0, 0, t] = w[t]
            if d_b[t]:
                m[0, 1, t] = w[t] * g_b[t]
                m[1, 1, t] = w[t]
        return m
    else:
        k = np.zeros((len(w) - 1, 2, 2))
        for t in range(len(w) - 1):
            k[t, 0, 0] = 1.0
            if d_b[t + 1]:
                if d_b[t]:
                    k[t, 0, 1] = f_b[t]
                    k[t, 1, 1] = 1.0
                else:
                    k[t, 0, 1] = g_b[t] + f_b[t]
        k = moving_semigroup(k, lag, _mm2)
        m = np.zeros((2, 2, len(w) - lag))
        for t in range(len(w) - lag):
            m[0, 0, t] = w[t]
            m[0, 1, t] = w[t] * (k[t, 0, 1] + g_b[t] * k[t, 1, 1])
            m[1, 1, t] = w[t] * k[t, 1, 1]
        return m


def reweight_integral_kernel(w, v, lag):
    m = _reweight_integral_kernel(w, v, lag)
    return _upper_blocks(m)


@nb.njit
def _reweight_integral_kernel(w, v, lag):
    _check_lag(w, lag)
    assert v.ndim == 1 and len(v) == len(w) - 1
    if lag == 0:
        m = np.zeros((2, 2, len(w)))
        for t in range(len(w)):
            # upper left block
            m[0, 0, t] = w[t]
            # lower right block
            m[1, 1, t] = w[t]
        return m
    else:
        k = np.zeros((len(w) - 1, 2, 2))
        for t in range(len(w) - 1):
            # upper left block
            k[t, 0, 0] = 1.0
            # upper right block
            k[t, 0, 1] = v[t]
            # lower right block
            k[t, 1, 1] = 1.0
        k = moving_semigroup(k, lag, _mm2)
        m = np.zeros((2, 2, len(w) - lag))
        for t in range(len(w) - lag):
            m[0, 0, t] = w[t]
            m[0, 1, t] = w[t] * k[t, 0, 1]
            m[1, 1, t] = w[t]
        return m


def forward_integral_kernel(w, d_f, v, f_f, g_f, lag):
    f_f = np.broadcast_to(f_f, len(w) - 1)
    m = _forward_integral_kernel(w, d_f, v, f_f, g_f, lag)
    return _upper_blocks(m)


@nb.njit
def _forward_integral_kernel(w, d_f, v, f_f, g_f, lag):
    _check_lag(w, lag)
    assert d_f.ndim == 1 and len(d_f) == len(w)
    assert v.ndim == 1 and len(v) == len(w) - 1
    assert f_f.ndim == 1 and len(f_f) == len(w) - 1
    assert g_f.ndim == 1 and len(g_f) == len(w)
    if lag == 0:
        m = np.zeros((3, 3, len(w)))
        for t in range(len(w)):
            # upper left block
            m[0, 0, t] = w[t]
            # lower right block
            if d_f[t]:
                m[1, 1, t] = w[t]
                m[1, 2, t] = w[t] * g_f[t]
            m[2, 2, t] = w[t]
        return m
    else:
        k = np.zeros((len(w) - 1, 3, 3))
        for t in range(len(w) - 1):
            # upper left block
            k[t, 0, 0] = 1.0
            # upper right block
            if d_f[t + 1]:
                k[t, 0, 1] = v[t]
            else:
                k[t, 0, 2] = v[t] * g_f[t + 1]
            # lower right block
            if d_f[t]:
                if d_f[t + 1]:
                    k[t, 1, 1] = 1.0
                    k[t, 1, 2] = f_f[t]
                else:
                    k[t, 1, 2] = g_f[t + 1] + f_f[t]
            k[t, 2, 2] = 1.0
        k = moving_semigroup(k, lag, _mm3)
        m = np.zeros((3, 3, len(w) - lag))
        for t in range(len(w) - lag):
            # upper left block
            m[0, 0, t] = w[t]
            # upper right block
            m[0, 1, t] = w[t] * k[t, 0, 1]
            m[0, 2, t] = w[t] * (k[t, 0, 2] + k[t, 0, 1] * g_f[t + lag])
            # lower right block
            m[1, 1, t] = w[t] * k[t, 1, 1]
            m[1, 2, t] = w[t] * (k[t, 1, 2] + k[t, 1, 1] * g_f[t + lag])
            m[2, 2, t] = w[t]
        return m


def backward_integral_kernel(w, d_b, v, f_b, g_b, lag):
    f_b = np.broadcast_to(f_b, len(w) - 1)
    m = _backward_integral_kernel(w, d_b, v, f_b, g_b, lag)
    return _upper_blocks(m)


@nb.njit
def _backward_integral_kernel(w, d_b, v, f_b, g_b, lag):
    _check_lag(w, lag)
    assert d_b.ndim == 1 and len(d_b) == len(w)
    assert v.ndim == 1 and len(v) == len(w) - 1
    assert f_b.ndim == 1 and len(f_b) == len(w) - 1
    assert g_b.ndim == 1 and len(g_b) == len(w)
    if lag == 0:
        m = np.zeros((3, 3, len(w)))
        for t in range(len(w)):
            # upper left block
            m[0, 0, t] = w[t]
            if d_b[t]:
                m[0, 1, t] = w[t] * g_b[t]
                m[1, 1, t] = w[t]
            # lower right block
            m[2, 2, t] = w[t]
        return m
    else:
        k = np.zeros((len(w) - 1, 3, 3))
        for t in range(len(w) - 1):
            # upper left block
            k[t, 0, 0] = 1.0
            if d_b[t + 1]:
                if d_b[t]:
                    k[t, 0, 1] = f_b[t]
                    k[t, 1, 1] = 1.0
                else:
                    k[t, 0, 1] = g_b[t] + f_b[t]
            # upper right block
            if d_b[t]:
                k[t, 1, 2] = v[t]
            else:
                k[t, 0, 2] = g_b[t] * v[t]
            # lower right block
            k[t, 2, 2] = 1.0
        k = moving_semigroup(k, lag, _mm3)
        m = np.zeros((3, 3, len(w) - lag))
        for t in range(len(w) - lag):
            # upper left block
            m[0, 0, t] = w[t]
            m[0, 1, t] = w[t] * (k[t, 0, 1] + g_b[t] * k[t, 1, 1])
            m[1, 1, t] = w[t] * k[t, 1, 1]
            # upper right block
            m[0, 2, t] = w[t] * (k[t, 0, 2] + g_b[t] * k[t, 1, 2])
            m[1, 2, t] = w[t] * k[t, 1, 2]
            # lower right block
            m[2, 2, t] = w[t]
        return m


def integral_kernel(w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag):
    f_f = np.broadcast_to(f_f, len(w) - 1)
    f_b = np.broadcast_to(f_b, len(w) - 1)
    m = _integral_kernel(w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag)
    return _upper_blocks(m)


@nb.njit
def _integral_kernel(w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag):
    _check_lag(w, lag)
    assert d_b.ndim == 1 and len(d_b) == len(w)
    assert d_f.ndim == 1 and len(d_f) == len(w)
    assert v.ndim == 1 and len(v) == len(w) - 1
    assert f_b.ndim == 1 and len(f_b) == len(w) - 1
    assert f_f.ndim == 1 and len(f_f) == len(w) - 1
    assert g_b.ndim == 1 and len(g_b) == len(w)
    assert g_f.ndim == 1 and len(g_f) == len(w)
    if lag == 0:
        m = np.zeros((4, 4, len(w)))
        for t in range(len(w)):
            # upper left block
            m[0, 0, t] = w[t]
            if d_b[t]:
                m[0, 1, t] = w[t] * g_b[t]
                m[1, 1, t] = w[t]
            # lower right block
            if d_f[t]:
                m[2, 2, t] = w[t]
                m[2, 3, t] = w[t] * g_f[t]
            m[3, 3, t] = w[t]
        return m
    else:
        k = np.zeros((len(w) - 1, 4, 4))
        for t in range(len(w) - 1):
            # upper left block
            k[t, 0, 0] = 1.0
            if d_b[t + 1]:
                if d_b[t]:
                    k[t, 0, 1] = f_b[t]
                    k[t, 1, 1] = 1.0
                else:
                    k[t, 0, 1] = g_b[t] + f_b[t]
            # upper right block
            if d_b[t]:
                if d_f[t + 1]:
                    k[t, 1, 2] = v[t]
                else:
                    k[t, 1, 3] = v[t] * g_f[t + 1]
            else:
                if d_f[t + 1]:
                    k[t, 0, 2] = g_b[t] * v[t]
                else:
                    k[t, 0, 3] = g_b[t] * v[t] * g_f[t + 1]
            # lower right block
            if d_f[t]:
                if d_f[t + 1]:
                    k[t, 2, 2] = 1.0
                    k[t, 2, 3] = f_f[t]
                else:
                    k[t, 2, 3] = g_f[t + 1] + f_f[t]
            k[t, 3, 3] = 1.0
        k = moving_semigroup(k, lag, _mm4)
        m = np.zeros((4, 4, len(w) - lag))
        for t in range(len(w) - lag):
            # upper left block
            m[0, 0, t] = w[t]
            m[0, 1, t] = w[t] * (k[t, 0, 1] + g_b[t] * k[t, 1, 1])
            m[1, 1, t] = w[t] * k[t, 1, 1]
            # upper right block
            m[0, 2, t] = w[t] * (k[t, 0, 2] + g_b[t] * k[t, 1, 2])
            m[0, 3, t] = w[t] * (
                k[t, 0, 3]
                + g_b[t] * k[t, 1, 3]
                + k[t, 0, 2] * g_f[t + lag]
                + g_b[t] * k[t, 1, 2] * g_f[t + lag]
            )
            m[1, 2, t] = w[t] * k[t, 1, 2]
            m[1, 3, t] = w[t] * (k[t, 1, 3] + k[t, 1, 2] * g_f[t + lag])
            # lower right block
            m[2, 2, t] = w[t] * k[t, 2, 2]
            m[2, 3, t] = w[t] * (k[t, 2, 3] + k[t, 2, 2] * g_f[t + lag])
            m[3, 3, t] = w[t]
        return m


@nb.njit
def _check_lag(w, lag):
    assert lag >= 0
    assert w.ndim == 1
    assert len(w) > lag
    for t in range(len(w) - lag, len(w)):
        assert w[t] == 0.0


def _upper_blocks(m):
    n = len(m)
    result = np.full((n, n), None)
    for i in range(n):
        for j in range(i, n):
            result[i, j] = m[i, j]
    return result
