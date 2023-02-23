import numpy as np

from .. import linalg
from ..integral import integral_coeffs as _integral_coeffs
from . import _dga, _matrix, _memory
from ._transitions import (
    backward_feynman_kac_transitions,
    backward_feynman_kac_unhomogenize,
    constant_transitions,
    forward_feynman_kac_transitions,
    forward_feynman_kac_unhomogenize,
)

__all__ = [
    "reweight_integral_coeffs",
    "forward_committor_integral_coeffs",
    "forward_mfpt_integral_coeffs",
    "forward_feynman_kac_integral_coeffs",
    "backward_committor_integral_coeffs",
    "backward_mfpt_integral_coeffs",
    "backward_feynman_kac_integral_coeffs",
    "tpt_integral_coeffs",
    "integral_coeffs",
]


def reweight_integral_coeffs(basis, weights, obslag, lag, mem=0, test=None):
    left = _reweight(basis, weights, lag, mem=mem, test=test)
    right = _constant(weights)
    return _combine(left, right, obslag, lag, mem=mem)


def forward_committor_integral_coeffs(
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    return forward_feynman_kac_integral_coeffs(
        w_basis,
        basis,
        weights,
        in_domain,
        np.zeros(len(weights)),
        guess,
        obslag,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )


def forward_mfpt_integral_coeffs(
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    return forward_feynman_kac_integral_coeffs(
        w_basis,
        basis,
        weights,
        in_domain,
        np.ones(len(weights)),
        guess,
        obslag,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )


def forward_feynman_kac_integral_coeffs(
    w_basis,
    basis,
    weights,
    in_domain,
    function,
    guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    left = _reweight(w_basis, weights, lag, mem=mem, test=w_test)
    right = _forward_feynman_kac(
        basis, weights, in_domain, function, guess, lag, mem=mem, test=test
    )
    return _combine(left, right, obslag, lag, mem=mem)


def backward_committor_integral_coeffs(
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    return backward_feynman_kac_integral_coeffs(
        w_basis,
        basis,
        weights,
        in_domain,
        np.zeros(len(weights)),
        guess,
        obslag,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )


def backward_mfpt_integral_coeffs(
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    return backward_feynman_kac_integral_coeffs(
        w_basis,
        basis,
        weights,
        in_domain,
        np.ones(len(weights)),
        guess,
        obslag,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )


def backward_feynman_kac_integral_coeffs(
    w_basis,
    basis,
    weights,
    in_domain,
    function,
    guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    left = _backward_feynman_kac(
        w_basis,
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )
    right = _constant(weights)
    return _combine(left, right, obslag, lag, mem=mem)


def tpt_integral_coeffs(
    w_basis,
    b_basis,
    f_basis,
    weights,
    in_domain,
    b_guess,
    f_guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    b_test=None,
    f_test=None,
):
    return integral_coeffs(
        w_basis,
        b_basis,
        f_basis,
        weights,
        in_domain,
        in_domain,
        np.zeros(len(weights)),
        np.zeros(len(weights)),
        b_guess,
        f_guess,
        obslag,
        lag,
        mem=mem,
        w_test=w_test,
        b_test=b_test,
        f_test=f_test,
    )


def integral_coeffs(
    w_basis,
    b_basis,
    f_basis,
    weights,
    b_domain,
    f_domain,
    b_function,
    f_function,
    b_guess,
    f_guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    b_test=None,
    f_test=None,
):
    left = _backward_feynman_kac(
        w_basis,
        b_basis,
        weights,
        b_domain,
        b_function,
        b_guess,
        lag,
        mem=mem,
        w_test=w_test,
        test=b_test,
    )
    right = _forward_feynman_kac(
        f_basis,
        weights,
        f_domain,
        f_function,
        f_guess,
        lag,
        mem=mem,
        test=f_test,
    )
    return _combine(left, right, obslag, lag, mem=mem)


def _constant(weights):
    out = []
    for w in weights:
        k = constant_transitions(len(w), 1)

        m = np.ones((len(w), 1))

        u = np.empty((len(w), 1, 2))
        u[:, 0, 0] = 1.0
        u[:, 0, 1] = -1.0

        out.append((k, m, u))
    return out


def _reweight(basis, weights, lag, mem=0, test=None):
    mats = [
        _matrix.reweight_matrix(basis, weights, t, test=test)
        for t in _memlags(lag, mem)
    ]
    v = _left_coeffs(mats)
    out = []
    for x_w, w in zip(basis, weights):
        k = constant_transitions(len(w), 1)

        m = np.ones((len(w), 1))

        u = np.empty((len(w), 1, mem + 2))
        u[:, 0] = w[:, None] * (v[-1] + x_w @ v[:-1])

        out.append((k, m, u))
    return out


def _forward_feynman_kac(
    basis, weights, in_domain, function, guess, lag, mem=0, test=None
):
    mats = [
        _matrix.forward_feynman_kac_matrix(
            basis, weights, in_domain, function, guess, t, test=test
        )
        for t in _memlags(lag, mem)
    ]
    v = _right_coeffs(mats)
    out = []
    for y_f, d_f, f_f, g_f in zip(basis, in_domain, function, guess):
        k = forward_feynman_kac_transitions(d_f, f_f, g_f, 1)
        m = forward_feynman_kac_unhomogenize(d_f, g_f)

        u = np.empty((len(d_f), 2, v.shape[-1]))
        u[:, 0] = y_f @ v[:-1]
        u[:, 1] = v[-1]

        out.append((k, m, u))
    return out


def _backward_feynman_kac(
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
    v = _left_coeffs(mats)
    out = []
    for x_w, x_b, w, d_b, f_b, g_b in zip(
        w_basis, basis, weights, in_domain, function, guess
    ):
        k = backward_feynman_kac_transitions(d_b, f_b, g_b, 1)
        m = backward_feynman_kac_unhomogenize(d_b, f_b)

        n = x_b.shape[1]
        u = np.empty((len(d_b), 2, v.shape[-1]))
        u[:, 0] = w[:, None] * (x_b @ v[:n])
        u[:, 1] = w[:, None] * (v[-1] + x_w @ v[n:-1])

        out.append((k, m, u))
    return out


def _left_coeffs(mats):
    mems = _memory.memory(mats)
    coeffs = _dga.backward_coeffs(mats, mems)
    neginv = -linalg.inv(mats[0])
    v = np.empty((len(coeffs), len(mems) + 2))
    v[:, 0] = coeffs
    v[:, 1] = (coeffs @ mats[1]) @ neginv
    for t in range(len(mems)):
        v[:, t + 2] = (coeffs @ mems[t]) @ neginv
    return v


def _right_coeffs(mats):
    mems = _memory.memory(mats)
    coeffs = _dga.forward_coeffs(mats, mems)
    neginv = -linalg.inv(mats[0])
    v = np.empty((len(coeffs), len(mems) + 2))
    v[:, 0] = coeffs
    v[:, 1] = neginv @ (mats[1] @ coeffs)
    for t in range(len(mems)):
        v[:, t + 2] = neginv @ (mems[t] @ coeffs)
    return v


def _combine(left, right, obslag, lag, mem=0):
    assert obslag in [0, 1]
    assert lag > 0
    assert mem >= 0
    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    out = []
    for (k_b, m_b, u_b), (k_f, m_f, u_f) in zip(left, right):
        nf, ni, nk = u_b.shape
        _, nj, nl = u_f.shape
        assert m_b.shape == (nf, ni)
        assert m_f.shape == (nf, nj)
        assert u_b.shape == (nf, ni, nk)
        assert u_f.shape == (nf, nj, nl)
        assert k_b.shape == (nf - 1, ni, ni)
        assert k_f.shape == (nf - 1, nj, nj)
        assert nf > lag
        end = nf - lag
        a = np.zeros((nf - 1, ni, nj))
        u_f_sum = np.cumsum(u_f, axis=-1)
        for n in range(mem + 1):
            s = (n + 1) * dlag
            kend = end + s - 1
            # u[t,i,j] = sum_{k+l <= mem-n} u_b[t,i,k] * u_f[t+s,j,l]
            u = np.zeros((end, ni, nj))
            for k in range(min(mem - n, nk - 1) + 1):
                u += (
                    u_b[:end, :, None, k]
                    * u_f_sum[s : end + s, None, :, min(mem - n - k, nl - 1)]
                )
            a[:kend] += _integral_coeffs(u, k_b[:kend], k_f[:kend], 1, s)
        if obslag == 0:
            c = np.zeros((nf, ni, nj))
            c[:-1] += a @ np.swapaxes(k_f, 1, 2)
            c[1:] += np.swapaxes(k_b, 1, 2) @ a
            a = 0.5 * c
            a = np.sum(m_b[:, :, None] * a * m_f[:, None, :], axis=(1, 2))
        else:
            a = np.sum(m_b[:-1, :, None] * a * m_f[1:, None, :], axis=(1, 2))
        out.append(a / dlag)
    return out


def _memlags(lag, mem):
    assert lag % (mem + 1) == 0
    return np.arange(0, lag + 1, lag // (mem + 1))
