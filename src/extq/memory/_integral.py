import numpy as np

from .. import linalg
from ..integral import integral_coeffs as _integral_coeffs
from . import _dga
from . import _kernel
from . import _matrix
from . import _memory

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
    for w in weights:

        k = np.ones((len(w) - 1, 1, 1))

        u = np.empty((len(w), 1, 2))
        u[:, 0, 0] = 1.0
        u[:, 0, 1] = -1.0

        yield k, u


def _reweight(basis, weights, lag, mem=0, test=None):
    mats = [
        _matrix.reweight_matrix(basis, weights, t, test=test)
        for t in _memlags(lag, mem)
    ]
    v = _left_coeffs(mats)
    for x_w, w in zip(basis, weights):

        k = np.ones((len(w) - 1, 1, 1))

        u = np.empty((len(w), 1, mem + 2))
        u[:, 0] = w[:, None] * (v[-1] + x_w @ v[:-1])

        yield k, u


def _forward_feynman_kac(
    basis, weights, in_domain, function, guess, lag, mem=0, test=None
):
    mats = [
        _matrix.forward_feynman_kac_matrix(
            basis, weights, in_domain, function, guess, t, test=test
        )
        for t in _memlags(lag, mem)
    ]
    return _forward(mats, basis, in_domain, function, guess)


def _forward(mats, basis, in_domain, function, guess):
    v = _right_coeffs(mats)
    for y_f, d_f, f_f, g_f in zip(basis, in_domain, function, guess):
        f_f = np.broadcast_to(f_f, len(d_f) - 1)

        k = _kernel.forward_transitions(d_f, f_f, g_f, 1)

        u = np.empty((len(d_f), 2, v.shape[-1]))
        u[:, 1] = v[-1]
        u[:, 0] = g_f[:, None] * u[:, 1] + d_f[:, None] * (y_f @ v[:-1])

        yield k, u


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
    return _backward(mats, w_basis, basis, weights, in_domain, function, guess)


def _backward(mats, w_basis, basis, weights, in_domain, function, guess):
    v = _left_coeffs(mats)
    for x_w, x_b, w, d_b, f_b, g_b in zip(
        w_basis, basis, weights, in_domain, function, guess
    ):
        f_b = np.broadcast_to(f_b, len(d_b) - 1)

        k = _kernel.backward_transitions(d_b, f_b, g_b, 1)

        n = x_b.shape[1]
        u = np.empty((len(d_b), 2, v.shape[-1]))
        u[:, 0] = w[:, None] * (v[-1] + x_w @ v[n:-1])
        u[:, 1] = g_b[:, None] * u[:, 0] + (d_b * w)[:, None] * (x_b @ v[:n])

        yield k, u


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
    dlag = lag // (mem + 1)
    out = []
    for (k_b, u_b), (k_f, u_f) in zip(left, right):
        a = _integral_memory_coeffs(k_b, k_f, u_b, u_f, lag, mem=mem)
        if obslag == 0:
            c = np.zeros(len(a) + 1)
            c[:-1] += np.einsum("tik,tjk->tij", a, k_f)[:, 0, 0]
            c[1:] += np.einsum("tkj,tki->tij", a, k_b)[:, 0, 0]
            c /= 2.0 * dlag
        else:
            c = a[:, 0, 0] / dlag
        out.append(c)
    return out


def _integral_memory_coeffs(k_b, k_f, u_b, u_f, lag, mem=0):
    dlag = lag // (mem + 1)
    out = 0.0
    for n in range(mem + 1):
        s = (n + 1) * dlag
        k = np.arange(u_b.shape[-1])
        l = np.arange(u_f.shape[-1])
        mask = (k[:, None] + l[None, :] <= mem - n).astype(float)
        c = np.einsum("kl,tik,tjl->tij", mask, u_b[:-s], u_f[s:])
        out += _integral_coeffs(c, k_b, k_f, 1, s)
    return out


def _memlags(lag, mem):
    assert lag % (mem + 1) == 0
    return np.arange(0, lag + 1, lag // (mem + 1))
