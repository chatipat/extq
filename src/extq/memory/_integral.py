import numpy as np

from .. import linalg
from ..integral import integral_coeffs as _integral_coeffs
from . import _kernel
from . import _matrix
from . import _memory


def reweight_integral_coeffs(basis, weights, obslag, lag, mem=0, test=None):
    left = _reweight(basis, weights, lag, mem=mem, test=test)
    right = _constant(weights)
    return _combine(left, right, lag, obslag, mem=mem)


def forward_committor_integral_coeffs(
    w_basis,
    basis,
    weights,
    domain,
    guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    left = _reweight(w_basis, weights, lag, mem=mem, test=w_test)
    right = _forward_committor(
        basis, weights, domain, guess, lag, mem=mem, test=test
    )
    return _combine(left, right, lag, obslag, mem=mem)


def forward_mfpt_integral_coeffs(
    w_basis,
    basis,
    weights,
    domain,
    guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    left = _reweight(w_basis, weights, lag, mem=mem, test=w_test)
    right = _forward_mfpt(
        basis, weights, domain, guess, lag, mem=mem, test=test
    )
    return _combine(left, right, lag, obslag, mem=mem)


def forward_feynman_kac_integral_coeffs(
    w_basis,
    basis,
    weights,
    domain,
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
        basis, weights, domain, function, guess, lag, mem=mem, test=test
    )
    return _combine(left, right, lag, obslag, mem=mem)


def backward_committor_integral_coeffs(
    w_basis,
    basis,
    weights,
    domain,
    guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    left = _backward_committor(
        w_basis,
        basis,
        weights,
        domain,
        guess,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )
    right = _constant(weights)
    return _combine(left, right, lag, obslag, mem=mem)


def backward_mfpt_integral_coeffs(
    w_basis,
    basis,
    weights,
    domain,
    guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    left = _backward_mfpt(
        w_basis,
        basis,
        weights,
        domain,
        guess,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )
    right = _constant(weights)
    return _combine(left, right, lag, obslag, mem=mem)


def backward_feynman_kac_integral_coeffs(
    w_basis,
    basis,
    weights,
    domain,
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
        domain,
        function,
        guess,
        lag,
        mem=mem,
        w_test=w_test,
        test=test,
    )
    right = _constant(weights)
    return _combine(left, right, lag, obslag, mem=mem)


def tpt_integral_coeffs(
    w_basis,
    b_basis,
    f_basis,
    weights,
    domain,
    b_guess,
    f_guess,
    obslag,
    lag,
    mem=0,
    w_test=None,
    b_test=None,
    f_test=None,
):
    left = _backward_committor(
        w_basis,
        b_basis,
        weights,
        domain,
        b_guess,
        lag,
        mem=mem,
        w_test=w_test,
        test=b_test,
    )
    right = _forward_committor(
        f_basis, weights, domain, f_guess, lag, mem=mem, test=f_test
    )
    return _combine(left, right, lag, obslag, mem=mem)


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
    return _combine(left, right, lag, obslag, mem=mem)


def _constant(weights):
    for w in weights:

        k = np.ones((len(w) - 1, 1, 1))

        u = np.empty((len(w), 1, 2))
        u[:, 0, 0] = 1.0
        u[:, 0, 1] = -1.0

        m = np.ones((len(w), 1))

        yield k, u, m


def _reweight(basis, weights, lag, mem=0, test=None):
    mats = [
        _matrix.reweight_matrix(basis, weights, t, test=test)
        for t in _memlags(lag, mem)
    ]
    v = _left_coeffs(mats)
    for x_w, w in zip(basis, weights):

        k = np.ones((len(w) - 1, 1, 1))

        u = np.empty((len(w), 1, mem + 2))
        u[:, 0] = w[:, None] * (v[0] + x_w @ v[1:])

        m = np.ones((len(w), 1))

        yield k, u, m


def _forward_committor(basis, weights, domain, guess, lag, mem=0, test=None):
    mats = [
        _matrix.forward_committor_matrix(
            basis, weights, domain, guess, t, test=test
        )
        for t in _memlags(lag, mem)
    ]
    function = [np.zeros(len(w) - 1) for w in weights]
    return _forward(mats, basis, domain, function, guess)


def _forward_mfpt(basis, weights, domain, guess, lag, mem=0, test=None):
    mats = [
        _matrix.forward_committor_matrix(
            basis, weights, domain, guess, t, test=test
        )
        for t in _memlags(lag, mem)
    ]
    function = [np.ones(len(w) - 1) for w in weights]
    return _forward(mats, basis, domain, function, guess)


def _forward_feynman_kac(
    basis, weights, domain, function, guess, lag, mem=0, test=None
):
    mats = [
        _matrix.forward_feynman_kac_matrix(
            basis, weights, domain, function, guess, t, test=test
        )
        for t in _memlags(lag, mem)
    ]
    return _forward(mats, basis, domain, function, guess)


def _forward(mats, basis, domain, function, guess):
    v = _right_coeffs(mats)
    for y_f, d_f, f_f, g_f in zip(basis, domain, function, guess):

        k = _kernel.forward_transitions(d_f, f_f, g_f)

        u = np.empty((len(d_f), 2, v.shape[-1]))
        u[:, 1] = v[-1]
        u[:, 0] = np.where(
            d_f[:, None], g_f[:, None] * u[:, 1] + y_f @ v[:-1], 0.0
        )

        m = np.empty((len(d_f), 2))
        m[:, 1] = np.where(d_f, 0.0, g_f)
        m[:, 0] = np.where(d_f, 1.0, 0.0)

        yield k, u, m


def _backward_committor(
    w_basis,
    basis,
    weights,
    domain,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = [
        _matrix.backward_committor_matrix(
            w_basis,
            basis,
            weights,
            domain,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
        for t in _memlags(lag, mem)
    ]
    function = [np.zeros(len(w) - 1) for w in weights]
    return _backward(mats, w_basis, basis, weights, domain, function, guess)


def _backward_mfpt(
    w_basis,
    basis,
    weights,
    domain,
    guess,
    lag,
    mem=0,
    w_test=None,
    test=None,
):
    mats = [
        _matrix.backward_mfpt_matrix(
            w_basis,
            basis,
            weights,
            domain,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
        for t in _memlags(lag, mem)
    ]
    function = [np.ones(len(w) - 1) for w in weights]
    return _backward(mats, w_basis, basis, weights, domain, function, guess)


def _backward_feynman_kac(
    w_basis,
    basis,
    weights,
    domain,
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
            domain,
            function,
            guess,
            t,
            w_test=w_test,
            test=test,
        )
        for t in _memlags(lag, mem)
    ]
    return _backward(mats, w_basis, basis, weights, domain, function, guess)


def _backward(mats, w_basis, basis, weights, domain, function, guess):
    v = _left_coeffs(mats)
    for x_w, x_b, w, d_b, f_b, g_b in zip(
        w_basis, basis, weights, domain, function, guess
    ):

        k = _kernel.backward_transitions(d_b, f_b, g_b)

        n = x_w.shape[1] + 1
        u = np.empty((len(d_b), 2, v.shape[-1]))
        u[:, 0] = w[:, None] * (v[0] + x_w @ v[1:n])
        u[:, 1] = np.where(
            d_b[:, None],
            g_b[:, None] * u[:, 0] + w[:, None] * (x_b @ v[n:]),
            0.0,
        )

        m = np.empty((len(d_b), 2))
        m[:, 0] = np.where(d_b, 0.0, g_b)
        m[:, 1] = np.where(d_b, 1.0, 0.0)

        yield k, u, m


def _left_coeffs(mats):
    mems = _memory.memory(mats)
    eye = _memory.identity(mats, mems)
    gen = _memory.generator(mats, mems)
    gen = linalg.solve(eye, gen)
    coeffs = np.concatenate(
        [[1.0], linalg.solve(gen.T[1:, 1:], -gen.T[1:, 0])]
    )
    coeffs = linalg.solve(eye.T, coeffs)
    neginv = -linalg.inv(mats[0])
    v = np.empty((len(coeffs), len(mems) + 2))
    v[:, 0] = coeffs
    v[:, 1] = (coeffs @ mats[1]) @ neginv
    for t in range(len(mems)):
        v[:, t + 2] = (coeffs @ mems[t]) @ neginv
    return v


def _right_coeffs(mats):
    mems = _memory.memory(mats)
    gen = _memory.generator(mats, mems)
    coeffs = np.concatenate(
        [linalg.solve(gen[:-1, :-1], -gen[:-1, -1]), [1.0]]
    )
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
    for (k_b, u_b, m_b), (k_f, u_f, m_f) in zip(left, right):
        a = _integral_memory_coeffs(k_b, k_f, u_b, u_f, lag, mem=mem)
        if obslag == 0:
            c = np.zeros(len(a) + 1)
            c[:-1] += np.einsum("tik,ti,tj,tjk->t", a, m_b[:-1], m_f[:-1], k_f)
            c[1:] += np.einsum("tik,tij,tj,tk->t", a, k_b, m_b[1:], m_f[1:])
            c /= 2.0 * dlag
        else:
            c = np.einsum("tij,ti,tj->t", a, m_b[:-1], m_f[1:]) / dlag
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
