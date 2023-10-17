import numpy as np

from .. import linalg
from ..fdm import augment_generator
from ..fdm.dga import _to_flat, feynman_kac_kernel
from ._dga import _dga_mem

__all__ = [
    "reweight",
    "reweight_matrices",
    "reweight_projection",
    "reweight_solution",
    "forward_committor",
    "forward_mfpt",
    "forward_feynman_kac",
    "forward_feynman_kac_matrices",
    "forward_feynman_kac_projection",
    "forward_feynman_kac_solution",
    "backward_committor",
    "backward_mfpt",
    "backward_feynman_kac",
    "backward_feynman_kac_matrices",
    "backward_feynman_kac_projection",
    "backward_feynman_kac_solution",
]


def reweight(
    generator,
    basis,
    weights,
    lag,
    mem,
    test_basis=None,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    assert (
        return_projection or return_solution or return_coef or return_mem_coef
    )
    a, b, c0 = reweight_matrices(
        generator, basis, weights, lag, mem, test_basis=test_basis
    )
    coef, mem_coef = _dga_mem(a, b, c0)
    out = []
    if return_projection:
        out.append(reweight_projection(basis, weights, coef))
    if return_solution:
        out.append(
            reweight_solution(
                generator, basis, weights, lag, mem, coef, mem_coef
            )
        )
    if return_coef:
        out.append(coef)
    if return_mem_coef:
        out.append(mem_coef)
    if len(out) == 1:
        out = out[0]
    return out


def reweight_matrices(generator, basis, weights, lag, mem, test_basis=None):
    if test_basis is None:
        test_basis = basis
    _, [x, y], [w] = _to_flat([test_basis, basis], [weights])
    _, k = y.shape
    L = generator.T
    a = np.zeros((mem + 1, k, k))
    b = np.zeros((mem + 1, k))
    wy = linalg.scale_rows(w, y)
    for m in range(mem + 1):
        t = lag * ((m + 1) / (mem + 1))
        wyt = linalg.expm_multiply(L * t, wy)
        wt = linalg.expm_multiply(L * t, w)
        a[m] = x.T @ (wyt - wy)
        b[m] = x.T @ (wt - w)
    c0 = x.T @ wy
    return a, b, c0


def reweight_projection(basis, weights, coef):
    shape, [y], [w] = _to_flat([basis], [weights])
    u = w * (y @ coef + 1.0)
    return u.reshape(shape)


def reweight_solution(generator, basis, weights, lag, mem, coef, mem_coef):
    shape, [y], [w] = _to_flat([basis], [weights])
    dlag = lag / (mem + 1)
    L = generator.T
    u = w * (y @ coef + 1.0)
    for m in range(mem):
        u = linalg.expm_multiply(L * dlag, u)
        u -= w * (y @ mem_coef[m])
    u = linalg.expm_multiply(L * dlag, u)
    return u.reshape(shape)


def forward_committor(
    generator,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    return forward_feynman_kac(
        generator,
        basis,
        weights,
        in_domain,
        0.0,
        guess,
        lag,
        mem,
        test_basis,
        return_projection,
        return_solution,
        return_coef,
        return_mem_coef,
    )


def forward_mfpt(
    generator,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    return forward_feynman_kac(
        generator,
        basis,
        weights,
        in_domain,
        1.0,
        guess,
        lag,
        mem,
        test_basis,
        return_projection,
        return_solution,
        return_coef,
        return_mem_coef,
    )


def forward_feynman_kac(
    generator,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    test_basis=None,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    assert (
        return_projection or return_solution or return_coef or return_mem_coef
    )
    a, b, c0 = forward_feynman_kac_matrices(
        generator,
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem,
        test_basis=test_basis,
    )
    coef, mem_coef = _dga_mem(a, b, c0)
    out = []
    if return_projection:
        out.append(forward_feynman_kac_projection(basis, guess, coef))
    if return_solution:
        out.append(
            forward_feynman_kac_solution(
                generator,
                basis,
                in_domain,
                function,
                guess,
                lag,
                mem,
                coef,
                mem_coef,
            )
        )
    if return_coef:
        out.append(coef)
    if return_mem_coef:
        out.append(mem_coef)
    if len(out) == 1:
        out = out[0]
    return out


def forward_feynman_kac_matrices(
    generator,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    test_basis=None,
):
    if test_basis is None:
        test_basis = basis
    _, [x, y], [w, d, f, g] = _to_flat(
        [test_basis, basis], [weights, in_domain, function, guess]
    )
    n, k = y.shape
    K = feynman_kac_kernel(generator, d, f, g)
    _, L = augment_generator(generator, *K)
    L1 = L[:n, :][:, :n]
    L2 = L[n:, :][:, n:]
    a = np.empty((mem + 1, k, k))
    b = np.empty((mem + 1, k))
    xw = linalg.scale_rows(w, x).T
    z = linalg.expm_multiply(L2 * lag, np.ones(n))
    y0 = linalg.scale_rows(z, y)
    for m in range(mem + 1):
        t = lag * ((m + 1) / (mem + 1))
        z = linalg.expm_multiply(L2 * (lag - t), np.ones(n))
        yt = linalg.expm_multiply(L1 * t, linalg.scale_rows(z, y))
        r = linalg.expm_multiply(L * t, np.concatenate([np.zeros(n), z]))[:n]
        a[m] = xw @ (yt - y0)
        b[m] = xw @ r
    c0 = xw @ y0
    return a, b, c0


def forward_feynman_kac_projection(basis, guess, coef):
    shape, [y], [g] = _to_flat([basis], [guess])
    u = y @ coef + g
    return u.reshape(shape)


def forward_feynman_kac_solution(
    generator, basis, in_domain, function, guess, lag, mem, coef, mem_coef
):
    shape, [y], [d, f, g] = _to_flat([basis], [in_domain, function, guess])
    n = len(g)
    dlag = lag / (mem + 1)
    K = feynman_kac_kernel(generator, d, f, g)
    _, L = augment_generator(generator, *K)
    u = np.concatenate([y @ coef, np.ones(n)])
    for m in range(mem):
        u = linalg.expm_multiply(L * dlag, u)
        u[:n] -= (y @ mem_coef[m]) * u[n:]
    u = linalg.expm_multiply(L * dlag, u)
    u = u[:n] + g * u[n:]
    return u.reshape(shape)


def backward_committor(
    generator,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    return backward_feynman_kac(
        generator,
        basis,
        weights,
        in_domain,
        0.0,
        guess,
        lag,
        mem,
        test_basis,
        return_projection,
        return_solution,
        return_coef,
        return_mem_coef,
    )


def backward_mfpt(
    generator,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    return backward_feynman_kac(
        generator,
        basis,
        weights,
        in_domain,
        1.0,
        guess,
        lag,
        mem,
        test_basis,
        return_projection,
        return_solution,
        return_coef,
        return_mem_coef,
    )


def backward_feynman_kac(
    generator,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    test_basis=None,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    assert (
        return_projection or return_solution or return_coef or return_mem_coef
    )
    a, b, c0 = backward_feynman_kac_matrices(
        generator,
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem,
        test_basis=test_basis,
    )
    coef, mem_coef = _dga_mem(a, b, c0)
    out = []
    if return_projection:
        out.append(backward_feynman_kac_projection(basis, guess, coef))
    if return_solution:
        out.append(
            backward_feynman_kac_solution(
                generator,
                basis,
                weights,
                in_domain,
                function,
                guess,
                lag,
                mem,
                coef,
                mem_coef,
            )
        )
    if return_coef:
        out.append(coef)
    if return_mem_coef:
        out.append(mem_coef)
    if len(out) == 1:
        out = out[0]
    return out


def backward_feynman_kac_matrices(
    generator,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    test_basis=None,
):
    if test_basis is None:
        test_basis = basis
    _, [x, y], [w, d, f, g] = _to_flat(
        [test_basis, basis], [weights, in_domain, function, guess]
    )
    genadj = generator.T.multiply(1.0 / w[:, None]).multiply(w[None, :])
    return forward_feynman_kac_matrices(
        genadj, y, w, d, f, g, lag, mem, test_basis=x
    )


def backward_feynman_kac_projection(basis, guess, coef):
    return forward_feynman_kac_projection(basis, guess, coef)


def backward_feynman_kac_solution(
    generator,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    coef,
    mem_coef,
):
    shape, [y], [w, d, f, g] = _to_flat(
        [basis], [weights, in_domain, function, guess]
    )
    genadj = generator.T.multiply(1.0 / w[:, None]).multiply(w[None, :])
    return forward_feynman_kac_solution(
        genadj, y, d, f, g, lag, mem, coef, mem_coef
    ).reshape(shape)
