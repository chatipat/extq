"""Finite difference reference calculation for DGA with memory."""

import numpy as np
import scipy as sp

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
    "solve",
]


def reweight(
    generator,
    basis,
    weights,
    lag,
    mem,
    test_basis=None,
    *,
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
    coef, mem_coef = solve(a, b, c0)
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
    dlag = lag / (mem + 1)

    L = generator.T
    T = sp.linalg.expm(L * dlag)  # adjoint transition operator
    wy = w[:, None] * y

    a = np.zeros((mem + 1, k, k))
    b = np.zeros((mem + 1, k))
    wy_t = wy
    w_t = w
    for m in range(mem + 1):
        wy_t = T @ wy_t
        w_t = T @ w_t
        a[m] = x.T @ (wy_t - wy)
        b[m] = x.T @ (w_t - w)
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
    T = sp.linalg.expm(L * dlag)  # adjoint transition operator

    u = w * (y @ coef + 1.0)  # dga projection
    for m in range(mem):
        u = T @ u - w * (y @ mem_coef[m])
    u = T @ u  # dga solution
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
    *,
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
        return_projection=return_projection,
        return_solution=return_solution,
        return_coef=return_coef,
        return_mem_coef=return_mem_coef,
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
    *,
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
        return_projection=return_projection,
        return_solution=return_solution,
        return_coef=return_coef,
        return_mem_coef=return_mem_coef,
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
    *,
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
    coef, mem_coef = solve(a, b, c0)
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
    _, k = y.shape
    dlag = lag / (mem + 1)

    L = generator
    Ld = L[np.ix_(d, d)]
    xwd = x[d].T * w[d]
    yd = y[d]
    rd = sp.linalg.solve(Ld, L[d] @ g + f[d])  # guess - solution
    Sd = sp.linalg.expm(dlag * Ld)  # stopped transition operator

    a = np.empty((mem + 1, k, k))
    b = np.empty((mem + 1, k))
    yd_t = yd
    rd_t = rd
    for m in range(mem + 1):
        yd_t = Sd @ yd_t
        rd_t = Sd @ rd_t
        a[m] = xwd @ (yd_t - yd)
        b[m] = xwd @ (rd_t - rd)
    c0 = xwd @ yd
    return a, b, c0


def forward_feynman_kac_projection(basis, guess, coef):
    shape, [y], [g] = _to_flat([basis], [guess])
    u = y @ coef + g
    return u.reshape(shape)


def forward_feynman_kac_solution(
    generator, basis, in_domain, function, guess, lag, mem, coef, mem_coef
):
    shape, [y], [d, f, g] = _to_flat([basis], [in_domain, function, guess])
    dlag = lag / (mem + 1)

    L = generator
    Ld = L[np.ix_(d, d)]
    yd = y[d]
    rd = sp.linalg.solve(Ld, L[d] @ g + f[d])  # guess - true solution
    Sd = sp.linalg.expm(dlag * Ld)  # stopped transition operator

    du = yd @ coef + rd  # dga projection - true solution
    for m in range(mem):
        du = Sd @ du - yd @ mem_coef[m]
    du = Sd @ du  # dga solution - true solution

    u = np.copy(g)
    u[d] -= rd  # true solution
    u[d] += du  # dga solution
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
    *,
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
        return_projection=return_projection,
        return_solution=return_solution,
        return_coef=return_coef,
        return_mem_coef=return_mem_coef,
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
    *,
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
        return_projection=return_projection,
        return_solution=return_solution,
        return_coef=return_coef,
        return_mem_coef=return_mem_coef,
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
    *,
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
    coef, mem_coef = solve(a, b, c0)
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
    _, k = y.shape
    dlag = lag / (mem + 1)

    L = generator.T
    Ld = L[np.ix_(d, d)]
    rd = L[d] * g - g[d, None] * L[d] + np.diag(f)[d]

    n = L.shape[0]
    nd = Ld.shape[0]

    A = np.block([[Ld, rd], [np.zeros((n, nd)), L]])
    eA = sp.linalg.expm(dlag * A)

    T = eA[nd:, nd:]  # time-reversed transition operator
    Sd = eA[:nd, :nd]  # time-reversed stopped transition operator
    Rd = eA[:nd, nd:]

    a = np.empty((mem + 1, k, k))
    b = np.empty((mem + 1, k))

    xd = x[d].T
    yd = y[d]

    w_t = np.empty((mem + 2, n))
    w_t[0] = w
    for m in range(1, mem + 2):
        w_t[m] = T @ w_t[m - 1]
    wyd_0 = w_t[mem + 1, d, None] * yd
    for m in range(mem + 1):
        wyd_t = w_t[mem - m, d, None] * yd
        wr = np.zeros(nd)
        for i in range(mem - m, mem + 1):
            wyd_t = Sd @ wyd_t
            wr = Sd @ wr + Rd @ w_t[i]
        a[m] = xd @ (wyd_t - wyd_0)
        b[m] = xd @ wr
    c0 = xd @ wyd_0
    return a, b, c0


def backward_feynman_kac_projection(basis, guess, coef):
    shape, [y], [g] = _to_flat([basis], [guess])
    u = y @ coef + g
    return u.reshape(shape)


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
    dlag = lag / (mem + 1)

    L = generator.T
    Ld = L[np.ix_(d, d)]
    rd = L[d] * g - g[d, None] * L[d] + np.diag(f)[d]

    n = L.shape[0]
    nd = Ld.shape[0]

    A = np.block([[Ld, rd], [np.zeros((n, nd)), L]])
    eA = sp.linalg.expm(dlag * A)

    T = eA[nd:, nd:]  # time-reversed transition operator
    Sd = eA[:nd, :nd]  # time-reversed stopped transition operator
    Rd = eA[:nd, nd:]

    yd = y[d]

    duw = w[d] * (yd @ coef)  # dga projection - guess
    for m in range(mem):
        duw = Sd @ duw + Rd @ w - w[d] * (yd @ mem_coef[m])
        w = T @ w
    duw = Sd @ duw + Rd @ w  # dga solution - guess
    w = T @ w

    uw = w * g
    uw[d] += duw  # dga solution
    u = uw / w

    return u.reshape(shape)


def solve(a, b, c0):
    """
    Solve DGA with memory for projection and memory-correction
    coefficients.

    Parameters
    ----------
    a : (mem + 1, n_basis, n_basis) ndarray of float
        DGA matrices for the homogeneous term.
    b : (mem + 1, n_basis) ndarray of float
        DGA matrices for the nonhomogeneous term.
    c0 : (n_basis, n_basis) ndarray of float
        Matrix of inner products of basis functions.

    Returns
    -------
    coef : (n_basis,) ndarray of float
        Projection coefficients.
    mem_coef : (mem, n_basis) ndarray of float
        Memory-correction coefficients.

    """
    mem = a.shape[0] - 1
    n_basis = a.shape[1]
    assert a.shape == (mem + 1, n_basis, n_basis)
    assert b.shape == (mem + 1, n_basis)

    b = b[..., None]

    inv = sp.linalg.inv(c0)
    a = inv @ a
    b = inv @ b
    c = a[::-1] + np.identity(n_basis)
    for n in range(1, mem + 1):
        a[n] -= np.sum(c[-n:] @ a[:n], axis=0)
        b[n] -= np.sum(c[-n:] @ b[:n], axis=0)

    b = b.reshape(b.shape[:2])

    coef = sp.linalg.solve(a[-1], -b[-1])
    mem_coef = a[:-1] @ coef + b[:-1]
    return coef, mem_coef


def _to_flat(bases, functions):
    """
    Flatten state space dimensions.

    This function broadcasts the state space dimensions of the input
    arrays against each other, and returns the shape of these dimensions
    and arrays with these dimensions flattened.

    Parameters
    ----------
    bases : sequence of (*shape[i], nbasis[i]) ndarray
        Vector-valued functions of the state space.
    functions : sequence of shape[i] ndarray
        Scalar functions of the state space.

    Returns
    -------
    out_shape : tuple
        Broadcasted shape of the state space:
        ``out_shape = numpy.broadcast_shapes(*shape)``.
        The flattened size of the state space is
        ``size = numpy.product(out_shape)``.
    out_bases : tuple of (size, nbasis[i]) ndarray
        Flattened broadcasted bases. Note that the last dimension is
        preserved: ``bases[i].shape[-1] == out_bases[i].shape[-1]``.
    out_functions : tuple of (size,) ndarray
        Flattened broadcasted functions.

    """
    # determine out_shape
    shapes = []
    nbases = []
    for basis in bases:
        *shape, nbasis = np.shape(basis)
        shapes.append(shape)
        nbases.append(nbasis)
    for function in functions:
        shape = np.shape(function)
        shapes.append(shape)
    out_shape = np.broadcast_shapes(*shapes)

    # broadcast and flatten arrays
    out_bases = tuple(
        np.broadcast_to(basis, (out_shape, *nbasis)).reshape((-1, nbasis))
        for basis, nbasis in zip(bases, nbases)
    )
    out_functions = tuple(
        np.ravel(np.broadcast_to(function, out_shape))
        for function in functions
    )

    return out_shape, out_bases, out_functions
