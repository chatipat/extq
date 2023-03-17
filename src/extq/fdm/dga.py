import numpy as np

from .. import linalg
from ._utils import augment_generator
from ._utils import build_kernel

__all__ = [
    "reweight",
    "forward_committor",
    "forward_mfpt",
    "forward_feynman_kac",
    "backward_committor_reweight",
    "backward_mfpt_reweight",
    "backward_feynman_kac_reweight",
]


def reweight(generator, basis, weights, guess, lag, test_basis=None):
    """
    Compute the exact DGA stationary distribution.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Infinitesimal generator for the process.
    basis : (n_points, n_basis) ndarray of float
        Basis for approximating the change of measure to the stationary
        distribution. The integral of each basis function with respect
        to `weights` should be zero.
    weights : (n_points,) ndarray of float
        Initial weight of each point.
    guess : (n_points,) ndarray of float
        Guess for the change of measure. The integral of this with
        respect to `weights` should be one.
    lag : float
        Lag time.
    test_basis : (n_points, n_basis) ndarray of float, optional
        Test functions against which to minimize the error. If None, use
        `basis`.

    Returns
    -------
    (n_points,) ndarray of float
        Approximated stationary distribution. This is the product of
        `weights` and the change of measure approximated with `basis`
        and `guess`.

    """
    if test_basis is None:
        test_basis = basis
    shape, (x, y), (w, g) = _to_flat((test_basis, basis), (weights, guess))
    v = _dga_coeffs(generator.T, x, linalg.scale_rows(w, y), w * g, lag)
    p = w * (y @ v + g)
    return p.reshape(shape)


def forward_committor(
    generator, basis, weights, in_domain, guess, lag, test_basis=None
):
    """
    Compute the exact DGA forward committor.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Infinitesimal generator for the process.
    basis : (n_points, n_basis) ndarray of float
        Basis for approximating the forward committor. Each basis
        function must be zero outside of the domain.
    weights : (n_points,) ndarray of float
        Weight of each point.
    in_domain : (n_points,) ndarray of bool
        Whether each point is in the domain.
    guess : (n_points,) ndarray of float
        Guess for the forward committor. Must obey boundary conditions.
    lag : float
        Lag time.
    test_basis : (n_points, n_basis) ndarray of float, optional
        Test functions against which to minimize the error. If None, use
        `basis`.

    Returns
    -------
    (n_points,) ndarray of float
        Approximated forward committor.

    """
    return forward_feynman_kac(
        generator,
        basis,
        weights,
        in_domain,
        0.0,
        guess,
        lag,
        test_basis=test_basis,
    )


def forward_mfpt(
    generator, basis, weights, in_domain, guess, lag, test_basis=None
):
    """
    Compute the exact DGA forward mean first passage time (MFPT).

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Infinitesimal generator for the process.
    basis : (n_points, n_basis) ndarray of float
        Basis for approximating the forward MFPT. Each basis function
        must be zero outside of the domain.
    weights : (n_points,) ndarray of float
        Weight of each point.
    in_domain : (n_points,) ndarray of bool
        Whether each point is in the domain.
    guess : (n_points,) ndarray of float
        Guess for the forward MFPT. Must obey boundary conditions.
    lag : float
        Lag time.
    test_basis : (n_points, n_basis) ndarray of float, optional
        Test functions against which to minimize the error. If None, use
        `basis`.

    Returns
    -------
    (n_points,) ndarray of float
        Approximated forward MFPT.

    """
    return forward_feynman_kac(
        generator,
        basis,
        weights,
        in_domain,
        1.0,
        guess,
        lag,
        test_basis=test_basis,
    )


def forward_feynman_kac(
    generator,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    test_basis=None,
):
    """
    Compute the exact DGA solution to a forward Feynman-Kac problem.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Infinitesimal generator for the process.
    basis : (n_points, n_basis) ndarray of float
        Basis for approximating the solution. Each basis function must
        be zero outside of the domain.
    weights : (n_points,) ndarray of float
        Weight of each point.
    in_domain : (n_points,) ndarray of bool
        Whether each point is in the domain.
    function : (n_points,) ndarray of float
        Function to integrate until the process leaves the domain.
    guess : (n_points,) ndarray of float
        Guess for the solution. Must obey boundary conditions.
    lag : float
        Lag time.
    test_basis : (n_points, n_basis) ndarray of float, optional
        Test functions against which to minimize the error. If None, use
        `basis`.

    Returns
    -------
    (n_points,) ndarray of float
        Approximated solution.

    """
    if test_basis is None:
        test_basis = basis
    shape, (x, y), (w, d, f, g) = _to_flat(
        (test_basis, basis), (weights, in_domain, function, guess)
    )
    n = len(w)
    w = w / np.sum(w)

    K = feynman_kac_kernel(generator, d, f, g)
    _, L = augment_generator(generator, *K)
    X = linalg.block_diag([x, np.zeros((n, 0))])
    Y = linalg.block_diag([y, np.zeros((n, 0))])
    W = np.concatenate([w, w])
    R = np.concatenate([np.zeros(n), np.ones(n)])

    v = _dga_coeffs(L, linalg.scale_rows(W, X), Y, R, lag)
    u = y @ v + g
    return u.reshape(shape)


def backward_committor_reweight(
    generator,
    basis,
    basis_w,
    weights,
    in_domain,
    guess,
    guess_w,
    lag,
    test_basis=None,
    test_basis_w=None,
):
    """
    Compute the exact DGA backward committor and stationary distribution.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Infinitesimal generator for the process.
    basis : (n_points, n_basis) ndarray of float
        Basis for approximating the backward committor. Each basis
        function must be zero outside of the domain.
    basis_w : (n_points, n_basis_w) ndarray of float
        Basis for approximating the change of measure to the stationary
        distribution. The integral of each basis function with respect
        to `weights` should be zero.
    weights : (n_points,) ndarray of float
        Weight of each point.
    in_domain : (n_points,) ndarray of bool
        Whether each point is in the domain.
    guess : (n_points,) ndarray of float
        Guess for the backward committor. Must obey boundary conditions.
    guess_w : (n_points,) ndarray of float
        Guess for the change of measure. The integral of this with
        respect to `weights` should be one.
    lag : float
        Lag time.
    test_basis : (n_points, n_basis) ndarray of float, optional
        Test functions against which to minimize the error of the
        product of the change of measure and the backward committor.
        If None, use `basis`.
    test_basis_w : (n_points, n_basis_w) ndarray of float, optional
        Test functions against which to minimize the error of the
        change of measure. If None, use `basis_w`.

    Returns
    -------
    qb : (n_points,) ndarray of float
        Approximated backward committor. Note that the statistic DGA
        approximates is ``qb * sd``.
    sd : (n_points,) ndarray of float
        Approximated stationary distribution. This is the product of
        `weights` and the change of measure approximated with `basis_w`
        and `guess_w`.

    """
    return backward_feynman_kac_reweight(
        generator,
        basis,
        basis_w,
        weights,
        in_domain,
        0.0,
        guess,
        guess_w,
        lag,
        test_basis=test_basis,
        test_basis_w=test_basis_w,
    )


def backward_mfpt_reweight(
    generator,
    basis,
    basis_w,
    weights,
    in_domain,
    guess,
    guess_w,
    lag,
    test_basis=None,
    test_basis_w=None,
):
    """
    Compute the exact DGA backward mean first passage time (MFPT) and
    stationary distribution.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Infinitesimal generator for the process.
    basis : (n_points, n_basis) ndarray of float
        Basis for approximating the backward MFPT. Each basis function
        must be zero outside of the domain.
    basis_w : (n_points, n_basis_w) ndarray of float
        Basis for approximating the change of measure to the stationary
        distribution. The integral of each basis function with respect
        to `weights` should be zero.
    weights : (n_points,) ndarray of float
        Weight of each point.
    in_domain : (n_points,) ndarray of bool
        Whether each point is in the domain.
    guess : (n_points,) ndarray of float
        Guess for the backward MFPT. Must obey boundary conditions.
    guess_w : (n_points,) ndarray of float
        Guess for the change of measure. The integral of this with
        respect to `weights` should be one.
    lag : float
        Lag time.
    test_basis : (n_points, n_basis) ndarray of float, optional
        Test functions against which to minimize the error of the
        product of the change of measure and the MFPT.
        If None, use `basis`.
    test_basis_w : (n_points, n_basis_w) ndarray of float, optional
        Test functions against which to minimize the error of the
        change of measure. If None, use `basis_w`.

    Returns
    -------
    mb : (n_points,) ndarray of float
        Approximated backward MFPT. Note that the statistic DGA
        approximates is ``mb * sd``.
    sd : (n_points,) ndarray of float
        Approximated stationary distribution. This is the product of
        `weights` and the change of measure approximated with `basis_w`
        and `guess_w`.

    """
    return backward_feynman_kac_reweight(
        generator,
        basis,
        basis_w,
        weights,
        in_domain,
        1.0,
        guess,
        guess_w,
        lag,
        test_basis=test_basis,
        test_basis_w=test_basis_w,
    )


def backward_feynman_kac_reweight(
    generator,
    basis,
    basis_w,
    weights,
    in_domain,
    function,
    guess,
    guess_w,
    lag,
    test_basis=None,
    test_basis_w=None,
):
    """
    Compute the exact DGA solution to a backward Feynman-Kac problem and
    stationary distribution.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Infinitesimal generator for the process.
    basis : (n_points, n_basis) ndarray of float
        Basis for approximating the solution. Each basis function must
        be zero outside of the domain.
    basis_w : (n_points, n_basis_w) ndarray of float
        Basis for approximating the change of measure to the stationary
        distribution. The integral of each basis function with respect
        to `weights` should be zero.
    weights : (n_points,) ndarray of float
        Weight of each point.
    in_domain : (n_points,) ndarray of bool
        Whether each point is in the domain.
    function : (n_points,) ndarray of float
        Function to integrate until the process leaves the domain.
    guess : (n_points,) ndarray of float
        Guess for the solution. Must obey boundary conditions.
    guess_w : (n_points,) ndarray of float
        Guess for the change of measure. The integral of this with
        respect to `weights` should be one.
    lag : float
        Lag time.
    test_basis : (n_points, n_basis) ndarray of float, optional
        Test functions against which to minimize the error of the
        product of the change of measure and the solution.
        If None, use `basis`.
    test_basis_w : (n_points, n_basis_w) ndarray of float, optional
        Test functions against which to minimize the error of the
        change of measure. If None, use `basis_w`.

    Returns
    -------
    ub : (n_points,) ndarray of float
        Approximated solution. Note that the statistic DGA approximates
        is ``ub * sd``.
    sd : (n_points,) ndarray of float
        Approximated stationary distribution. This is the product of
        `weights` and the change of measure approximated with `basis_w`
        and `guess_w`.

    """
    if test_basis is None:
        test_basis = basis
    if test_basis_w is None:
        test_basis_w = basis_w
    shape, (xu, yu, xw, yw), (w, d, f, gu, gw) = _to_flat(
        (test_basis, basis, test_basis_w, basis_w),
        (weights, in_domain, function, guess, guess_w),
    )
    n = len(w)
    w = w / np.sum(w)

    K = feynman_kac_kernel(generator.T, d, f, gu)
    _, L = augment_generator(generator.T, *K)
    X = linalg.block_diag([xu, xw])
    Y = linalg.block_diag([yu, yw])
    W = np.concatenate([w, w])
    R = np.concatenate([np.zeros(n), gw])

    v = _dga_coeffs(L, X, linalg.scale_rows(W, Y), W * R, lag)
    ucom, com = (Y @ v + R).reshape(2, n)
    return (ucom / com + gu).reshape(shape), (w * com).reshape(shape)


def feynman_kac_kernel(generator, in_domain, function, guess):
    """
    Augmenting kernel for a Feynman-Kac problem.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Infinitesimal generator for the process. The augmenting kernel
        is only evaluated at nonzero entries of this matrix. The values
        of the entries are not used otherwise.
    in_domain : (n_points,) ndarray of bool
        Whether each point is in the domain.
    function : (n_points,) ndarray of float
        Function to integrate until the process leaves the domain.
    guess : (n_points,) ndarray of float
        Guess for the solution. Must obey boundary conditions. This is
        used to homogenize the boundary conditions.

    Returns
    -------
    (n_indices * n_points, n_indices * n_points) sparse matrix of float
        Augmenting kernel.

    """
    d, f, g = np.broadcast_arrays(in_domain, function, guess)
    d = np.ravel(d)
    f = np.ravel(f)
    g = np.ravel(g)
    assert generator.shape == (d.size, d.size)

    def entries(row, col, spatial, temporal):
        spatial[0, 0] = d[row] * d[col]
        spatial[0, 1] = d[row] * (g[col] - g[row])
        spatial[1, 1] = 1.0

        temporal[0, 1] = 0.5 * d[row] * (f[row] + d[col] * f[col])

    return build_kernel(generator, 2, entries)


def _dga_coeffs(generator, x, y, g, lag):
    """
    Compute the exact DGA coefficients at a given lag time.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Infinitesimal generator for the process.
    x : (n_points, n_basis) ndarray of float
        Test functions against which to minimize the error.
    y : (n_points, n_basis) ndarray of float
        Basis for approximating the solution. Must be orthogonal to the
        constraints.
    g : (n_points,) ndarray of float
        Guess for the solution. Must satisfy constraints (e.g., boundary
        conditions or normalization).
    lag : float
        Lag time.

    Returns
    -------
    (n_basis,) ndarray of float
        Coefficients for the exact DGA solution.

    """
    dx = _difference(generator.T, x, lag)
    a = dx.T @ y  # x.T @ (expm(generator * lag) @ y - y) / lag
    b = dx.T @ g  # x.T @ (expm(generator * lag) @ g - g) / lag
    v = linalg.solve(a, -b)
    return v


def _difference(generator, f, lag):
    """
    Compute the time derivative or difference quotient of a function.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Infinitesimal generator for the process.
    f : (n_points,) or (n_points, n_functions) ndarray of float
        Function for which to compute the time derivative or difference
        quotient.
    lag : float
        Time interval for the difference quotient. Must be non-negative.
        If zero, the difference quotient is the forward derivative.

    Returns
    -------
    (n_points,) or (n_points,n_functions) ndarray of float
        Time derivative or difference quotient of `f`.

    """
    assert lag >= 0
    if lag == 0:
        return generator @ f
    else:
        return (linalg.expm_multiply(generator * lag, f) - f) / lag


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
        np.broadcast_to(basis, out_shape + (nbasis,)).reshape((-1, nbasis))
        for basis, nbasis in zip(bases, nbases)
    )
    out_functions = tuple(
        np.ravel(np.broadcast_to(function, out_shape))
        for function in functions
    )

    return out_shape, out_bases, out_functions
