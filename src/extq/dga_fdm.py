import scipy as sp

from . import dgastat_fdm
from .dga import DGA

__all__ = [
    "reweight",
    "forward_committor",
    "forward_mfpt",
    "forward_feynman_kac",
    "backward_committor",
    "backward_mfpt",
    "backward_feynman_kac",
]


def reweight(
    generator, basis, weights, lag, *, test_basis=None, method=None, output="projection"
):
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
    stat = dgastat_fdm.StationaryDistribution(
        generator, basis, weights, test_basis=test_basis
    )
    if method is None:
        method = DGA(lag)
    out = method.fit_transform(stat, output=output)
    return out


def forward_committor(
    generator,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    test_basis=None,
    method=None,
    output="projection",
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
        method=method,
        output=output,
    )


def forward_mfpt(
    generator,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    test_basis=None,
    method=None,
    output="projection",
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
        method=method,
        output=output,
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
    method=None,
    output="projection",
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
    stat = dgastat_fdm.ForwardFeynmanKac(
        generator, basis, weights, in_domain, function, guess, test_basis=test_basis
    )
    if method is None:
        method = DGA(lag)
    out = method.fit_transform(stat, output=output)
    return out


def backward_committor(
    generator,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    test_basis=None,
    method=None,
    output="projection",
):
    """
    Compute the exact DGA backward committor.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Infinitesimal generator for the process.
    basis : (n_points, n_basis) ndarray of float
        Basis for approximating the backward committor. Each basis
        function must be zero outside of the domain.
    weights : (n_points,) ndarray of float
        Weight of each point.
    in_domain : (n_points,) ndarray of bool
        Whether each point is in the domain.
    guess : (n_points,) ndarray of float
        Guess for the backward committor. Must obey boundary conditions.
    lag : float
        Lag time.
    test_basis : (n_points, n_basis) ndarray of float, optional
        Test functions against which to minimize the error. If None, use
        `basis`.

    Returns
    -------
    (n_points,) ndarray of float
        Approximated backward committor.

    """
    return backward_feynman_kac(
        generator,
        basis,
        weights,
        in_domain,
        0.0,
        guess,
        lag,
        test_basis=test_basis,
        method=method,
        output=output,
    )


def backward_mfpt(
    generator,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    test_basis=None,
    method=None,
    output="projection",
):
    """
    Compute the exact DGA backward mean first passage time (MFPT).

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Infinitesimal generator for the process.
    basis : (n_points, n_basis) ndarray of float
        Basis for approximating the backward MFPT. Each basis function
        must be zero outside of the domain.
    weights : (n_points,) ndarray of float
        Weight of each point.
    in_domain : (n_points,) ndarray of bool
        Whether each point is in the domain.
    guess : (n_points,) ndarray of float
        Guess for the backward MFPT. Must obey boundary conditions.
    lag : float
        Lag time.
    test_basis : (n_points, n_basis) ndarray of float, optional
        Test functions against which to minimize the error. If None, use
        `basis`.

    Returns
    -------
    (n_points,) ndarray of float
        Approximated backward MFPT.

    """
    return backward_feynman_kac(
        generator,
        basis,
        weights,
        in_domain,
        1.0,
        guess,
        lag,
        test_basis=test_basis,
        method=method,
        output=output,
    )


def backward_feynman_kac(
    generator,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    test_basis=None,
    method=None,
    output="projection",
):
    """
    Compute the exact DGA solution to a backward Feynman-Kac problem.

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
    weights = sp.linalg.expm(generator.T * lag) @ weights
    stat = dgastat_fdm.BackwardFeynmanKac(
        generator, basis, weights, in_domain, function, guess, test_basis=test_basis
    )
    if method is None:
        method = DGA(lag)
    out = method.fit_transform(stat, output=output)
    return out
