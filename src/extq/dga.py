from . import _dgamat, linalg
from .utils import normalize_weights, shift_weights, uniform_weights

__all__ = [
    "reweight",
    "forward_committor",
    "forward_mfpt",
    "forward_feynman_kac",
    "backward_committor",
    "backward_mfpt",
    "backward_feynman_kac",
]


def reweight(basis, lag, maxlag=None, guess=None, test_basis=None, *, normalize=True):
    """Estimate the change of measure to the invariant distribution.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the change of measure.
    lag : int
        Lag time in unit of frames.
    maxlag : int
        Number of frames at the end of each trajectory that are required
        to have zero weight. This is the maximum lag time the output
        weights can be used with by other methods.
    guess : list of (n_frames[i],) ndarray of float, optional
        Guess for the change of measure. The last maxlag frames of
        each trajectory must be zero.
        If None, use uniform weights (except for the last lag frames).
    test_basis : list of (n_frames[i], n_basis) ndarray of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the change of
        measure. If None, use the basis that is used to estimate the
        change of measure.
    normalize : bool, optional
        If True (default), normalize output to one.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the change of measure at each frame of the
        trajectory.

    """
    if maxlag is None:
        maxlag = lag
    assert 0 < lag <= maxlag
    if guess is None:
        guess = uniform_weights(basis, maxlag)
    a, b = _dgamat.reweight_matrices(basis, lag, guess, test_basis=test_basis)
    coef = -linalg.solve(a, b)
    out = [w * (y @ coef + 1.0) for y, w in zip(basis, guess, strict=True)]
    if normalize:
        out = normalize_weights(out)
    return out


def forward_committor(basis, weights, in_domain, guess, lag, test_basis=None):
    """Estimate the forward committor using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the committor.
        If None, use the basis that is used to estimate the committor.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the forward committor at each frame of the
        trajectory.

    """
    return forward_feynman_kac(
        basis,
        weights,
        in_domain,
        0.0,
        guess,
        lag,
        test_basis=test_basis,
    )


def forward_mfpt(basis, weights, in_domain, guess, lag, test_basis=None):
    """Estimate the forward mean first passage time using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the mean first passage time. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the forward mean first passage time at each frame of
        the trajectory.

    """
    return forward_feynman_kac(
        basis,
        weights,
        in_domain,
        1.0,
        guess,
        lag,
        test_basis=test_basis,
    )


def forward_feynman_kac(
    basis, weights, in_domain, function, guess, lag, test_basis=None
):
    """Solve the forward Feynman-Kac formula using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the solution of the Feynman-Kac formula.
        Must be zero outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    function : list of (n_frames[i]-1,) ndarray of float
        Function to integrate. Note that is defined over transitions,
        not frames.
    guess : list of (n_frames[i],) ndarray of float
        Guess of the solution. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the solution of the forward Feynman-Kac formula at
        each frame of the trajectory.

    """
    a, b = _dgamat.forward_feynman_kac_matrices(
        basis, weights, in_domain, function, guess, lag, test_basis=test_basis
    )
    coef = -linalg.solve(a, b)
    return transform(coef, basis, guess)


def backward_committor(basis, weights, in_domain, guess, lag, test_basis=None):
    """Estimate the backward committor using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the committor. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the committor.
        If None, use the basis that is used to estimate the committor.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the backward committor at each frame of the
        trajectory.

    """
    return backward_feynman_kac(
        basis,
        weights,
        in_domain,
        0.0,
        guess,
        lag,
        test_basis=test_basis,
    )


def backward_mfpt(basis, weights, in_domain, guess, lag, test_basis=None):
    """Estimate the backward mean first passage time using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the mean first passage time. Must be zero
        outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    guess : list of (n_frames[i],) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the mean first
        passage time. If None, use the basis that is used to estimate
        the mean first passage time.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the backward mean first passage time at each frame of
        the trajectory.

    """
    return backward_feynman_kac(
        basis,
        weights,
        in_domain,
        1.0,
        guess,
        lag,
        test_basis=test_basis,
    )


def backward_feynman_kac(
    basis, weights, in_domain, function, guess, lag, test_basis=None
):
    """Solve the backward Feynman-Kac formula using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the solution of the Feynman-Kac formula.
        Must be zero outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    function : list of (n_frames[i]-1,) ndarray of float
        Function to integrate. Note that is defined over transitions,
        not frames.
    guess : list of (n_frames[i],) ndarray of float
        Guess of the solution. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the solution of the backward Feynman-Kac formula at
        each frame of the trajectory.

    """
    weights = shift_weights(weights, lag)
    a, b = _dgamat.backward_feynman_kac_matrices(
        basis, weights, in_domain, function, guess, lag, test_basis=test_basis
    )
    coef = -linalg.solve(a, b)
    return transform(coef, basis, guess)


def transform(coef, basis, guess):
    return [y @ coef + g for y, g in zip(basis, guess, strict=True)]
