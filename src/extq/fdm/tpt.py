import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def forward_committor(generator, weights, in_domain, guess):
    """Compute the forward committor.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Reweighting factor to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    guess : (M,) ndarray of float
        Guess for the committor. Must obey boundary conditions.

    Returns
    -------
    (M,) ndarray of float
        Forward committor at each point.

    """
    a = generator[in_domain, :][:, in_domain]
    b = -generator[in_domain, :] @ guess
    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return (
        guess
        + scipy.sparse.identity(len(weights), format="csr")[:, in_domain]
        @ coeffs
    )


def backward_committor(generator, weights, in_domain, guess):
    """Compute the backward committor.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Reweighting factor to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    guess : (M,) ndarray of float
        Guess for the committor. Must obey boundary conditions.

    Returns
    -------
    (M,) ndarray of float
        Backward committor at each point.

    """
    adjoint_generator = (
        scipy.sparse.diags(1.0 / weights)
        @ generator.T
        @ scipy.sparse.diags(weights)
    )
    a = adjoint_generator[in_domain, :][:, in_domain]
    b = -adjoint_generator[in_domain, :] @ guess
    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return (
        guess
        + scipy.sparse.identity(len(weights), format="csr")[:, in_domain]
        @ coeffs
    )


def forward_mfpt(generator, weights, in_domain, guess):
    """Compute the forward mean first passage time.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Change of measure to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    guess : (M,) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.

    Returns
    -------
    (M,) ndarray of float
        Forward mean first passage time at each point.

    """
    a = generator[in_domain, :][:, in_domain]
    b = -generator[in_domain, :] @ guess - 1.0
    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return (
        guess
        + scipy.sparse.identity(len(weights), format="csr")[:, in_domain]
        @ coeffs
    )


def backward_mfpt(generator, weights, in_domain, guess):
    """Compute the backward mean first passage time.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Change of measure to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    guess : (M,) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.

    Returns
    -------
    (M,) ndarray of float
        Backward mean first passage time at each point.

    """
    adjoint_generator = (
        scipy.sparse.diags(1.0 / weights)
        @ generator.T
        @ scipy.sparse.diags(weights)
    )
    a = adjoint_generator[in_domain, :][:, in_domain]
    b = -adjoint_generator[in_domain, :] @ guess - 1.0
    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return (
        guess
        + scipy.sparse.identity(len(weights), format="csr")[:, in_domain]
        @ coeffs
    )


def forward_feynman_kac(generator, weights, in_domain, function, guess):
    """Solve the forward Feynman-Kac formula.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Change of measure to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    function : (M,) ndarray of float
        Function to integrate. Must be zero outside the domain.
    guess : (M,) ndarray of float
        Guess of the solution. Must obey boundary conditions.

    Returns
    -------
    (M,) ndarray of float
        Solution of the Feynman-Kac formula at each point.

    """
    assert np.all(function[np.logical_not(in_domain)] == 0.0)
    a = generator[in_domain, :][:, in_domain]
    b = -generator[in_domain, :] @ guess - function[in_domain]
    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return (
        guess
        + scipy.sparse.identity(len(weights), format="csr")[:, in_domain]
        @ coeffs
    )


def backward_feynman_kac(generator, weights, in_domain, function, guess):
    """Solve the backward Feynman-Kac formula.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Change of measure to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    function : (M,) ndarray of float
        Function to integrate. Must be zero outside the domain.
    guess : (M,) ndarray of float
        Guess of the solution. Must obey boundary conditions.

    Returns
    -------
    (M,) ndarray of float
        Solution of the Feynman-Kac formula at each point.

    """
    assert np.all(function[np.logical_not(in_domain)] == 0.0)
    adjoint_generator = (
        scipy.sparse.diags(1.0 / weights)
        @ generator.T
        @ scipy.sparse.diags(weights)
    )
    a = adjoint_generator[in_domain, :][:, in_domain]
    b = -adjoint_generator[in_domain, :] @ guess - function[in_domain]
    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return (
        guess
        + scipy.sparse.identity(len(weights), format="csr")[:, in_domain]
        @ coeffs
    )


def reweight(generator):
    """Compute the reweighting factors to the invariant distribution.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.

    Returns
    -------
    (M,) ndarray of float
        Reweighting factor at each point.

    """
    # w, v = sparse.linalg.eigs(tmat.T, k=1, which="LR")
    # fixed_index = np.argmax(np.abs(v[:, 0]))
    fixed_index = np.random.randint(generator.shape[0])
    mask = np.full(generator.shape[0], True)
    mask[fixed_index] = False

    a = generator.T[mask, :][:, mask]
    b = -generator.T[mask, fixed_index]
    coeffs = scipy.sparse.linalg.spsolve(a, b)

    weights = np.empty(generator.shape[0])
    weights[fixed_index] = 1.0
    weights[mask] = coeffs
    weights /= np.sum(weights)
    return weights


def rate(generator, forward_q, backward_q, weights, rxn_coords=None):
    """Compute the TPT rate.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    forward_q : (M,) ndarray of float
        Forward committor at each point.
    backward_q : (M,) ndarray of float
        Backward committor at each point.
    weights : (M,) ndarray of float.
        Reweighting factor at each point.
    rxn_coords : (M,) ndarray of float, optional
        Reaction coordinate at each point. This must be zero in the
        reactant state and one in the product state. If None, estimate
        the rate without using a reaction coordinate.

    Returns
    -------
    float
        TPT rate.

    """
    if rxn_coords is None:
        numer = (weights * backward_q) @ generator @ forward_q
    else:
        numer = (weights * backward_q) @ (
            generator @ (forward_q * rxn_coords)
            - rxn_coords * (generator @ forward_q)
        )
    denom = np.sum(weights)
    return numer / denom


def current(generator, forward_q, backward_q, weights, cv):
    """Compute the reactive current at each point.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    forward_q : (M,) ndarray of float
        Forward committor at each point.
    backward_q : (M,) ndarray of float
        Backward committor at each point.
    weights : (M,) ndarray of float.
        Reweighting factor at each point.
    cv : (M,) ndarray of float
        Collective variable at each point.

    Returns
    -------
    (M,) ndarray of float
        Reactive current at each point.

    """
    forward_flux = (weights * backward_q) * (
        generator @ (forward_q * cv) - cv * (generator @ forward_q)
    )
    backward_flux = forward_q * (
        generator.T @ ((weights * backward_q) * cv)
        - cv * (generator.T @ (weights * backward_q))
    )
    numer = 0.5 * (forward_flux - backward_flux)
    denom = np.sum(weights)
    return numer / denom
