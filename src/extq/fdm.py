import numba as nb
import numpy as np
from scipy import sparse


def transition_matrix_to_generator_matrix(tmat, dt):
    """Compute the generator matrix from the transition matrix.

    Parameters
    ----------
    tmat : (M, M) sparse matrix
        Transition matrix.
    dt : int
        Lag time of the transition matrix.

    Returns
    -------
    (M, M) sparse matrix
        Generator matrix.

    """
    return (tmat - sparse.identity(tmat.shape[0])) / dt


def transition_matrix_reversible_2d(
    potential,
    xlo,
    xnum,
    ylo,
    ynum,
    sep,
    kT,
):
    """
    Compute the transition matrix for a reversible 2D potential.

    Parameters
    ----------
    potential : callable
        Potential function for a 2D system. This function is called as
        potential(x, y), and must be vectorized.
    xlo, ylo : float
        Minimum x/y value.
    xnum, ynum : int
        Number of x/y values to evaluate.
    sep : float
        Separation between each x/y value.
    kT : float
        Temperature of the system, in units of the potential.

    Returns
    -------
    sparse matrix
        Transition matrix.
    float
        Time step of the transition matrix.

    """

    # precompute potential and indices
    xind, yind = np.ogrid[:xnum, :ynum]
    xv, yv = xlo + sep * xind, ylo + sep * yind
    u = potential(xv, yv) / kT
    ind = np.ravel_multi_index((xind, yind), (xnum, ynum))

    # possible transitions per step
    transitions = [
        (np.s_[:-1, :], np.s_[1:, :]),
        (np.s_[1:, :], np.s_[:-1, :]),
        (np.s_[:, :-1], np.s_[:, 1:]),
        (np.s_[:, 1:], np.s_[:, :-1]),
    ]

    # compute transition matrix
    tmat = _transition_matrix_reversible_helper(
        transitions, u, ind, (xnum, ynum)
    )

    # time step of transition matrix
    dt = sep ** 2 / (8.0 * kT)

    return tmat, dt


def transition_matrix_reversible_3d(
    potential,
    xlo,
    xnum,
    ylo,
    ynum,
    zlo,
    znum,
    sep,
    kT,
):
    """
    Compute the transition matrix for a reversible 3D potential.

    Parameters
    ----------
    potential : callable
        Potential function for a 3D system. This function is called as
        potential(x, y, z), and must be vectorized.
    xlo, ylo, zlo : float
        Minimum x/y/z value.
    xnum, ynum, znum : int
        Number of x/y/z values to evaluate.
    sep : float
        Separation between each x/y/z value.
    kT : float
        Temperature of the system, in units of the potential.

    Returns
    -------
    sparse matrix
        Transition matrix.
    float
        Time step of the transition matrix.

    """

    # precompute potential and indices
    xind, yind, zind = np.ogrid[:xnum, :ynum, :znum]
    xv, yv, zv = xlo + sep * xind, ylo + sep * yind, zlo + sep * zind
    u = potential(xv, yv, zv) / kT
    ind = np.ravel_multi_index((xind, yind, zind), (xnum, ynum, znum))

    # possible transitions per step
    transitions = [
        (np.s_[:-1, :, :], np.s_[1:, :, :]),
        (np.s_[1:, :, :], np.s_[:-1, :, :]),
        (np.s_[:, :-1, :], np.s_[:, 1:, :]),
        (np.s_[:, 1:, :], np.s_[:, :-1, :]),
        (np.s_[:, :, :-1], np.s_[:, :, 1:]),
        (np.s_[:, :, 1:], np.s_[:, :, :-1]),
    ]

    # compute transition matrix
    tmat = _transition_matrix_reversible_helper(
        transitions, u, ind, (xnum, ynum, znum)
    )

    # time step of transition matrix
    dt = sep ** 2 / (12.0 * kT)

    return tmat, dt


def _transition_matrix_reversible_helper(transitions, u, ind, shape):
    data = []
    row_ind = []
    col_ind = []
    p0 = np.ones(shape)

    # probabilities of transitioning to adjacent cell
    for row, col in transitions:
        p = (1.0 / len(transitions)) / (1.0 + np.exp((u[col] - u[row])))
        p0[row] -= p
        data.append(p.ravel())
        row_ind.append(ind[row].ravel())
        col_ind.append(ind[col].ravel())

    # probability of not transitioning
    data.append(p0.ravel())
    row_ind.append(ind.ravel())
    col_ind.append(ind.ravel())

    # assemble sparse transition matrix
    data = np.concatenate(data)
    row_ind = np.concatenate(row_ind)
    col_ind = np.concatenate(col_ind)
    return sparse.csr_matrix(
        (data, (row_ind, col_ind)), shape=(p0.size, p0.size)
    )


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
    coeffs = sparse.linalg.spsolve(a, b)
    return (
        guess
        + sparse.identity(len(weights), format="csr")[:, in_domain] @ coeffs
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
        sparse.diags(1.0 / weights) @ generator.T @ sparse.diags(weights)
    )
    a = adjoint_generator[in_domain, :][:, in_domain]
    b = -adjoint_generator[in_domain, :] @ guess
    coeffs = sparse.linalg.spsolve(a, b)
    return (
        guess
        + sparse.identity(len(weights), format="csr")[:, in_domain] @ coeffs
    )


def reweight(tmat):
    """Compute the reweighting factors to the invariant distribution.

    Parameters
    ----------
    tmat : (M, M) sparse matrix
        Transition matrix.

    Returns
    -------
    (M,) ndarray of float
        Reweighting factor at each point.

    """
    w, v = sparse.linalg.eigs(tmat.T, k=1, which="LR")
    fixed_index = np.argmax(np.abs(v[:, 0]))
    mask = np.full(tmat.shape[0], True)
    mask[fixed_index] = False

    gen = tmat - sparse.identity(tmat.shape[0])
    a = gen.T[mask, :][:, mask]
    b = -gen.T[mask, fixed_index]
    coeffs = sparse.linalg.spsolve(a, b)

    weights = np.empty(tmat.shape[0])
    weights[fixed_index] = 1.0
    weights[mask] = coeffs
    weights /= np.sum(weights)
    return weights


def rate(generator, forward_q, backward_q, weights, rxn_coords):
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
    rxn_coords : (M,) ndarray of float
        Reaction coordinate at each point. This must be zero in the
        reactant state and one in the product state.

    Returns
    -------
    float
        TPT rate.

    """
    numer = np.sum(
        (weights * backward_q)
        * (
            generator @ (forward_q * rxn_coords)
            - rxn_coords * (generator @ forward_q)
        )
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
