import numba as nb
import numpy as np
from scipy import sparse


def generator_matrix_reversible_2d(
    potential,
    xlo,
    xnum,
    ylo,
    ynum,
    sep,
    kT,
):
    """
    Compute the generator matrix for a reversible 2D potential.

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
        Generator matrix.

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

    # generator matrix
    gen = (tmat - sparse.identity(tmat.shape[0])) / dt

    return gen


def generator_matrix_reversible_3d(
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
    Compute the generator matrix for a reversible 3D potential.

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
        Generator matrix.

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

    # generator matrix
    gen = (tmat - sparse.identity(tmat.shape[0])) / dt

    return gen


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


def generator_matrix_irreversible_2d(
    drift,
    diffusion,
    xlo,
    xhi,
    xnum,
    ylo,
    yhi,
    ynum,
):
    """Compute the generator matrix for an irreversible 2D potential.

    Parameters
    ----------
    drift, diffusion : callable
        Drift and diffusion functions for a 2D system. These functions
        are called as drift(x, y) and diffusion(x, y), and must
        be vectorized.
    xlo, ylo : float
        Minimum x/y values.
    xhi, yhi : float
        Maximum x/y values.
    xnum, ynum : int
        Number of x/y values to evaluate.

    Returns
    -------
    sparse matrix
        Generator matrix.

    """

    # precompute indices and drift/diffusion terms
    xind, yind = np.ogrid[:xnum, :ynum]
    ind = np.ravel_multi_index((xind, yind), (xnum, ynum))
    xsep = (xhi - xlo) / (xnum - 1.0)
    ysep = (yhi - ylo) / (ynum - 1.0)
    xv, yv = xlo + xsep * xind, ylo + ysep * yind
    mu_x, mu_y = drift(xv, yv)
    sigma_x, sigma_y = diffusion(xv, yv)

    data = []
    row_ind = []
    col_ind = []
    p0 = np.zeros((xnum, ynum))

    # probability of transitioning to adjacent cell

    row, col = np.s_[:-1, :], np.s_[1:, :]
    p = 0.5 * mu_x[row] / xsep + sigma_x[row] / xsep ** 2
    p0[row] -= p
    data.append(p.ravel())
    row_ind.append(ind[row].ravel())
    col_ind.append(ind[col].ravel())

    row, col = np.s_[1:, :], np.s_[:-1, :]
    p = -0.5 * mu_x[row] / xsep + sigma_x[row] / xsep ** 2
    p0[row] -= p
    data.append(p.ravel())
    row_ind.append(ind[row].ravel())
    col_ind.append(ind[col].ravel())

    row, col = np.s_[:, :-1], np.s_[:, 1:]
    p = 0.5 * mu_y[row] / ysep + sigma_y[row] / ysep ** 2
    p0[row] -= p
    data.append(p.ravel())
    row_ind.append(ind[row].ravel())
    col_ind.append(ind[col].ravel())

    row, col = np.s_[:, 1:], np.s_[:, :-1]
    p = -0.5 * mu_y[row] / ysep + sigma_y[row] / ysep ** 2
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


def generator_matrix_irreversible_3d(
    drift,
    diffusion,
    xlo,
    xhi,
    xnum,
    ylo,
    yhi,
    ynum,
    zlo,
    zhi,
    znum,
):
    """Compute the generator matrix for an irreversible 3D potential.

    Parameters
    ----------
    drift, diffusion : callable
        Drift and diffusion functions for a 3D system. These functions
        are called as drift(x, y, z) and diffusion(x, y, z), and must
        be vectorized.
    xlo, ylo, zlo : float
        Minimum x/y/z values.
    xhi, yhi, zhi : float
        Maximum x/y/z values.
    xnum, ynum, znum : int
        Number of x/y/z values to evaluate.

    Returns
    -------
    sparse matrix
        Generator matrix.

    """

    # precompute indices and drift/diffusion terms
    xind, yind, zind = np.ogrid[:xnum, :ynum, :znum]
    ind = np.ravel_multi_index((xind, yind, zind), (xnum, ynum, znum))
    xsep = (xhi - xlo) / (xnum - 1.0)
    ysep = (yhi - ylo) / (ynum - 1.0)
    zsep = (zhi - zlo) / (znum - 1.0)
    xv, yv, zv = xlo + xsep * xind, ylo + ysep * yind, zlo + zsep * zind
    mu_x, mu_y, mu_z = drift(xv, yv, zv)
    sigma_x, sigma_y, sigma_z = diffusion(xv, yv, zv)

    data = []
    row_ind = []
    col_ind = []
    p0 = np.zeros((xnum, ynum, znum))

    # probability of transitioning to adjacent cell

    row, col = np.s_[:-1, :, :], np.s_[1:, :, :]
    p = 0.5 * mu_x[row] / xsep + sigma_x[row] / xsep ** 2
    p0[row] -= p
    data.append(p.ravel())
    row_ind.append(ind[row].ravel())
    col_ind.append(ind[col].ravel())

    row, col = np.s_[1:, :, :], np.s_[:-1, :, :]
    p = -0.5 * mu_x[row] / xsep + sigma_x[row] / xsep ** 2
    p0[row] -= p
    data.append(p.ravel())
    row_ind.append(ind[row].ravel())
    col_ind.append(ind[col].ravel())

    row, col = np.s_[:, :-1, :], np.s_[:, 1:, :]
    p = 0.5 * mu_y[row] / ysep + sigma_y[row] / ysep ** 2
    p0[row] -= p
    data.append(p.ravel())
    row_ind.append(ind[row].ravel())
    col_ind.append(ind[col].ravel())

    row, col = np.s_[:, 1:, :], np.s_[:, :-1, :]
    p = -0.5 * mu_y[row] / ysep + sigma_y[row] / ysep ** 2
    p0[row] -= p
    data.append(p.ravel())
    row_ind.append(ind[row].ravel())
    col_ind.append(ind[col].ravel())

    row, col = np.s_[:, :, :-1], np.s_[:, :, 1:]
    p = 0.5 * mu_z[row] / zsep + sigma_z[row] / zsep ** 2
    p0[row] -= p
    data.append(p.ravel())
    row_ind.append(ind[row].ravel())
    col_ind.append(ind[col].ravel())

    row, col = np.s_[:, :, 1:], np.s_[:, :, :-1]
    p = -0.5 * mu_z[row] / zsep + sigma_z[row] / zsep ** 2
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
    coeffs = sparse.linalg.spsolve(a, b)
    return (
        guess
        + sparse.identity(len(weights), format="csr")[:, in_domain] @ coeffs
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
        sparse.diags(1.0 / weights) @ generator.T @ sparse.diags(weights)
    )
    a = adjoint_generator[in_domain, :][:, in_domain]
    b = -adjoint_generator[in_domain, :] @ guess - 1.0
    coeffs = sparse.linalg.spsolve(a, b)
    return (
        guess
        + sparse.identity(len(weights), format="csr")[:, in_domain] @ coeffs
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
    coeffs = sparse.linalg.spsolve(a, b)
    return (
        guess
        + sparse.identity(len(weights), format="csr")[:, in_domain] @ coeffs
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
        sparse.diags(1.0 / weights) @ generator.T @ sparse.diags(weights)
    )
    a = adjoint_generator[in_domain, :][:, in_domain]
    b = -adjoint_generator[in_domain, :] @ guess - function[in_domain]
    coeffs = sparse.linalg.spsolve(a, b)
    return (
        guess
        + sparse.identity(len(weights), format="csr")[:, in_domain] @ coeffs
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
    coeffs = sparse.linalg.spsolve(a, b)

    weights = np.empty(generator.shape[0])
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
