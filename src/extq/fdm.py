import numba as nb
import numpy as np
from scipy import sparse


def generator_reversible_2d(potential, kT, x, y):
    """
    Compute the generator matrix for a reversible 2D potential.

    Parameters
    ----------
    potential : (nx, ny) ndarray of float
        Potential energy for a 2D system.
    kT : float
        Temperature of the system, in units of energy.
    x : (nx,) ndarray of float
        X coordinates. Must be evenly spaced.
    y : (ny,) ndarray of float
        Y coordinates. Must be evenly spaced.

    Returns
    -------
    sparse matrix
        Generator matrix.

    """

    xsep = (x[-1] - x[0]) / (len(x) - 1)
    ysep = (y[-1] - y[0]) / (len(y) - 1)
    assert np.allclose(x[1:] - x[:-1], xsep)
    assert np.allclose(y[1:] - y[:-1], ysep)

    shape = (len(x), len(y))
    ind = np.ravel_multi_index(np.ogrid[: len(x), : len(y)], shape)

    # possible transitions per step
    transitions = [
        (np.s_[:-1, :], np.s_[1:, :], xsep),
        (np.s_[1:, :], np.s_[:-1, :], xsep),
        (np.s_[:, :-1], np.s_[:, 1:], ysep),
        (np.s_[:, 1:], np.s_[:, :-1], ysep),
    ]

    return _generator_reversible_helper(transitions, potential, kT, ind, shape)


def generator_reversible_3d(potential, kT, x, y, z):
    """
    Compute the generator matrix for a reversible 3D potential.

    Parameters
    ----------
    potential : (nx, ny, nz) ndarray of float
        Potential energy for a 3D system.
    kT : float
        Temperature of the system, in units of energy.
    x : (nx,) ndarray of float
        X coordinates. Must be evenly spaced.
    y : (ny,) ndarray of float
        Y coordinates. Must be evenly spaced.
    z : (nz,) ndarray of float
        Z coordinates. Must be evenly spaced.

    Returns
    -------
    sparse matrix
        Generator matrix.

    """

    xsep = (x[-1] - x[0]) / (len(x) - 1)
    ysep = (y[-1] - y[0]) / (len(y) - 1)
    zsep = (z[-1] - z[0]) / (len(z) - 1)
    assert np.allclose(x[1:] - x[:-1], xsep)
    assert np.allclose(y[1:] - y[:-1], ysep)
    assert np.allclose(z[1:] - z[:-1], zsep)

    shape = (len(x), len(y), len(z))
    ind = np.ravel_multi_index(np.ogrid[: len(x), : len(y), : len(z)], shape)

    # possible transitions per step
    transitions = [
        (np.s_[:-1, :, :], np.s_[1:, :, :], xsep),
        (np.s_[1:, :, :], np.s_[:-1, :, :], xsep),
        (np.s_[:, :-1, :], np.s_[:, 1:, :], ysep),
        (np.s_[:, 1:, :], np.s_[:, :-1, :], ysep),
        (np.s_[:, :, :-1], np.s_[:, :, 1:], zsep),
        (np.s_[:, :, 1:], np.s_[:, :, :-1], zsep),
    ]

    return _generator_reversible_helper(transitions, potential, kT, ind, shape)


def _generator_reversible_helper(transitions, u, kT, ind, shape):
    data = []
    row_ind = []
    col_ind = []
    p0 = np.zeros(shape)

    # transitioning to adjacent cell
    for row, col, sep in transitions:
        p = (2.0 * kT / sep ** 2) / (1.0 + np.exp((u[col] - u[row]) / kT))
        p0[row] -= p
        data.append(p.ravel())
        row_ind.append(ind[row].ravel())
        col_ind.append(ind[col].ravel())

    # not transitioning
    data.append(p0.ravel())
    row_ind.append(ind.ravel())
    col_ind.append(ind.ravel())

    data = np.concatenate(data)
    row_ind = np.concatenate(row_ind)
    col_ind = np.concatenate(col_ind)
    return sparse.csr_matrix(
        (data, (row_ind, col_ind)), shape=(p0.size, p0.size)
    )


def generator_irreversible_2d(
    drift_x, drift_y, diffusion_x, diffusion_y, x, y
):
    """Compute the generator matrix for an irreversible 2D potential.

    Parameters
    ----------
    drift_x, drift_y : (nx, ny) ndarray of float
        Drift for a 2D system.
    diffusion_x, diffusion_y : (nx, ny) ndarray of float
        Diffusion for a 2D system.
    x : (nx,) ndarray of float
        X coordinates. Must be evenly spaced.
    y : (ny,) ndarray of float
        Y coordinates. Must be evenly spaced.

    Returns
    -------
    sparse matrix
        Generator matrix.

    """

    xsep = (x[-1] - x[0]) / (len(x) - 1)
    ysep = (y[-1] - y[0]) / (len(y) - 1)
    assert np.allclose(x[1:] - x[:-1], xsep)
    assert np.allclose(y[1:] - y[:-1], ysep)

    shape = (len(x), len(y))
    ind = np.ravel_multi_index(np.ogrid[: len(x), : len(y)], shape)

    # possible transitions per step
    transitions = [
        (np.s_[:-1, :], np.s_[1:, :], drift_x, diffusion_x, xsep),
        (np.s_[1:, :], np.s_[:-1, :], drift_x, diffusion_x, -xsep),
        (np.s_[:, :-1], np.s_[:, 1:], drift_y, diffusion_y, ysep),
        (np.s_[:, 1:], np.s_[:, :-1], drift_y, diffusion_y, -ysep),
    ]

    return _generator_irreversible_helper(transitions, ind, shape)


def generator_irreversible_3d(
    drift_x, drift_y, drift_z, diffusion_x, diffusion_y, diffusion_z, x, y, z
):
    """Compute the generator matrix for an irreversible 3D potential.

    Parameters
    ----------
    drift_x, drift_y, drift_z : (nx, ny, nz) ndarray of float
        Drift for a 3D system.
    diffusion_x, diffusion_y, diffusion_z : (nx, ny, nz) ndarray of float
        Diffusion for a 3D system.
    x : (nx,) ndarray of float
        X coordinates. Must be evenly spaced.
    y : (ny,) ndarray of float
        Y coordinates. Must be evenly spaced.
    z : (nz,) ndarray of float
        Z coordinates. Must be evenly spaced.

    Returns
    -------
    sparse matrix
        Generator matrix.

    """

    xsep = (x[-1] - x[0]) / (len(x) - 1)
    ysep = (y[-1] - y[0]) / (len(y) - 1)
    zsep = (z[-1] - z[0]) / (len(z) - 1)
    assert np.allclose(x[1:] - x[:-1], xsep)
    assert np.allclose(y[1:] - y[:-1], ysep)
    assert np.allclose(z[1:] - z[:-1], zsep)

    shape = (len(x), len(y), len(z))
    ind = np.ravel_multi_index(np.ogrid[: len(x), : len(y), : len(z)], shape)

    # possible transitions per step
    transitions = [
        (np.s_[:-1, :, :], np.s_[1:, :, :], drift_x, diffusion_x, xsep),
        (np.s_[1:, :, :], np.s_[:-1, :, :], drift_x, diffusion_x, -xsep),
        (np.s_[:, :-1, :], np.s_[:, 1:, :], drift_y, diffusion_y, ysep),
        (np.s_[:, 1:, :], np.s_[:, :-1, :], drift_y, diffusion_y, -ysep),
        (np.s_[:, :, :-1], np.s_[:, :, 1:], drift_z, diffusion_z, zsep),
        (np.s_[:, :, 1:], np.s_[:, :, :-1], drift_z, diffusion_z, -zsep),
    ]

    return _generator_irreversible_helper(transitions, ind, shape)


def _generator_irreversible_helper(transitions, ind, shape):
    data = []
    row_ind = []
    col_ind = []
    p0 = np.zeros(shape)

    # transitioning to adjacent cell
    for row, col, drift, diffusion, sep in transitions:
        p = 0.5 * drift[row] / sep + diffusion[row] / sep ** 2
        p0[row] -= p
        data.append(p.ravel())
        row_ind.append(ind[row].ravel())
        col_ind.append(ind[col].ravel())

    # not transitioning
    data.append(p0.ravel())
    row_ind.append(ind.ravel())
    col_ind.append(ind.ravel())

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


def extended_forward_committor(
    generator, weights, transitions, in_domain, guess
):
    """Compute the forward extended committor.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    weights : (n_points,) ndarray of float
        Change of measure to the invariant distribution for each point.
    transitions : (n_indices, n_indices) array-like
        Possible transitions between indices. Each element
        transitions[i, j] may be a scalar or a sparse matrix of shape
        (n_points, n_points).
    in_domain : (n_indices, n_points) ndarray of bool
        Whether each point is in the domain.
    guess : (n_indices, n_points) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.

    Returns
    -------
    (n_indices, n_points) ndarray of float
        Forward extended committor at each point.

    """
    gen = sparse.bmat(
        [[generator.multiply(mij) for mij in mi] for mi in transitions],
        format="csr",
    )
    pi = np.concatenate([weights] * len(transitions))
    d = np.concatenate(in_domain)
    g = np.concatenate(guess)
    qp = forward_committor(gen, pi, d, g)
    return qp.reshape(len(transitions), len(weights))


def extended_backward_committor(
    generator, weights, transitions, in_domain, guess
):
    """Compute the backward extended committor.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    weights : (n_points,) ndarray of float
        Change of measure to the invariant distribution for each point.
    transitions : (n_indices, n_indices) array-like
        Possible transitions between indices. Each element
        transitions[i, j] may be a scalar or a sparse matrix of shape
        (n_points, n_points).
    in_domain : (n_indices, n_points) ndarray of bool
        Whether each point is in the domain.
    guess : (n_indices, n_points) ndarray of float
        Guess for the extended committor. Must obey boundary conditions.

    Returns
    -------
    (n_indices, n_points) ndarray of float
        Backward extended committor at each point.

    """
    gen = sparse.bmat(
        [[generator.multiply(mij) for mij in mi] for mi in transitions],
        format="csr",
    )
    pi = np.concatenate([weights] * len(transitions))
    d = np.concatenate(in_domain)
    g = np.concatenate(guess)
    qm = backward_committor(gen, pi, d, g)
    return qm.reshape(len(transitions), len(weights))


def extended_rate(
    generator, forward_q, backward_q, weights, transitions, rxn_coords=None
):
    """Compute the TPT rate using extended committors.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    forward_q : (n_indices, n_points) ndarray of float
        Forward extended committor at each point.
    backward_q : (n_indices, n_points) ndarray of float
        Backward extended committor at each point.
    weights : (n_points,) ndarray of float.
        Change of measure to the invariant distribution at each point.
    rxn_coords : (n_indices, n_points) ndarray of float, optional
        Reaction coordinate at each point. This must be zero in the
        reactant state and one in the product state. If None, estimate
        the rate without using a reaction coordinate.

    Returns
    -------
    float
        TPT rate.

    """
    gen = sparse.bmat(
        [[generator.multiply(mij) for mij in mi] for mi in transitions],
        format="csr",
    )
    qp = np.concatenate(forward_q)
    qm = np.concatenate(backward_q)
    pi = np.concatenate([weights] * len(transitions))
    if rxn_coords is None:
        h = None
    else:
        h = np.concatenate(rxn_coords)
    r = rate(gen, qp, qm, pi, h)
    return r * len(transitions)


def extended_current(
    generator, forward_q, backward_q, weights, transitions, cv
):
    """Compute the reactive current using extended committors.

    Parameters
    ----------
    generator : (n_points, n_points) sparse matrix of float
        Generator matrix.
    forward_q : (n_indices, n_points) ndarray of float
        Forward extended committor at each point.
    backward_q : (n_indices, n_points) ndarray of float
        Backward extended committor at each point.
    weights : (n_points,) ndarray of float.
        Change of measure to the invariant distribution at each point.
    transitions : (n_indices, n_indices) array-like
        Possible transitions between indices. Each element
        transitions[i, j] may be a scalar or a sparse matrix of shape
        (n_points, n_points).
    rxn_coords : (n_indices, n_points) ndarray of float
        Collective variable at each point.

    Returns
    -------
    (n_indices, n_points) ndarray of float
        Reactive current at each point.

    """
    gen = sparse.bmat(
        [[generator.multiply(mij) for mij in mi] for mi in transitions],
        format="csr",
    )
    qp = np.concatenate(forward_q)
    qm = np.concatenate(backward_q)
    pi = np.concatenate([weights] * len(transitions))
    h = np.concatenate(cv)
    j = current(gen, qp, qm, pi, h)
    return j.reshape(len(transitions), len(weights)) * len(transitions)
