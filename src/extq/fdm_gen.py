import numpy as np
import scipy as sp

__all__ = [
    "generator_from_potential",
    "generator_reversible_1d",
    "generator_reversible_2d",
    "generator_reversible_3d",
    "generator_irreversible_1d",
    "generator_irreversible_2d",
    "generator_irreversible_3d",
]


def generator_from_potential(
    U, kT=1.0, D=None, spacing=1.0, periodic=False, acceptance_ratio=-1
):
    """
    Compute the generator matrix from a potential.

    This function assumes that the system is a time-reversible
    drift-diffusion process.

    Parameters
    ----------
    U : array of float
        Potential energy of the system.
    kT : float
        Temperature of the system, in units of energy.
    D : float, optional
        Diffusion constant. If None (default), set to `kT`.
    spacing : {float, 1D array of float}, optional
        Grid spacing.
    periodic : {bool, 1D array of bool}, optional
        Whether each dimension is periodic. Non-periodic dimensions
        are assumed to have infinite potential outside of the box.
    acceptance_ratio : {callable, int, float}, optional
        Function that satisfies ``f(x)/f(-x) == exp(-x)``. If a number,
        computes the power mean of 1 and ``exp(-x)``.

    Returns
    -------
    sparse array
        Generator matrix.

    """
    size = U.size
    shape = U.shape
    shape_arr = np.array(shape)[:, None]
    ndim = len(shape)

    if D is None:
        D = kT
    spacing = np.broadcast_to(spacing, ndim)
    periodic = np.broadcast_to(periodic, ndim)

    # moves (+1 or -1 in each axis)
    unit_shifts = np.identity(ndim, dtype=int)
    moves = np.concatenate([unit_shifts, -unit_shifts], axis=1)

    # flattened grid of all coordinates
    grid = np.indices(shape).reshape(ndim, -1)

    # source and destination coordinates for each possible move
    src = np.repeat(grid, moves.shape[1], axis=1)
    dst = src + np.tile(moves, (1, size))

    # distance for each transition
    h = np.tile(spacing, 2 * size)

    # wrap periodic axes
    dst[periodic] %= shape_arr[periodic]

    # remove transitions that leave the box (for non-periodic axes)
    valid = np.all((dst >= 0) & (dst < shape_arr), axis=0)
    src = src[:, valid]
    dst = dst[:, valid]
    h = h[valid]

    # multi-index must be a tuple of arrays
    src = tuple(src)
    dst = tuple(dst)

    # transition rates
    dU = U[dst] - U[src]
    if callable(acceptance_ratio):
        # acceptance_ratio(x) / acceptance_ratio(-x) == exp(-x)
        rate = (D / h**2) * acceptance_ratio(dU / kT)
    else:
        p = acceptance_ratio
        if p == 0:
            rate = (D / h**2) * np.exp(-0.5 * dU / kT)
        elif p == np.inf:
            rate = (D / h**2) * np.max(1, np.exp(-dU / kT))
        elif p == -np.inf:
            rate = (D / h**2) * np.min(1, np.exp(-dU / kT))
        else:
            rate = (D / h**2) * (0.5 * (1 + np.exp(-p * dU / kT))) ** (1 / p)

    # convert to flat indices (scipy.sparse only supports 2D)
    src = np.ravel_multi_index(src, shape)
    dst = np.ravel_multi_index(dst, shape)

    # subtract row sum from diagonal
    data = np.concatenate([rate, -rate])
    row = np.concatenate([src, src])
    col = np.concatenate([dst, src])

    return sp.sparse.csr_array((data, (row, col)), shape=(size, size))


def generator_reversible_1d(potential, kT, x):
    """Compute the generator matrix for a reversible 1D potential.

    Parameters
    ----------
    potential : (nx,) ndarray of float
        Potential energy for a 1D system.
    kT : float
        Temperature of the system, in units of energy.
    x : (nx,) ndarray of float
        X coordinates. Must be evenly spaced.

    Returns
    -------
    sparse matrix
        Generator matrix.

    """

    xsep = (x[-1] - x[0]) / (len(x) - 1)
    assert np.allclose(x[1:] - x[:-1], xsep)

    shape = (len(x),)
    ind = np.arange(len(x))

    # possible transitions per step
    transitions = [
        (np.s_[:-1], np.s_[1:], xsep),
        (np.s_[1:], np.s_[:-1], xsep),
    ]

    return _generator_reversible_helper(transitions, potential, kT, ind, shape)


def generator_reversible_2d(potential, kT, x, y):
    """Compute the generator matrix for a reversible 2D potential.

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
    """Compute the generator matrix for a reversible 3D potential.

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
        p = (2.0 * kT / sep**2) / (1.0 + np.exp((u[col] - u[row]) / kT))
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
    return sp.sparse.csr_matrix(
        (data, (row_ind, col_ind)), shape=(p0.size, p0.size)
    )


def generator_irreversible_1d(drift, diffusion, x):
    """Compute the generator matrix for an irreversible 1D potential.

    Parameters
    ----------
    drift : (nx,) ndarray of float
        Drift for a 1D system.
    diffusion : (nx,) ndarray of float
        Diffusion for a 1D system.
    x : (nx,) ndarray of float
        X coordinates. Must be evenly spaced.

    Returns
    -------
    sparse matrix
        Generator matrix.

    """

    xsep = (x[-1] - x[0]) / (len(x) - 1)
    assert np.allclose(x[1:] - x[:-1], xsep)

    shape = (len(x),)
    ind = np.arange(len(x))

    # possible transitions per step
    transitions = [
        (np.s_[:-1, :], np.s_[1:, :], drift, diffusion, xsep),
        (np.s_[1:, :], np.s_[:-1, :], drift, diffusion, -xsep),
    ]

    return _generator_irreversible_helper(transitions, ind, shape)


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
        p = 0.5 * drift[row] / sep + diffusion[row] / sep**2
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
    return sp.sparse.csr_matrix(
        (data, (row_ind, col_ind)), shape=(p0.size, p0.size)
    )
