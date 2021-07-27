import numpy as np
import scipy.sparse


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
    return scipy.sparse.csr_matrix(
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
    return scipy.sparse.csr_matrix(
        (data, (row_ind, col_ind)), shape=(p0.size, p0.size)
    )
