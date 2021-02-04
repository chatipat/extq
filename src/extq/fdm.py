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
    dt = 8.0 * kT / sep ** 2

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
    dt = 12.0 * kT / sep ** 2

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
