"""Functions for computing memory and related matrices."""

from .. import linalg

__all__ = ["memory", "extrapolate"]


def memory(mats):
    """
    Compute memory matrices from correlation matrices.

    Parameters
    ---------
    mats : sequence of (n_basis, n_basis) {ndarray, sparse matrix} of float
        Sequence of correlation matrices at equally-spaced lag times,
        starting at a lag time of zero. Note that ``len(mats) >= 2``.

    Returns
    -------
    mems : list of (n_basis, n_basis) {ndarray, sparse matrix} of float
        List of memory matrices at equally-spaced lag times (with the
        same spacing as `mats`), starting at a lag time of zero.
        Note that ``len(mems) = len(mats) - 2``.

    """
    inv = linalg.inv(mats[0])
    tmat = inv @ mats[1]
    temp = []
    mems = []
    for t in range(len(mats) - 2):
        mems.append(
            mats[t + 2]
            - mats[t + 1] @ tmat
            - sum(mats[t - s] @ temp[s] for s in range(t))
        )
        temp.append(inv @ mems[t])
    return mems


def extrapolate(mats, mems, lag):
    """
    Extrapolate correlation matrices to longer lag times.

    Parameters
    ----------
    mats : sequence of (n_basis, n_basis) {ndarray, sparse matrix} of float
        Sequence of correlation matrices at equally-spaced lag times,
        starting at a lag time of zero.
    mems : sequence of (n_basis, n_basis) {ndarray, sparse matrix} of float, optional
        Sequence of memory matrices at equally-spaced lag times (with
        the same spacing as `mats`), starting at a lag time of zero.
    lag : int
        Maximum lag time up to which to extrapolate.

    Returns
    -------
    xmats : list of (n_basis, n_basis) {ndarray, sparse matrix} of float
        Length ``lag+1`` list of extrapolated correlation matrices. The
        first ``len(mats)`` matrices of `xmats` is taken from `mats`
        and the subsequent matrices are extrapolated from `mats` using
        `mems`.

    """
    inv = linalg.inv(mats[0])
    tmat = inv @ mats[1]
    temp = [inv @ mem for mem in mems]
    xmats = []
    for t in range(lag + 1):
        if t < len(mats):
            xmats.append(mats[t])
        elif t < len(mems) + 2:
            xmats.append(
                xmats[t - 1] @ tmat
                + mems[t - 2]
                + sum(xmats[t - s - 2] @ temp[s] for s in range(t - 2))
            )
        else:
            xmats[t].append(
                xmats[t - 1] @ tmat
                + sum(xmats[t - s - 2] @ temp[s] for s in range(len(mems)))
            )
    return xmats
