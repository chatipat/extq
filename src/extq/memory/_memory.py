"""Functions for computing memory and related matrices."""

from .. import linalg


def identity(mats, mems=[]):
    r"""
    Compute the memory-corrected zero-lag-time correlation matrix.

    The memory-corrected zero-lag-time correlation matrix is defined as

    ..math :: C_0 + \sum_{k=0}^{K-1} (k+1) M_k

    where :math:`C_0` is the zero-lag-time correlation matrix
    ``mats[0]`` and :math:`M_k` are memory matrices ``mems[k]``.

    Parameters
    ----------
    mats : sequence of (n_basis, n_basis) ndarray of float
        Sequence of correlation matrices at equally-spaced lag times,
        starting at a lag time of zero. Note that only ``mats[0]`` is
        used.
    mems : sequence of (n_basis, n_basis) ndarray of float, optional
        Sequence of memory matrices at equally-spaced lag times (with
        the same spacing as `mats`), starting at a lag time of zero.

    Returns
    -------
    (n_basis, n_basis) ndarray of float
        Memory-corrected zero-lag-time correlation matrix.

    """
    return mats[0] + sum((s + 1) * mems[s] for s in range(len(mems)))


def generator(mats, mems=[]):
    r"""
    Compute the memory-corrected generator matrix.

    The memory-corrected generator is defined as

    ..math :: A + \sum_{k=0}^{K-1} M_k

    where :math:`A` is the generator matrix ``mats[1]-mats[0]`` and
    :math:`M_k` are memory matrices ``mems[k]``.

    Parameters
    ----------
    mats : sequence of (n_basis, n_basis) {ndarray, sparse matrix} of float
        Sequence of correlation matrices at equally-spaced lag times,
        starting at a lag time of zero. Note that only ``mats[0]`` and
        ``mats[1]`` are used.
    mems : sequence of (n_basis, n_basis) {ndarray, sparse matrix} of float, optional
        Sequence of memory matrices at equally-spaced lag times (with
        the same spacing as `mats`), starting at a lag time of zero.

    Returns
    -------
    (n_basis, n_basis) ndarray of float
        Memory-corrected generator matrix.

    """
    return mats[1] - mats[0] + sum(mems[s] for s in range(len(mems)))


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
