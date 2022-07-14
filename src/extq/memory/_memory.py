from .. import linalg


def identity(mats, mems=[]):
    return mats[0] + sum((s + 1) * mems[s] for s in range(len(mems)))


def generator(mats, mems=[]):
    """Compute the `magic' generator + memory operator
    from a series of correlation matrices.
    The operator is defined as
    .. math:: A + \sum_{k=0}^KM_k
    where :math:`A` is the generator and :math:`M_k` are the
    memory matrices.

    Parameters
    ----------
    mats : array-like of ndarray (n_basis, n_basis) of float
        Correlation matrices
    mems : array-like of ndarray (n_basis, n_basis) of float, optional
        Memory matrices

    Returns
    -------
    ndarray (n_basis, n_basis) of float
        Generator operator + memory
    """
    return mats[1] - mats[0] + sum(mems[s] for s in range(len(mems)))


def memory(mats):
    """Compute the memory matrices from correlation matrices.

    Parameters
    ---------
    mats : array-like (n_mats,) of ndarray (n_basis, n_basis) of float
        Correlation matrices

    Returns
    -------
    mems : array-like (n_mats - 2,) of ndarray (n_basis, n_basis) of float
        Memory matrices
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
    """Extrapolate correlation matrices to longer lag time
    using memory matrices.

    Parameters
    ----------
    mats : array-like of ndarray (n_basis, n_basis) of float
        Correlation matrices
    mems : array-like of ndarray (n_basis, n_basis) of float
        Memory matrices
    lag : int

    Returns
    -------
    xmats : list (lag + 1,) of ndarray (n_basis, n_basis) of float
        Extrapolated correlation matrices
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
