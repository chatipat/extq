import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def moving_matmul(a, k):
    """Calculate a moving window of matrix products.

    Note that this function modifies the input array in-place.

    Parameters
    ----------
    a : (m, n, n) ndarray
        Input time series of square matrices. This array is also used
        for the output, and must be C-contiguous.
    k : int
        Size of the moving window.

    Returns
    -------
    (m - k + 1, n, n) ndarray
        Moving matrix product of the input time series.
        Each point is the product of k sequential matrices.

    """
    assert k >= 1
    assert a.ndim == 3
    assert a.shape[1] == a.shape[2]

    if k == 1:
        return a

    in_len = a.shape[0]
    dim = a.shape[1]
    out_len = in_len - k + 1

    n_slices = out_len // (k + 1)  # number of full slices
    remaining = out_len % (k + 1)  # elements left over at the end

    # temporary arrays
    backward_acc = np.empty((k, dim, dim), dtype=a.dtype)
    forward_acc = np.empty((k, dim, dim), dtype=a.dtype)

    # full slices
    for i in range(0, n_slices * (k + 1), k + 1):
        # accumulate backward
        # backward_acc is composed of elements
        #   i:i+k, i+1:i+k, ..., i+k-1:i+k
        backward_acc[k - 1] = a[i + k - 1]
        for j in range(k - 2, -1, -1):
            np.dot(
                a[i + j],
                backward_acc[j + 1],
                backward_acc[j],
            )

        # accumulate forward
        # forward_acc[m, :] is composed of elements
        #   i+k:i+k+1, ..., i+k:i+2k-1, i+k:i+2k
        forward_acc[0] = a[i + k]
        for j in range(2, k + 1):
            np.dot(
                forward_acc[j - 2],
                a[i + j + k - 1],
                forward_acc[j - 1],
            )

        # combine the two accumulations to get
        #   i:i+k, i+1:i+k+1, ..., i+k-1:i+2k-1, i+k:i+2k
        a[i] = backward_acc[0]
        for j in range(1, k):
            np.dot(
                backward_acc[j],
                forward_acc[j - 1],
                a[i + j],
            )
        a[i + k] = forward_acc[k - 1]

    # last partial slice
    if remaining > 0:
        i = n_slices * (k + 1)

        # accumulate backward
        backward_acc[k - 1] = a[i + k - 1]
        for j in range(k - 2, -1, -1):
            np.dot(
                a[i + j],
                backward_acc[j + 1],
                backward_acc[j],
            )

        # accumulate forward
        # note this is truncated by the end of the input
        if remaining > 1:
            forward_acc[0] = a[i + k]
        for j in range(2, min(remaining, k)):
            np.dot(
                forward_acc[j - 2],
                a[i + j + k - 1],
                forward_acc[j - 1],
            )

        # combine the two accumulations
        # note that forward_acc[k-1] isn't needed
        # because it's the last element of a full slice
        a[i] = backward_acc[0]
        for j in range(1, min(remaining, k)):
            np.dot(
                backward_acc[j],
                forward_acc[j - 1],
                a[i + j],
            )

    return a[:out_len]


def forward_modified_transitions(m, d, f, g, lag, dtype=np.float64):
    ni, _, nt = m.shape
    assert m.shape == (ni, ni, nt)
    assert d.shape == (ni, nt + 1)
    assert f.shape == (ni, ni, nt)
    assert g.shape == (ni, nt + 1)
    r = _forward_modified_transitions_helper(ni, nt, m, d, f, g, dtype)
    r = moving_matmul(r, lag)
    result = np.moveaxis(r, 0, -1)
    return result


@nb.njit(fastmath=True)
def _forward_modified_transitions_helper(ni, nt, m, d, f, g, dtype):
    r = np.zeros((nt, ni + 1, ni + 1), dtype=dtype)
    for n in range(nt):
        for i in range(ni):
            if d[i, n]:
                for j in range(ni):
                    r[n, i, j] = m[i, j, n]
                    r[n, i, ni] += m[i, j, n] * f[i, j, n]  # integral
            else:
                r[n, i, ni] = g[i, n]  # boundary conditions
        r[n, ni, ni] = 1.0
    return r


def backward_modified_transitions(m, d, f, g, lag, dtype=np.float64):
    ni, _, nt = m.shape
    assert m.shape == (ni, ni, nt)
    assert d.shape == (ni, nt + 1)
    assert f.shape == (ni, ni, nt)
    assert g.shape == (ni, nt + 1)
    r = _backward_modified_transitions_helper(ni, nt, m, d, f, g, dtype)
    r = moving_matmul(r, lag)
    result = np.moveaxis(r, 0, -1)
    return result


@nb.njit(fastmath=True)
def _backward_modified_transitions_helper(ni, nt, m, d, f, g, dtype):
    r = np.zeros((nt, ni + 1, ni + 1), dtype=dtype)
    for n in range(nt):
        for j in range(ni):
            if d[j, n + 1]:
                for i in range(ni):
                    r[n, i, j] = m[i, j, n]
                    r[n, ni, j] += m[i, j, n] * f[i, j, n]  # integral
            else:
                r[n, ni, j] = g[j, n + 1]  # boundary conditions
        r[n, ni, ni] = 1.0
    return r
