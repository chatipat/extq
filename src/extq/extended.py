import numpy as np


def moving_matmul(a, k):
    """Calculate a moving window of matrix products.

    Parameters
    ----------
    a : (m, n, n) ndarray
        Input time series of square matrices.
    k : int
        Size of the moving window.

    Returns
    -------
    (m - k + 1, n, n) ndarray
        Moving matrix product of the input time series.
        Each point is the product of k sequential matrices.

    """
    assert k >= 1
    if k == 1:
        return a

    in_len, dim, _ = a.shape
    out_len = in_len - k + 1
    n_slices = (out_len + k) // (k + 1)

    # pad end of input with identity matrices
    # so we don't have to special-case the points at the end
    padded = np.empty((n_slices * (k + 1) + k - 1, dim, dim), dtype=a.dtype)
    padded[:in_len] = a
    padded[in_len:] = np.identity(dim, dtype=a.dtype)

    # accumulate backward
    # backward_acc[m, :] is composed of elements
    #   n-k:n, n-k+1:n, ..., n-1:n, n:n
    # where n = m*(k+1) + k
    backward_slices = padded[: -(k - 1)].reshape(n_slices, k + 1, dim, dim)
    backward_acc = np.empty((n_slices, k + 1, dim, dim), dtype=a.dtype)
    backward_acc[:, k] = np.identity(dim, dtype=a.dtype)
    backward_acc[:, k - 1] = backward_slices[:, k - 1]
    for i in range(k - 2, -1, -1):
        np.matmul(
            backward_slices[:, i],
            backward_acc[:, i + 1],
            out=backward_acc[:, i],
        )

    # accumulate forward
    # forward_acc[m, :] is composed of elements
    #   n:n, n:n+1, ..., n:n+k-1, n:n+k
    # where n = m*(k+1) + k
    forward_slices = padded[k - 1 :].reshape(n_slices, k + 1, dim, dim)
    forward_acc = np.empty((n_slices, k + 1, dim, dim), dtype=a.dtype)
    forward_acc[:, 0] = np.identity(dim, dtype=a.dtype)
    forward_acc[:, 1] = forward_slices[:, 1]
    for i in range(2, k + 1):
        np.matmul(
            forward_acc[:, i - 1],
            forward_slices[:, i],
            out=forward_acc[:, i],
        )

    # combine the two accumulations to get
    #   n-k:n, n-k+1:n+1, ..., n-1:n+k-1, n:n+k
    # where n = m*(k+1) + k
    # then discard padding
    result = backward_acc @ forward_acc
    result = result.reshape(n_slices * (k + 1), dim, dim)
    result = result[:out_len]

    return result
