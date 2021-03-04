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
    if n_slices * (k + 1) != out_len:
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


@nb.njit
def moving_matmul_Am_B_An(a, b, k):
    """Calculate a moving window of matrix products.

    This function calculates sums of matrix products of the form

    .. math::

        \sum_{m+n=k-1} A(t) \dots A(t+m-1) B(t+m) A(t+m+1) \dots A(t+k)

    Parameters
    ----------
    a, b : (m, n, n) ndarray
        Input time series of square matrices. These arrays must be
        C-contiguous.
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
    assert b.ndim == 3
    assert a.shape[1] == b.shape[1]
    assert a.shape[1] == a.shape[2]
    assert b.shape[1] == b.shape[2]

    if k == 1:
        return b

    dtype = b.dtype
    in_len = b.shape[0]
    dim = b.shape[1]
    out_len = in_len - k + 1

    n_slices = out_len // (k + 1)  # number of full slices
    remaining = out_len % (k + 1)  # elements left over at the end

    # temporary arrays
    backward_acc_a = np.empty((k, dim, dim), dtype=dtype)
    backward_acc_b = np.empty((k, dim, dim), dtype=dtype)
    forward_acc_a = np.empty((k, dim, dim), dtype=dtype)
    forward_acc_b = np.empty((k, dim, dim), dtype=dtype)

    result = np.empty((out_len, dim, dim), dtype=dtype)

    # full slices
    for i in range(0, n_slices * (k + 1), k + 1):
        # accumulate backward
        # backward_acc is composed of elements
        #   i:i+k, i+1:i+k, ..., i+k-1:i+k
        backward_acc_a[k - 1] = a[i + k - 1]
        backward_acc_b[k - 1] = b[i + k - 1]
        for j in range(k - 2, -1, -1):
            backward_acc_a[j] = a[i + j] @ backward_acc_a[j + 1]
            backward_acc_b[j] = (
                a[i + j] @ backward_acc_b[j + 1]
                + b[i + j] @ backward_acc_a[j + 1]
            )

        # accumulate forward
        # forward_acc[m, :] is composed of elements
        #   i+k:i+k+1, ..., i+k:i+2k-1, i+k:i+2k
        forward_acc_a[0] = a[i + k]
        forward_acc_b[0] = b[i + k]
        for j in range(2, k + 1):
            forward_acc_a[j - 1] = forward_acc_a[j - 2] @ a[i + j + k - 1]
            forward_acc_b[j - 1] = (
                forward_acc_b[j - 2] @ a[i + j + k - 1]
                + forward_acc_a[j - 2] @ b[i + j + k - 1]
            )

        # combine the two accumulations to get
        #   i:i+k, i+1:i+k+1, ..., i+k-1:i+2k-1, i+k:i+2k
        result[i] = backward_acc_b[0]
        for j in range(1, k):
            result[i + j] = (
                backward_acc_a[j] @ forward_acc_b[j - 1]
                + backward_acc_b[j] @ forward_acc_a[j - 1]
            )
        result[i + k] = forward_acc_b[k - 1]

    # last partial slice
    if n_slices * (k + 1) != out_len:
        i = n_slices * (k + 1)

        # accumulate backward
        backward_acc_a[k - 1] = a[i + k - 1]
        backward_acc_b[k - 1] = b[i + k - 1]
        for j in range(k - 2, -1, -1):
            backward_acc_a[j] = a[i + j] @ backward_acc_a[j + 1]
            backward_acc_b[j] = (
                a[i + j] @ backward_acc_b[j + 1]
                + b[i + j] @ backward_acc_a[j + 1]
            )

        # accumulate forward
        # note this is truncated by the end of the input
        forward_acc_a[0] = a[i + k]
        forward_acc_b[0] = b[i + k]
        for j in range(2, min(remaining, k)):
            forward_acc_a[j - 1] = forward_acc_a[j - 2] @ a[i + j + k - 1]
            forward_acc_b[j - 1] = (
                forward_acc_b[j - 2] @ a[i + j + k - 1]
                + forward_acc_a[j - 2] @ b[i + j + k - 1]
            )

        # combine the two accumulations
        # note that forward_acc[k-1] isn't needed
        # because it's the last element of a full slice
        result[i] = backward_acc_b[0]
        for j in range(1, min(remaining, k)):
            result[i + j] = (
                backward_acc_a[j] @ forward_acc_b[j - 1]
                + backward_acc_b[j] @ forward_acc_a[j - 1]
            )

    return result


@nb.njit
def moving_matmul_Am_B_Cn(a, b, c, k):
    """Calculate a moving window of matrix products.

    This function calculates sums of matrix products of the form

    .. math::

        \sum_{m+n=k-1} A(t) \dots A(t+m-1) B(t+m) C(t+m+1) \dots C(t+k)

    Parameters
    ----------
    a, b, c : (m, n, n) ndarray
        Input time series of square matrices. These arrays must be
        C-contiguous.
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
    assert b.ndim == 3
    assert c.ndim == 3
    assert a.shape[1] == b.shape[1]
    assert b.shape[1] == c.shape[1]
    assert a.shape[1] == a.shape[2]
    assert b.shape[1] == b.shape[2]
    assert c.shape[1] == c.shape[2]

    if k == 1:
        return b

    dtype = b.dtype
    in_len = b.shape[0]
    dim = b.shape[1]
    out_len = in_len - k + 1

    n_slices = out_len // (k + 1)  # number of full slices
    remaining = out_len % (k + 1)  # elements left over at the end

    # temporary arrays
    backward_acc_a = np.empty((k, dim, dim), dtype=dtype)
    backward_acc_b = np.empty((k, dim, dim), dtype=dtype)
    backward_acc_c = np.empty((k, dim, dim), dtype=dtype)
    forward_acc_a = np.empty((k, dim, dim), dtype=dtype)
    forward_acc_b = np.empty((k, dim, dim), dtype=dtype)
    forward_acc_c = np.empty((k, dim, dim), dtype=dtype)

    result = np.empty((out_len, dim, dim), dtype=dtype)

    # full slices
    for i in range(0, n_slices * (k + 1), k + 1):
        # accumulate backward
        # backward_acc is composed of elements
        #   i:i+k, i+1:i+k, ..., i+k-1:i+k
        backward_acc_a[k - 1] = a[i + k - 1]
        backward_acc_b[k - 1] = b[i + k - 1]
        backward_acc_c[k - 1] = c[i + k - 1]
        for j in range(k - 2, -1, -1):
            backward_acc_a[j] = a[i + j] @ backward_acc_a[j + 1]
            backward_acc_b[j] = (
                a[i + j] @ backward_acc_b[j + 1]
                + b[i + j] @ backward_acc_c[j + 1]
            )
            backward_acc_c[j] = c[i + j] @ backward_acc_c[j + 1]

        # accumulate forward
        # forward_acc[m, :] is composed of elements
        #   i+k:i+k+1, ..., i+k:i+2k-1, i+k:i+2k
        forward_acc_a[0] = a[i + k]
        forward_acc_b[0] = b[i + k]
        forward_acc_c[0] = c[i + k]
        for j in range(2, k + 1):
            forward_acc_a[j - 1] = forward_acc_a[j - 2] @ a[i + j + k - 1]
            forward_acc_b[j - 1] = (
                forward_acc_a[j - 2] @ b[i + j + k - 1]
                + forward_acc_b[j - 2] @ c[i + j + k - 1]
            )
            forward_acc_c[j - 1] = forward_acc_c[j - 2] @ c[i + j + k - 1]

        # combine the two accumulations to get
        #   i:i+k, i+1:i+k+1, ..., i+k-1:i+2k-1, i+k:i+2k
        result[i] = backward_acc_b[0]
        for j in range(1, k):
            result[i + j] = (
                backward_acc_a[j] @ forward_acc_b[j - 1]
                + backward_acc_b[j] @ forward_acc_c[j - 1]
            )
        result[i + k] = forward_acc_b[k - 1]

    # last partial slice
    if n_slices * (k + 1) != out_len:
        i = n_slices * (k + 1)

        # accumulate backward
        backward_acc_a[k - 1] = a[i + k - 1]
        backward_acc_b[k - 1] = b[i + k - 1]
        backward_acc_c[k - 1] = c[i + k - 1]
        for j in range(k - 2, -1, -1):
            backward_acc_a[j] = a[i + j] @ backward_acc_a[j + 1]
            backward_acc_b[j] = (
                a[i + j] @ backward_acc_b[j + 1]
                + b[i + j] @ backward_acc_c[j + 1]
            )
            backward_acc_c[j] = c[i + j] @ backward_acc_c[j + 1]

        # accumulate forward
        # note this is truncated by the end of the input
        forward_acc_a[0] = a[i + k]
        forward_acc_b[0] = b[i + k]
        forward_acc_c[0] = c[i + k]
        for j in range(2, min(remaining, k)):
            forward_acc_a[j - 1] = forward_acc_a[j - 2] @ a[i + j + k - 1]
            forward_acc_b[j - 1] = (
                forward_acc_a[j - 2] @ b[i + j + k - 1]
                + forward_acc_b[j - 2] @ c[i + j + k - 1]
            )
            forward_acc_c[j - 1] = forward_acc_c[j - 2] @ c[i + j + k - 1]

        # combine the two accumulations
        # note that forward_acc[k-1] isn't needed
        # because it's the last element of a full slice
        result[i] = backward_acc_b[0]
        for j in range(1, min(remaining, k)):
            result[i + j] = (
                backward_acc_a[j] @ forward_acc_b[j - 1]
                + backward_acc_b[j] @ forward_acc_c[j - 1]
            )

    return result
