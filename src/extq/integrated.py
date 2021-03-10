import numba as nb
import numpy as np


def moving_matmul_sum(a, b, k):
    """Calculate a moving window for extended integrated DGA.

    This function calculates

    .. math::

        c_i = b_i + b_{i+1} a_i + \dots + b_{i+k-1} a_{i+k-2} \dots a_i

    Note that this function modifies the input array in-place.

    Parameters
    ----------
    a : (m - 1, n, n) ndarray
        Input time series of square matrices. This array must be
        C-contiguous.
    b : (m, l, n) ndarray
        Input time series of matrices. This array is also used for the
        output, and must be C-contiguous.
    k : int
        Size of the moving window. This is usually lag+1.

    Returns
    -------
    (m - k + 1, n, n) ndarray
        Moving window for the input time series.

    """
    assert k >= 1
    assert a.ndim == 3
    assert b.ndim == 3
    assert a.shape[0] + 1 == b.shape[0]
    assert a.shape[1] == a.shape[2]
    assert a.shape[1] == b.shape[2]

    if k == 1:
        return b

    in_len = b.shape[0]
    dim1 = b.shape[1]
    dim2 = b.shape[2]
    out_len = in_len - k + 1

    n_slices = out_len // (k + 1)  # number of full slices
    remaining = out_len % (k + 1)  # elements left over at the end

    # temporary arrays
    backward_acc_a = np.empty((k - 1, dim2, dim2))
    backward_acc_b = np.empty((k, dim1, dim2))
    forward_acc_a = np.empty((k - 1, dim2, dim2))
    forward_acc_b = np.empty((k, dim1, dim2))

    # full slices
    for i in range(0, n_slices * (k + 1), k + 1):
        # backward_acc_a is composed of elements
        #   i+1:i+k, i+2:i+k, ..., i+k-1:i+k
        # it is used when combining backward_acc_b and forward_acc_b
        backward_acc_a[k - 2] = a[i + k - 1]
        for j in range(k - 2, 0, -1):
            backward_acc_a[j - 1] = backward_acc_a[j] @ a[i + j]

        # backward_acc_b is composed of elements
        #   i:i+k, i+1:i+k, ..., i+k-1:i+k
        backward_acc_b[k - 1] = b[i + k - 1]
        for j in range(k - 2, -1, -1):
            backward_acc_b[j] = backward_acc_b[j + 1] @ a[i + j] + b[i + j]

        # forward_acc_a is composed of elements
        #   i+k:i+k+1, i+k:i+k+2, ..., i+k:i+2k-2
        # it is used to compute forward_acc_b
        forward_acc_a[0] = a[i + k]
        for j in range(2, k):
            forward_acc_a[j - 1] = a[i + j + k - 1] @ forward_acc_a[j - 2]

        # forward_acc_b is composed of elements
        #   i+k:i+k+1, ..., i+k:i+2k-1, i+k:i+2k
        forward_acc_b[0] = b[i + k]
        for j in range(2, k + 1):
            forward_acc_b[j - 1] = (
                b[i + j + k - 1] @ forward_acc_a[j - 2] + forward_acc_b[j - 2]
            )

        # combine the accumulations to get
        #   i:i+k, i+1:i+k+1, ..., i+k-1:i+2k-1, i+k:i+2k
        b[i] = backward_acc_b[0]
        for j in range(1, k):
            b[i + j] = (
                forward_acc_b[j - 1] @ backward_acc_a[j - 1]
                + backward_acc_b[j]
            )
        b[i + k] = forward_acc_b[k - 1]

    # last partial slice
    if remaining > 0:
        i = n_slices * (k + 1)

        # only backward_acc_b[0] needed for one remaining element
        if remaining > 1:
            backward_acc_a[k - 2] = a[i + k - 1]
            for j in range(k - 2, 0, -1):
                backward_acc_a[j - 1] = backward_acc_a[j] @ a[i + j]

        backward_acc_b[k - 1] = b[i + k - 1]
        for j in range(k - 2, -1, -1):
            backward_acc_b[j] = backward_acc_b[j + 1] @ a[i + j] + b[i + j]

        # truncated by the end of the input
        if remaining > 2:
            forward_acc_a[0] = a[i + k]
        for j in range(2, min(remaining, k) - 1):
            forward_acc_a[j - 1] = a[i + j + k - 1] @ forward_acc_a[j - 2]

        # truncated by the end of the input
        if remaining > 1:
            forward_acc_b[0] = b[i + k]
        for j in range(2, min(remaining, k)):
            forward_acc_b[j - 1] = (
                b[i + j + k - 1] @ forward_acc_a[j - 2] + forward_acc_b[j - 2]
            )

        b[i] = backward_acc_b[0]
        for j in range(1, min(remaining, k)):
            b[i + j] = (
                forward_acc_b[j - 1] @ backward_acc_a[j - 1]
                + backward_acc_b[j]
            )

    return b[:out_len]
