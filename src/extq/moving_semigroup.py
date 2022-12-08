import numba as nb
import numpy as np


@nb.njit
def moving_semigroup(a, k, f, *args):
    """
    Calculate a moving window of an associative binary operation.

    Note that this function modifies the input array in-place.

    Parameters
    ----------
    a : (m, ...) ndarray
        Input time series. This must be at least 2D. This array is also
        used for the output, and must be C-contiguous.
    k : int
        Size of the moving window.
    f : callable
        Associative binary operation taking two input arguments and
        one output argument. This must be Numba-compiled.
    *args
        Additional arguments to `f`, if needed.

    Returns
    -------
    (m - k + 1, ...) ndarray
        Output time series. Each output point is the result of
        `k` sequential input points reduced using the operation.

    """
    assert k >= 1
    assert a.ndim >= 2  # indexing a 1D ndarray yields a scalar

    if k == 1:
        return a

    out_len = a.shape[0] - k + 1
    assert out_len >= 0

    # temporary arrays
    acc = np.empty((k,) + a.shape[1:], dtype=a.dtype)

    for n in range(out_len):
        j = n % (k + 1)

        # backward and forward accumulations
        if j == 0:
            for jj in range(k - 1, -1, -1):
                if jj == k - 1:
                    acc[jj] = a[n + jj]
                else:
                    f(a[n + jj], acc[jj + 1], acc[jj], *args)
        elif j == 1:
            acc[j - 1] = a[n + k - 1]
        else:
            f(acc[j - 2], a[n + k - 1], acc[j - 1], *args)

        # combine accumulations
        if j == 0:
            a[n] = acc[j]
        elif j == k:
            a[n] = acc[j - 1]
        else:
            f(acc[j], acc[j - 1], a[n], *args)

    return a[:out_len]


def moving_matmul(a, k):
    """
    Calculate a moving matrix product.

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
        Output time series. Each output point is matrix product of `k`
        sequential input points.

    """
    assert a.ndim == 3 and a.shape[1] == a.shape[2]
    return moving_semigroup(a, k, _choose_mm(a.shape[1]))


def _choose_mm(n):
    assert n > 0
    if n == 1:
        return mm1
    elif n == 2:
        return mm2
    elif n == 3:
        return mm3
    elif n == 4:
        return mm4
    else:
        return mm


@nb.njit(fastmath=True)
def mm(a, b, c):
    np.dot(a, b, c)


@nb.njit(fastmath=True)
def mm1(a, b, c):
    c_0_0 = a[0, 0] * b[0, 0]
    c[0, 0] = c_0_0


@nb.njit(fastmath=True)
def mm2(a, b, c):
    c_0_0 = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0]
    c_0_1 = a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1]
    c_1_0 = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0]
    c_1_1 = a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1]
    c[0, 0] = c_0_0
    c[0, 1] = c_0_1
    c[1, 0] = c_1_0
    c[1, 1] = c_1_1


@nb.njit(fastmath=True)
def mm3(a, b, c):
    c_0_0 = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0]
    c_0_1 = a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1] + a[0, 2] * b[2, 1]
    c_0_2 = a[0, 0] * b[0, 2] + a[0, 1] * b[1, 2] + a[0, 2] * b[2, 2]
    c_1_0 = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0] + a[1, 2] * b[2, 0]
    c_1_1 = a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1] + a[1, 2] * b[2, 1]
    c_1_2 = a[1, 0] * b[0, 2] + a[1, 1] * b[1, 2] + a[1, 2] * b[2, 2]
    c_2_0 = a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0] + a[2, 2] * b[2, 0]
    c_2_1 = a[2, 0] * b[0, 1] + a[2, 1] * b[1, 1] + a[2, 2] * b[2, 1]
    c_2_2 = a[2, 0] * b[0, 2] + a[2, 1] * b[1, 2] + a[2, 2] * b[2, 2]
    c[0, 0] = c_0_0
    c[0, 1] = c_0_1
    c[0, 2] = c_0_2
    c[1, 0] = c_1_0
    c[1, 1] = c_1_1
    c[1, 2] = c_1_2
    c[2, 0] = c_2_0
    c[2, 1] = c_2_1
    c[2, 2] = c_2_2


@nb.njit(fastmath=True)
def mm4(a, b, c):
    c_0_0 = (
        a[0, 0] * b[0, 0]
        + a[0, 1] * b[1, 0]
        + a[0, 2] * b[2, 0]
        + a[0, 3] * b[3, 0]
    )
    c_0_1 = (
        a[0, 0] * b[0, 1]
        + a[0, 1] * b[1, 1]
        + a[0, 2] * b[2, 1]
        + a[0, 3] * b[3, 1]
    )
    c_0_2 = (
        a[0, 0] * b[0, 2]
        + a[0, 1] * b[1, 2]
        + a[0, 2] * b[2, 2]
        + a[0, 3] * b[3, 2]
    )
    c_0_3 = (
        a[0, 0] * b[0, 3]
        + a[0, 1] * b[1, 3]
        + a[0, 2] * b[2, 3]
        + a[0, 3] * b[3, 3]
    )
    c_1_0 = (
        a[1, 0] * b[0, 0]
        + a[1, 1] * b[1, 0]
        + a[1, 2] * b[2, 0]
        + a[1, 3] * b[3, 0]
    )
    c_1_1 = (
        a[1, 0] * b[0, 1]
        + a[1, 1] * b[1, 1]
        + a[1, 2] * b[2, 1]
        + a[1, 3] * b[3, 1]
    )
    c_1_2 = (
        a[1, 0] * b[0, 2]
        + a[1, 1] * b[1, 2]
        + a[1, 2] * b[2, 2]
        + a[1, 3] * b[3, 2]
    )
    c_1_3 = (
        a[1, 0] * b[0, 3]
        + a[1, 1] * b[1, 3]
        + a[1, 2] * b[2, 3]
        + a[1, 3] * b[3, 3]
    )
    c_2_0 = (
        a[2, 0] * b[0, 0]
        + a[2, 1] * b[1, 0]
        + a[2, 2] * b[2, 0]
        + a[2, 3] * b[3, 0]
    )
    c_2_1 = (
        a[2, 0] * b[0, 1]
        + a[2, 1] * b[1, 1]
        + a[2, 2] * b[2, 1]
        + a[2, 3] * b[3, 1]
    )
    c_2_2 = (
        a[2, 0] * b[0, 2]
        + a[2, 1] * b[1, 2]
        + a[2, 2] * b[2, 2]
        + a[2, 3] * b[3, 2]
    )
    c_2_3 = (
        a[2, 0] * b[0, 3]
        + a[2, 1] * b[1, 3]
        + a[2, 2] * b[2, 3]
        + a[2, 3] * b[3, 3]
    )
    c_3_0 = (
        a[3, 0] * b[0, 0]
        + a[3, 1] * b[1, 0]
        + a[3, 2] * b[2, 0]
        + a[3, 3] * b[3, 0]
    )
    c_3_1 = (
        a[3, 0] * b[0, 1]
        + a[3, 1] * b[1, 1]
        + a[3, 2] * b[2, 1]
        + a[3, 3] * b[3, 1]
    )
    c_3_2 = (
        a[3, 0] * b[0, 2]
        + a[3, 1] * b[1, 2]
        + a[3, 2] * b[2, 2]
        + a[3, 3] * b[3, 2]
    )
    c_3_3 = (
        a[3, 0] * b[0, 3]
        + a[3, 1] * b[1, 3]
        + a[3, 2] * b[2, 3]
        + a[3, 3] * b[3, 3]
    )
    c[0, 0] = c_0_0
    c[0, 1] = c_0_1
    c[0, 2] = c_0_2
    c[0, 3] = c_0_3
    c[1, 0] = c_1_0
    c[1, 1] = c_1_1
    c[1, 2] = c_1_2
    c[1, 3] = c_1_3
    c[2, 0] = c_2_0
    c[2, 1] = c_2_1
    c[2, 2] = c_2_2
    c[2, 3] = c_2_3
    c[3, 0] = c_3_0
    c[3, 1] = c_3_1
    c[3, 2] = c_3_2
    c[3, 3] = c_3_3
