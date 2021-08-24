import numba as nb
import numpy as np


@nb.njit
def moving_semigroup(a, k, f, *args):
    """Calculate a moving window of an associative binary operation.

    Note that this function modifies the input array in-place.

    Parameters
    ----------
    a : (m, ...) ndarray
        Input time series of square matrices. This array is also used
        for the output, and must be C-contiguous.
    k : int
        Size of the moving window.
    f : callable
        Associative binary operation taking two input arguments and
        one output argument. This must be Numba-compiled.
    *args
        Additional arguments to f, if needed.

    Returns
    -------
    (m - k + 1, ...) ndarray
        Output time series. Each output point is the result of
        k sequential input points reduced using the operation.

    """
    assert k >= 1
    assert a.ndim >= 2  # indexing a 1D ndarray yields a scalar

    if k == 1:
        return a

    out_len = a.shape[0] - k + 1

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
