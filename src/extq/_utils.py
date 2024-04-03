import numpy as np


def sum_windows(a, start, end):
    r"""
    Return the sum of the elements within each window.

    The output of this function is::

        out[i] = np.sum(a[start[i] : end[i]], axis=0)

    Parameters
    ----------
    a : (n, *shape) array_like
        Input array.
    start : (n_windows,) ndarray of int
        Starting index (inclusive) of each window.
    end : (n_windows,) ndarray of int
        Ending index (exclusive) of each window.

    Returns
    -------
    out : (n_windows, *shape) ndarray
        Sum of the elements within each window.

    """
    c = np.concatenate(
        [np.zeros((1, *a.shape[1:]), a.dtype), np.cumsum(a, axis=0)]
    )
    return c[end] - c[start]
