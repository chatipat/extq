import numba as nb
import numpy as np


@nb.njit
def swa(data, start, end, f, *args):
    """Calculate sliding windows of an associative binary operator.

    This function implements the sliding window algorithm from [1],
    which calculates the result with the minimal number of operator
    applications. The overhead of this function is high when there is
    little overlap between sliding windows and when `f` is inexpensive.

    Note that this function modifies the data argument.

    [1] Basin, D.; Klaedtke, F.; Zălinescu, E.
    "Greedily Computing Associative Aggregations on Sliding Windows."
    *Information Processing Letters* **2015**, *115* (2), 186–192.

    Parameters
    ----------
    data : (n_in, ...) ndarray
        Input time series with at least 2 dimensions. Note that this
        array is modified.
    start : (n_out,) ndarray of int
        Starting times (inclusive) of the sliding windows.
    end : (n_out,) ndarray of int
        Ending times (exclusive) of the sliding windows.
    f : callable
        Associative binary operation taking two input arguments,
        one output argument, and *args, i.e., f(in1, in2, out, *args).
        This must be Numba-compiled.
    *args
        Additional arguments to f, if needed.

    Returns
    -------
    (n_out, ...) ndarray
        Output sliding windows.

    """
    n_in = len(data)
    n_out = len(start)

    assert data.ndim >= 2  # indexing a 1D ndarray yields a scalar
    assert start.ndim == 1 and end.ndim == 1 and len(start) == len(end)
    for n in range(n_out):
        assert 0 <= start[n] < end[n] <= n_in
    for n in range(n_out - 1):
        assert start[n] <= start[n + 1]
        assert end[n] <= end[n + 1]

    indices = np.arange(1, n_in + 1)  # temporary array
    result = np.empty((n_out,) + data.shape[1:], dtype=data.dtype)
    for n in range(n_out):
        last = _swa_pass1(indices, start[n], end[n])
        result[n] = _swa_pass2(data, indices, start[n], end[n], last, f, *args)
    return result


@nb.njit
def _swa_pass1(indices, start, end):
    # determine calculation sequence
    p = start
    last = -1
    while p < end:
        i = p
        p = indices[i]
        indices[i] = last
        last = i
    assert p == end and last != -1
    return last


@nb.njit
def _swa_pass2(data, indices, start, end, last, f, *args):
    # perform calculation
    p = indices[last]
    indices[last] = end
    while p != -1:
        i = p
        p = indices[i]
        indices[i] = end
        f(data[i], data[last], data[i], *args)
        last = i
    assert last == start
    return data[start]
