import numba as nb
import numpy as np
from extq.stop import (
    backward_stop,
    backward_stop_numba,
    forward_stop,
    forward_stop_numba,
)


def test_forward_stop():
    # test against known answer
    for d, ref in forward_d_ref_list():
        assert np.array_equal(forward_stop_ref(d), ref)
        assert np.array_equal(forward_stop(d), ref)
        assert np.array_equal(forward_stop_numba(d), ref)

    # test against reference implementation
    for d in d_list():
        ref = forward_stop_ref(d)
        assert np.array_equal(forward_stop(d), ref)
        assert np.array_equal(forward_stop_numba(d), ref)


def test_backward_stop():
    # test against known answer
    for d, ref in backward_d_ref_list():
        assert np.array_equal(backward_stop_ref(d), ref)
        assert np.array_equal(backward_stop(d), ref)
        assert np.array_equal(backward_stop_numba(d), ref)

    # test against reference implementation
    for d in d_list():
        ref = backward_stop_ref(d)
        assert np.array_equal(backward_stop(d), ref)
        assert np.array_equal(backward_stop_numba(d), ref)


@nb.njit
def forward_stop_ref(d):
    """Reference implementation of forward_stop."""
    (n,) = d.shape
    out = np.zeros(n)
    for t in range(n):
        for s in range(t, n):
            if not d[s]:
                out[t] = s
                break
        else:
            out[t] = n
    return out


@nb.njit
def backward_stop_ref(d):
    """Reference implementation of backward_stop."""
    (n,) = d.shape
    out = np.zeros(n)
    for t in range(n):
        for s in range(t, -1, -1):
            if not d[s]:
                out[t] = s
                break
        else:
            out[t] = -1
    return out


def forward_d_ref_list():
    for n in range(1, 100 + 1):
        d = np.full(n, True)
        ref = np.full(n, n)
        yield d, ref

    for n in range(1, 100 + 1):
        d = np.full(n, False)
        ref = np.arange(n)
        yield d, ref


def backward_d_ref_list():
    for n in range(1, 100 + 1):
        d = np.full(n, True)
        ref = np.full(n, -1)
        yield d, ref

    for n in range(1, 100 + 1):
        d = np.full(n, False)
        ref = np.arange(n)
        yield d, ref


def d_list():
    for n in range(1, 10 + 1):
        yield from np.indices([2] * n).reshape(n, -1).T.astype(bool)

    rng = np.random.default_rng()
    for _ in range(1000):
        n = rng.integers(10 + 1, 1000 + 1)
        yield rng.integers(2, size=n, dtype=bool)
