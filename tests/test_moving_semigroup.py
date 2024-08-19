import numba as nb
import numpy as np
from extq.moving_semigroup import moving_semigroup, moving_semigroup_numba


def test_moving_semigroup():
    for n, k in lengths_and_windows():
        a = np.stack([np.arange(n), np.arange(n) + 1], axis=-1)
        out = moving_semigroup(a, k, combine)
        ref = np.stack(
            [np.arange(n - k + 1), np.arange(n - k + 1) + k], axis=-1
        )
        assert np.array_equal(out, ref)


def test_moving_semigroup_numba():
    for n, k in lengths_and_windows():
        a = np.stack([np.arange(n), np.arange(n) + 1], axis=-1)
        out = moving_semigroup_numba(a, k, combine_numba)
        ref = np.stack(
            [np.arange(n - k + 1), np.arange(n - k + 1) + k], axis=-1
        )
        assert np.array_equal(out, ref)


def combine(a, b, out):
    assert a.shape == b.shape == out.shape
    assert out.shape[-1] == 2

    # check for aliasing
    out[:] = -1
    assert not np.any(a == -1)  # `out` and `a` don't alias
    assert not np.any(b == -1)  # `out` and `b` don't alias

    assert np.all(a[..., 1] == b[..., 0])

    out[..., 0] = a[..., 0]
    out[..., 1] = b[..., 1]


@nb.njit
def combine_numba(a, b, out):
    if not a.shape == b.shape == out.shape == (2,):
        raise AssertionError

    # check for aliasing
    out[0] = out[1] = -1
    if a[0] == -1 or a[1] == -1 or b[0] == -1 or b[1] == -1:
        raise AssertionError

    if not a[1] == b[0]:
        raise AssertionError

    out[0] = a[0]
    out[1] = b[1]


def lengths_and_windows():
    out = []

    for n in range(100 + 1):
        for k in range(1, n + 1):
            out.append((n, k))

    rng = np.random.default_rng(42)
    for _ in range(1000):
        n = rng.integers(100 + 1, 1000 + 1)
        k = rng.integers(1, n + 1)
        out.append((n, k))

    return out
