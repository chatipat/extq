import numba as nb
import numpy as np

from .moving_semigroup import moving_semigroup


def moving_matmul(m, lag):
    assert lag > 0
    assert m.ndim == 2
    assert m.shape[0] == m.shape[1]
    ni = m.shape[0]

    # use optimized matmul for small matrices
    if ni == 1:
        mm = _mm1
    elif ni == 2:
        mm = _mm2
    elif ni == 3:
        mm = _mm3
    elif ni == 4:
        mm = _mm4
    else:
        mm = _mm

    # find number of transitions `n`
    # and assert all entries have the same `n`
    n = None
    for i in range(ni):
        for j in range(ni):
            if m[i, j] is not None:
                if n is None:
                    n = len(m[i, j])
                assert len(m[i, j]) == n
    assert n is not None

    # convert blocks to dense array and apply moving matmul
    a = np.zeros((n, ni, ni))
    for i in range(ni):
        for j in range(ni):
            if m[i, j] is not None:
                a[:, i, j] = m[i, j]
    a = moving_semigroup(a, lag, mm)
    assert a.shape == (n + 1 - lag, ni, ni)

    # convert result to blocks with same sparsity pattern
    mlag = np.full((ni, ni), None)
    for i in range(ni):
        for j in range(ni):
            if m[i, j] is None:
                assert np.all(a[:, i, j] == 0.0)
            else:
                mlag[i, j] = a[:, i, j]
    return mlag


@nb.njit(fastmath=True)
def _mm(a, b, c):
    np.dot(a, b, c)


@nb.njit(fastmath=True)
def _mm1(a, b, c):
    c_0_0 = a[0, 0] * b[0, 0]
    c[0, 0] = c_0_0


@nb.njit(fastmath=True)
def _mm2(a, b, c):
    c_0_0 = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0]
    c_0_1 = a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1]
    c_1_0 = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0]
    c_1_1 = a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1]
    c[0, 0] = c_0_0
    c[0, 1] = c_0_1
    c[1, 0] = c_1_0
    c[1, 1] = c_1_1


@nb.njit(fastmath=True)
def _mm3(a, b, c):
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
def _mm4(a, b, c):
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
