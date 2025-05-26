import numpy as np
import scipy as sp

from . import linalg


def evaluate(kernel, dt=1.0):
    _check_shapes(kernel)
    s02, s01, s11, s12, t01, t11, t12 = np.broadcast_arrays(*kernel)
    dtype = np.result_type(*kernel)

    n, n0, n2 = s02.shape[-3:]
    s = linalg.batched_block(
        [
            [np.identity(n0, dtype=dtype), s01, s02],
            [None, s11, s12],
            [None, None, np.identity(n2, dtype=dtype)],
        ]
    )
    t = linalg.batched_block(
        [
            [None, t01, None],
            [None, t11, t12],
            [None, None, None],
        ]
    )
    t = sp.linalg.expm(t * dt)

    out = t[..., 0, :, :]
    for i in range(n):
        out = out @ s[..., i, :, :] @ t[..., i + 1, :, :]
    return out


def add(kernel1, kernel2):
    return sum([kernel1, kernel2])


def sum(kernels):
    for k in kernels:
        _check_shapes(k)
    s02, s01, s11, s12, t01, t11, t12 = [np.broadcast_arrays(*a) for a in zip(*kernels)]

    s02 = np.sum(s02, axis=0)
    s01 = np.concatenate(s01, axis=-1)
    s11 = linalg.batched_block_diag(*s11)
    s12 = np.concatenate(s12, axis=-2)

    t01 = np.concatenate(t01, axis=-1)
    t11 = linalg.batched_block_diag(*t11)
    t12 = np.concatenate(t12, axis=-2)

    k = s02, s01, s11, s12, t01, t11, t12
    _check_shapes(k)
    return k


def multiply(kernel1, kernel2):
    _check_shapes(kernel1)
    _check_shapes(kernel2)
    s02, s01, s11, s12, t01, t11, t12 = [
        np.broadcast_arrays(*a) for a in zip(kernel1, kernel2)
    ]

    i11 = [np.identity(a.shape[-1], dtype=a.dtype) for a in t11]

    s02 = np.prod(s02, axis=0)
    s01 = linalg.batched_kron(*s01, axis=-1)
    s11 = linalg.batched_kron(*s11)
    s12 = linalg.batched_kron(*s12, axis=-2)

    t01 = linalg.batched_kron(*t01, axis=-1)
    t11 = linalg.batched_kron(t11[0], i11[1]) + linalg.batched_kron(i11[0], t11[1])
    t12 = linalg.batched_kron(*t12, axis=-2)

    k = s02, s01, s11, s12, t01, t11, t12
    _check_shapes(k)
    return k


def prod(kernels):
    out = kernels[0]
    for k in kernels[1:]:
        out = multiply(out, k)
    return out


def seq(kernel1, kernel2):
    _check_shapes(kernel1)
    _check_shapes(kernel2)
    s02, s01, s11, s12, t01, t11, t12 = kernel1
    s24, s23, s33, s34, t23, t33, t34 = kernel2

    n0, n1 = s01.shape[-2:]
    n3, n4 = s34.shape[-2:]
    dtype = np.result_type(*kernel1, *kernel2)

    s04 = np.zeros((n0, n4), dtype=dtype)
    s0x = linalg.batched_block([[s01, s02 @ t23[1:]]])
    sxx = linalg.batched_block([[s11, s12 @ t23[1:] + t12[:-1] @ s23], [None, s33]])
    sx4 = linalg.batched_block([[t12[:-1] @ s24], [s34]])

    t0x = linalg.batched_block([[t01, np.zeros((n0, n3), dtype=dtype)]])
    txx = linalg.batched_block([[t11, t12 @ t23], [None, t33]])
    tx4 = linalg.batched_block([[np.zeros((n1, n4), dtype=dtype)], [t34]])

    kernel = s04, s0x, sxx, sx4, t0x, txx, tx4
    _check_shapes(kernel)
    return kernel


def _check_shapes(kernel):
    s02, s01, s11, s12, t01, t11, t12 = kernel
    n, n0, n1 = t01.shape[-3:]
    n2 = t12.shape[-1]

    assert s02.shape[-3:] == (n - 1, n0, n2)
    assert s01.shape[-3:] == (n - 1, n0, n1)
    assert s11.shape[-3:] == (n - 1, n1, n1)
    assert s12.shape[-3:] == (n - 1, n1, n2)
    assert t01.shape[-3:] == (n, n0, n1)
    assert t11.shape[-3:] == (n, n1, n1)
    assert t12.shape[-3:] == (n, n1, n2)
