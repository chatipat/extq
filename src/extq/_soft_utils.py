import numpy as np


def soft_forward_committor_kernel(v, r, dt):
    n = len(v)

    assert v.shape == (n,)
    assert r.shape == (n,)
    assert dt >= 0

    if dt == 0:
        # treat dt as a positive infinitesimal
        vdt = np.where(np.isfinite(v), 0, v)
    else:
        vdt = v * dt

    kernel = np.zeros((n, 2, 2))
    kernel[:, 0, 0] = np.exp(-vdt)
    kernel[:, 0, 1] = -np.expm1(-vdt) * r
    kernel[:, 1, 1] = 1

    return kernel


def soft_backward_committor_kernel(v, r, dt):
    n = len(v)

    assert v.shape == (n,)
    assert r.shape == (n,)
    assert dt >= 0

    if dt == 0:
        # treat dt as a positive infinitesimal
        vdt = np.where(np.isfinite(v), 0, v)
    else:
        vdt = v * dt

    kernel = np.zeros((n, 2, 2))
    kernel[:, 0, 0] = np.exp(-vdt)
    kernel[:, 1, 0] = -np.expm1(-vdt) * r
    kernel[:, 1, 1] = 1

    return kernel


def soft_tpt_kernel(v, rp, rm, dt):
    n = len(v)

    assert v.shape == (n,)
    assert rp.shape == (n,)
    assert rm.shape == (n,)

    if dt == 0:
        # treat dt as a positive infinitesimal
        vdt = np.where(np.isfinite(v), 0, v)
    else:
        vdt = v * dt

    p_cont = np.exp(-vdt)
    p_stop = -np.expm1(-vdt)

    kernel = np.zeros((n, 2, 2))
    kernel[:, 0, 0] = p_cont
    kernel[:, 0, 1] = p_stop * rp
    kernel[:, 1, 0] = p_stop * rm
    kernel[:, 1, 1] = (vdt - p_stop) * rm * rp

    return kernel
