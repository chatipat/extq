import numpy as np
from more_itertools import zip_equal

from .moving_semigroup import moving_matmul
from .utils import normalize_weights


def soft_rate(
    forward_q,
    backward_q,
    weights,
    stop_rate,
    forward_boundary,
    backward_boundary,
    rxn_coord,
    lag,
    *,
    dt=1.0,
    normalize=True,
):
    assert lag > 0
    assert dt > 0

    if normalize:
        weights = normalize_weights(weights)

    out = 0.0
    for qp, qm, w, v, rp, rm, h in zip_equal(
        forward_q,
        backward_q,
        weights,
        stop_rate,
        forward_boundary,
        backward_boundary,
        rxn_coord,
    ):
        n_frames = len(w)
        assert qp.shape == (n_frames,)
        assert qm.shape == (n_frames,)
        assert w.shape == (n_frames,)
        assert v.shape == (n_frames,)
        assert rp.shape == (n_frames,)
        assert rm.shape == (n_frames,)
        assert h.shape == (n_frames,)

        # make sure frames beyond end of trajectory aren't needed
        assert not np.any(w[-lag:])
        if n_frames <= lag:  # no windows with nonzero weight
            continue

        k_half = _rate_kernel(v, rp, rm, h, dt / 2)
        k_jump = _rate_kernel_jump(h)
        # windows start/end at the center of each frame
        k = k_half[:-1] @ k_jump @ k_half[1:]
        k = moving_matmul(k)
        k = k[:, :2, 2:]

        # expected number of transition paths for each window
        p = (
            qm[:-lag] * k[:, 0, 0] * qp[lag:]  #  interior fragments
            + qm[:-lag] * k[:, 0, 1]  # ending fragments
            + k[:, 1, 0] * qp[lag:]  # starting fragments
            + k[:, 1, 1]  # complete transition paths
        )

        out += np.sum(w[:-lag] * p) / (lag * dt)
    return out


def _rate_kernel(v, rp, rm, h, dt):
    n = len(v)

    assert v.shape == (n,)
    assert rp.shape == (n,)
    assert rm.shape == (n,)
    assert h.shape == (n,)
    assert dt >= 0

    if dt == 0:
        # treat dt as a positive infinitesimal
        vdt = np.where(np.isfinite(v), 0, v)
    else:
        vdt = v * dt

    p_cont = np.exp(-vdt)
    p_stop = -np.expm1(-vdt)

    # integrate frame t for time dt, assuming values are constant
    kernel = np.zeros((n, 4, 4))

    kernel00 = kernel[:, :2, :2]
    kernel00[:, 0, 0] = p_cont
    kernel00[:, 1, 0] = p_stop * rm
    kernel00[:, 1, 1] = 1

    kernel01 = kernel[:, :2, 2:]
    kernel01[:, 1, 0] = p_stop * (h - 0) * rm
    kernel01[:, 0, 1] = p_stop * (1 - h) * rp
    kernel01[:, 1, 1] = (vdt - p_stop) * rm * rp

    kernel11 = kernel[:, 2:, 2:]
    kernel11[:, 0, 0] = p_cont
    kernel11[:, 0, 1] = p_stop * rp
    kernel11[:, 1, 1] = 1

    return kernel


def _rate_kernel_jump(h):
    # integrate jump from frame t to frame t+1 (infinitesimal time)
    kernel_jump = np.zeros((len(h) - 1, 4, 4))
    kernel_jump01 = kernel_jump[:, :2, 2:]
    kernel_jump01[:, 0, 0] = np.diff(h)
    return kernel_jump
