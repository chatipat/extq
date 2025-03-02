import numpy as np

from ._soft_utils import (
    soft_backward_committor_kernel,
    soft_forward_committor_kernel,
    soft_tpt_kernel,
)
from .integral import integral_coeffs, integral_windows
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
    for qp, qm, w, v, rp, rm, h in zip(
        forward_q,
        backward_q,
        weights,
        stop_rate,
        forward_boundary,
        backward_boundary,
        rxn_coord,
        strict=True,
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

        # changes in h within each frame
        obs = np.zeros((n_frames, 2, 2))
        obs[:, 0, 0] = 0  # interior fragments (h -> h)
        obs[:, 0, 1] = 1 - h  # ending fragments (h -> 1)
        obs[:, 1, 0] = h  # starting fragments (0 -> h)
        obs[:, 1, 1] = 1  # complete transition paths (0 -> 1)

        # integrate within frame t for time dt/2,
        # assuming values are constant
        km_half = soft_backward_committor_kernel(v, rm, dt / 2)
        kp_half = soft_forward_committor_kernel(v, rp, dt / 2)
        kt_half = soft_tpt_kernel(v, rp, rm, dt / 2) * obs

        # integrate jump from frame t to frame t+1 (infinitesimal time)
        ks = np.zeros((n_frames - 1, 2, 2))
        ks[:, 0, 0] = np.diff(h)

        # windows start/end at the center of each frame
        km = km_half[:-1] @ km_half[1:]
        kp = kp_half[:-1] @ kp_half[1:]
        k = (
            km_half[:-1] @ ks @ kp_half[1:]
            + km_half[:-1] @ kt_half[1:]
            + kt_half[:-1] @ kp_half[1:]
        )
        k = integral_windows(km, kp, k, 1, lag)

        # expected number of transition paths for each window
        p = (
            qm[:-lag] * k[:, 0, 0] * qp[lag:]  # interior fragments
            + qm[:-lag] * k[:, 0, 1]  # ending fragments
            + k[:, 1, 0] * qp[lag:]  # starting fragments
            + k[:, 1, 1]  # complete transition paths
        )

        out += np.sum(w[:-lag] * p) / (lag * dt)
    return out


def soft_density(
    forward_q,
    backward_q,
    weights,
    stop_rate,
    forward_boundary,
    backward_boundary,
    lag,
    *,
    dt=1.0,
    normalize=True,
):
    assert lag > 0
    assert dt > 0

    if normalize:
        weights = normalize_weights(weights)

    out = []
    for qp, qm, w, v, rp, rm in zip(
        forward_q,
        backward_q,
        weights,
        stop_rate,
        forward_boundary,
        backward_boundary,
        strict=True,
    ):
        n_frames = len(w)
        assert qp.shape == (n_frames,)
        assert qm.shape == (n_frames,)
        assert w.shape == (n_frames,)
        assert v.shape == (n_frames,)
        assert rp.shape == (n_frames,)
        assert rm.shape == (n_frames,)

        # make sure frames beyond end of trajectory aren't needed
        assert not np.any(w[-lag:])
        if n_frames <= lag:  # no windows with nonzero weight
            out.append(np.zeros(n_frames))
            continue

        # windows start/end at the center of each frame,
        # so take half of frame t and half of frame t+1

        # backward committor kernel
        km_half = soft_backward_committor_kernel(v, rm, dt / 2)
        km = km_half[:-1] @ km_half[1:]

        # forward committor kernel
        kp_half = soft_forward_committor_kernel(v, rp, dt / 2)
        kp = kp_half[:-1] @ kp_half[1:]

        # committor outer product
        q_outer = np.zeros((n_frames - lag, 2, 2))
        q_outer[:, 0, 0] = w[:-lag] * qm[:-lag] * qp[lag:]
        q_outer[:, 0, 1] = w[:-lag] * qm[:-lag]
        q_outer[:, 1, 0] = w[:-lag] * qp[lag:]
        q_outer[:, 1, 1] = w[:-lag]

        # reactive density kernel
        k_half = soft_tpt_kernel(v, rp, rm, dt / 2)

        coef = integral_coeffs(q_outer, km, kp, 1, lag)

        p = np.zeros(n_frames)
        p[:-1] += np.einsum("tik,tij,tjk->t", coef, k_half[:-1], kp_half[1:])
        p[1:] += np.einsum("tik,tij,tjk->t", coef, km_half[:-1], k_half[1:])
        p /= 2 * lag
        out.append(p)

    return out


def soft_current(
    forward_q,
    backward_q,
    weights,
    stop_rate,
    forward_boundary,
    backward_boundary,
    cv,
    lag,
    *,
    dt=1.0,
    normalize=True,
):
    assert lag > 0
    assert dt > 0

    if normalize:
        weights = normalize_weights(weights)

    out = []
    for qp, qm, w, v, rp, rm, f in zip(
        forward_q,
        backward_q,
        weights,
        stop_rate,
        forward_boundary,
        backward_boundary,
        cv,
        strict=True,
    ):
        n_frames = len(w)
        assert qp.shape == (n_frames,)
        assert qm.shape == (n_frames,)
        assert w.shape == (n_frames,)
        assert v.shape == (n_frames,)
        assert rp.shape == (n_frames,)
        assert rm.shape == (n_frames,)
        assert f.shape == (n_frames,)

        # make sure frames beyond end of trajectory aren't needed
        assert not np.any(w[-lag:])
        if n_frames <= lag:  # no windows with nonzero weight
            out.append(np.zeros(n_frames))
            continue

        # windows start/end at the center of each frame,
        # so take half of frame t and half of frame t+1

        # backward committor kernel
        km_half = soft_backward_committor_kernel(v, rm, dt / 2)
        km = km_half[:-1] @ km_half[1:]

        # forward committor kernel
        kp_half = soft_forward_committor_kernel(v, rp, dt / 2)
        kp = kp_half[:-1] @ kp_half[1:]

        # committor outer product
        q_outer = np.zeros((n_frames - lag, 2, 2))
        q_outer[:, 0, 0] = w[:-lag] * qm[:-lag] * qp[lag:]
        q_outer[:, 0, 1] = w[:-lag] * qm[:-lag]
        q_outer[:, 1, 0] = w[:-lag] * qp[lag:]
        q_outer[:, 1, 1] = w[:-lag]

        coef = integral_coeffs(q_outer, km, kp, 1, lag)
        coef = np.einsum("tij,tik,tlj->tkl", coef, km_half[:-1], kp_half[1:])
        coef = coef[:, 0, 0] * np.diff(f)

        j = np.zeros(n_frames)
        j[:-1] += coef
        j[1:] += coef
        j /= 2 * lag * dt
        out.append(j)

    return out
