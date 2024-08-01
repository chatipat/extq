import numpy as np
from more_itertools import zip_equal

from . import linalg
from .moving_semigroup import moving_matmul


def soft_forward_committor(
    basis,
    weights,
    stop_rate,
    boundary,
    guess,
    lag,
    *,
    dt=1.0,
    test_basis=None,
):
    assert lag > 0

    if test_basis is None:
        test_basis = basis

    n_basis = None

    a = 0.0
    b = 0.0

    for x, y, w, v, r, g in zip_equal(
        test_basis, basis, weights, stop_rate, boundary, guess
    ):
        n_frames = x.shape[0]
        n_basis = x.shape[1] if n_basis is None else n_basis

        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)
        assert v.shape == (n_frames,)
        assert r.shape == (n_frames,)
        assert g.shape == (n_frames,)

        iw = np.flatnonzero(w)  # start of window
        if len(iw) == 0:
            continue
        ix = iw  # initial time
        iy = ix + lag  # final time
        assert iy[-1] < n_frames  # all times < n_frames

        # continuous time equation is evaluated with discrete sampling
        # using nearest neighbor interpolation

        # currently, windows start/end at the center of the time steps
        # a better choice is to integrate the start/end time over the
        # time steps, but this is much more complicated to implement
        k_half = _forward_committor_kernel(v, r, dt / 2)
        if lag == 1:
            k = k_half[ix] @ k_half[iy]
        else:
            k = _forward_committor_kernel(v, r, dt)
            k = moving_matmul(k, lag - 1)[1:-1]
            k = k_half[ix] @ k[iw] @ k_half[iy]

        wx = linalg.scale_rows(w[iw], x[ix])
        a += wx.T @ (linalg.scale_rows(k[:, 0, 0], y[iy]) - y[ix])
        b += wx.T @ (g[iy] - g[ix] + k[:, 0, 1])
    coef = -linalg.solve(a, b)
    q = [y @ coef + g for y, g in zip_equal(basis, guess)]
    return q


def _forward_committor_kernel(v, r, dt):
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


def soft_backward_committor(
    basis,
    weights,
    stop_rate,
    boundary,
    guess,
    lag,
    *,
    dt=1.0,
    test_basis=None,
):
    assert lag > 0

    if test_basis is None:
        test_basis = basis

    n_basis = None

    a = 0.0
    b = 0.0

    for x, y, w, v, r, g in zip_equal(
        test_basis, basis, weights, stop_rate, boundary, guess
    ):
        n_frames = x.shape[0]
        n_basis = x.shape[1] if n_basis is None else n_basis

        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)
        assert v.shape == (n_frames,)
        assert r.shape == (n_frames,)
        assert g.shape == (n_frames,)

        iw = np.flatnonzero(w)  # start of window
        if len(iw) == 0:
            continue
        ix = iw + lag  # initial time
        iy = ix - lag  # final time
        assert ix[-1] < n_frames  # all times < n_frames

        # continuous time equation is evaluated with discrete sampling
        # using nearest neighbor interpolation

        # currently, windows start/end at the center of the time steps
        # a better choice is to integrate the start/end time over the
        # time steps, but this is much more complicated to implement
        k_half = _backward_committor_kernel(v, r, dt / 2)
        if lag == 1:
            k = k_half[iy] @ k_half[ix]
        else:
            k = _backward_committor_kernel(v, r, dt)
            k = moving_matmul(k, lag - 1)[1:-1]
            k = k_half[iy] @ k[iw] @ k_half[ix]

        wx = linalg.scale_rows(w[iw], x[ix])
        a += wx.T @ (linalg.scale_rows(k[:, 0, 0], y[iy]) - y[ix])
        b += wx.T @ (g[iy] - g[ix] + k[:, 1, 0])
    coef = -linalg.solve(a, b)
    q = [y @ coef + g for y, g in zip_equal(basis, guess)]
    return q


def _backward_committor_kernel(v, r, dt):
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
