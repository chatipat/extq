"""DGA with time-convolutionless memory estimators for statistics."""

import numpy as np

from . import linalg, utils
from ._utils import sum_windows
from .stop import backward_stop, forward_stop
from .utils import normalize_weights, uniform_weights


def stationary_distribution(
    basis,
    lag1,
    lag2,
    *,
    maxlag=None,
    guess=None,
    test_basis=None,
    normalize=True,
):
    if maxlag is None:
        maxlag = lag2
    if guess is None:
        guess = uniform_weights(basis, maxlag)

    # solve DGA for coef
    a, b = _stationary_distribution_matrices(
        basis, guess, lag1, lag2, test_basis=test_basis
    )
    coef = -linalg.solve(a, b)

    # compute solutions from coef
    # solution1 and solution2 have the same projection
    solution0 = _transform_distribution(coef, basis, guess)
    if normalize:
        solution0 = normalize_weights(solution0)
    solution1 = _time_lagged_stationary_distribution(solution0, lag1)
    solution2 = _time_lagged_stationary_distribution(solution0, lag2)

    # project solution onto affine model
    projcoef = _affine_projection_coef_distribution(
        solution1, basis, guess, test_basis=test_basis
    )
    projcoef_ = _affine_projection_coef_distribution(
        solution2, basis, guess, test_basis=test_basis
    )
    assert np.allclose(projcoef, projcoef_)
    projection = _transform_distribution(projcoef, basis, guess)
    if normalize:
        projection = normalize_weights(projection)

    return projection, solution1, solution2


def forecast(
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag1,
    lag2,
    *,
    test_basis=None,
):
    # solve DGA for coef
    a, b = _forecast_matrices(
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag1,
        lag2,
        test_basis=test_basis,
    )
    coef = -linalg.solve(a, b)

    # compute solutions from coef
    # solution1 and solution2 have the same projection
    solution0 = _transform(coef, basis, guess)
    solution1 = _time_lagged_forecast(solution0, in_domain, function, lag1)
    solution2 = _time_lagged_forecast(solution0, in_domain, function, lag2)

    # project solution onto affine model
    projcoef = _affine_projection_coef(
        solution1, basis, weights, guess, test_basis=test_basis
    )
    projcoef_ = _affine_projection_coef(
        solution2, basis, weights, guess, test_basis=test_basis
    )
    assert np.allclose(projcoef, projcoef_)
    projection = _transform(projcoef, basis, guess)

    return projection, solution1, solution2


def aftcast(
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag1,
    lag2,
    *,
    test_basis=None,
):
    # shift weights because time 0 is at end of window
    weights = utils.shift_weights(weights, 0, lag2)

    # solve DGA for coef
    a, b = _aftcast_matrices(
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag1,
        lag2,
        test_basis=test_basis,
    )
    coef = -linalg.solve(a, b)

    # compute solutions from coef
    # solution1 and solution2 have the same projection
    solution0 = _transform(coef, basis, guess)
    solution1 = _time_lagged_aftcast(solution0, in_domain, function, lag1)
    solution2 = _time_lagged_aftcast(solution0, in_domain, function, lag2)

    # project solution onto affine model
    projcoef = _affine_projection_coef(
        solution1, basis, weights, guess, test_basis=test_basis
    )
    projcoef_ = _affine_projection_coef(
        solution2, basis, weights, guess, test_basis=test_basis
    )
    assert np.allclose(projcoef, projcoef_)
    projection = _transform(projcoef, basis, guess)

    return projection, solution1, solution2


def _stationary_distribution_matrices(
    basis,
    guess,
    lag1,
    lag2,
    *,
    test_basis=None,
):
    assert 0 <= lag1 < lag2
    if test_basis is None:
        test_basis = basis
    n_basis = None
    a = 0.0
    b = 0.0
    for x, y, w in zip(test_basis, basis, guess, strict=True):
        n_frames = x.shape[0]
        n_basis = x.shape[1] if n_basis is None else n_basis
        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)

        iw = np.flatnonzero(w)  # start of window
        if len(iw) == 0:
            continue
        ix = iw  # initial time
        iy = ix + lag1  # final time
        assert iy[-1] < n_frames  # all times < n_frames

        wdx = linalg.scale_rows(w[iw], x[iy] - x[ix])
        a += wdx.T @ y[iw]
        b += np.ravel(wdx.sum(axis=0))
    return a, b


def _forecast_matrices(
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag1,
    lag2,
    *,
    test_basis=None,
):
    if test_basis is None:
        test_basis = basis
    function = _broadcast_integrand(function, guess)

    assert 0 <= lag1 < lag2
    n_basis = basis[0].shape[-1]

    a = 0.0
    b = 0.0

    for x, y, w, d, f, g in zip(
        test_basis, basis, weights, in_domain, function, guess, strict=True
    ):
        n_frames = len(w)
        assert x.shape == y.shape == (n_frames, n_basis)
        assert w.shape == d.shape == g.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)

        t0 = np.flatnonzero(w)
        if len(t0) == 0:
            continue
        stop = forward_stop(d)[t0]
        t1 = np.minimum(t0 + lag1, stop)
        t2 = np.minimum(t0 + lag2, stop)
        assert t2[-1] < n_frames

        xw = linalg.scale_rows(w[t0], x[t0]).T
        a += xw @ (y[t2] - y[t1])
        b += xw @ (g[t2] - g[t1] + sum_windows(f, t1, t2))

    return a, b


def _aftcast_matrices(
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag1,
    lag2,
    *,
    test_basis=None,
):
    if test_basis is None:
        test_basis = basis
    function = _broadcast_integrand(function, guess)

    assert 0 <= lag1 < lag2
    n_basis = basis[0].shape[-1]

    a = 0.0
    b = 0.0

    for x, y, w, d, f, g in zip(
        test_basis, basis, weights, in_domain, function, guess, strict=True
    ):
        n_frames = len(w)
        assert x.shape == y.shape == (n_frames, n_basis)
        assert w.shape == d.shape == g.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)

        t0 = np.flatnonzero(w)
        if len(t0) == 0:
            continue
        stop = backward_stop(d)[t0]
        t1 = np.maximum(t0 - lag1, stop)
        t2 = np.maximum(t0 - lag2, stop)
        assert t2[0] >= 0

        xw = linalg.scale_rows(w[t0], x[t0]).T
        a += xw @ (y[t2] - y[t1])
        b += xw @ (g[t2] - g[t1] + sum_windows(f, t2, t1))

    return a, b


def _time_lagged_stationary_distribution(values, lag):
    assert lag >= 0
    out = []
    for u0 in values:
        (n_frames,) = u0.shape
        u1 = np.zeros(n_frames)
        t0 = np.flatnonzero(u0)
        if len(t0) > 0:
            t1 = t0 + lag
            assert t1[-1] < n_frames
            u1[t1] = u0[t0]
        out.append(u1)
    return out


def _time_lagged_forecast(values, in_domain, function, lag):
    function = _broadcast_integrand(function, values)

    assert lag >= 0

    out = []
    for u0, d, f in zip(values, in_domain, function, strict=True):
        n_frames = len(u0)
        assert u0.shape == d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)

        # frames beyond endpoints of trajectory are nan
        u1 = np.full(n_frames, np.nan)

        if n_frames > lag:
            t0 = np.arange(n_frames)
            t1 = np.minimum(t0 + lag, forward_stop(d))

            mask = t1 < n_frames
            t0 = t0[mask]
            t1 = t1[mask]

            u1[t0] = u0[t1] + sum_windows(f, t0, t1)

        out.append(u1)
    return out


def _time_lagged_aftcast(values, in_domain, function, lag):
    function = _broadcast_integrand(function, values)

    assert lag >= 0

    out = []
    for u0, d, f in zip(values, in_domain, function, strict=True):
        n_frames = len(u0)
        assert u0.shape == d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)

        # frames beyond endpoints of trajectory are nan
        u1 = np.full(n_frames, np.nan)

        if n_frames > lag:
            t0 = np.arange(n_frames)
            t1 = np.maximum(t0 - lag, backward_stop(d))

            mask = t1 >= 0
            t0 = t0[mask]
            t1 = t1[mask]

            u1[t0] = u0[t1] + sum_windows(f, t1, t0)

        out.append(u1)
    return out


def _affine_projection_coef_distribution(values, basis, origin, *, test_basis=None):
    if test_basis is None:
        test_basis = basis

    n_basis = basis[0].shape[-1]

    a = 0.0
    b = 0.0

    for x, y, w, u in zip(test_basis, basis, origin, values, strict=True):
        n_frames = len(w)
        assert x.shape == y.shape == (n_frames, n_basis)
        assert w.shape == u.shape == (n_frames,)

        a += x.T @ linalg.scale_rows(w, y)
        b += x.T @ (u - w)

    return linalg.solve(a, b)


def _affine_projection_coef(values, basis, weights, origin, *, test_basis=None):
    if test_basis is None:
        test_basis = basis

    n_basis = basis[0].shape[-1]

    a = 0.0
    b = 0.0

    for x, y, w, g, u in zip(test_basis, basis, weights, origin, values, strict=True):
        n_frames = len(w)
        assert x.shape == y.shape == (n_frames, n_basis)
        assert w.shape == g.shape == u.shape == (n_frames,)

        t = np.flatnonzero(w)
        if len(t) == 0:
            continue

        xw = linalg.scale_rows(w[t], x[t]).T
        a += xw @ y[t]
        b += xw @ (u[t] - g[t])

    return linalg.solve(a, b)


def _transform_distribution(coef, basis, guess):
    return [w * (y @ coef + 1.0) for y, w in zip(basis, guess, strict=True)]


def _transform(coef, basis, guess):
    return [y @ coef + g for y, g in zip(basis, guess, strict=True)]


def _broadcast_integrand(f, trajs):
    if not np.iterable(f):
        f = [np.broadcast_to(f, traj.shape[0] - 1) for traj in trajs]
    return f
