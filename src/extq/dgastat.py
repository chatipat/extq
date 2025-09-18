from abc import ABC, abstractmethod

import numpy as np
import scipy as sp

from . import linalg
from ._soft_utils import soft_backward_committor_kernel, soft_forward_committor_kernel
from ._utils import sum_windows
from .moving_semigroup import moving_matmul
from .stop import backward_stop, forward_stop


class Statistic(ABC):
    @abstractmethod
    def matrices(
        self, lag: int
    ) -> tuple[np.ndarray | sp.sparse.sparray | sp.sparse.spmatrix, np.ndarray]: ...
    @abstractmethod
    def gram_matrix(self) -> np.ndarray | sp.sparse.sparray | sp.sparse.spmatrix: ...
    @abstractmethod
    def transform(self, coef: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def transform_difference(self, coef: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def propagate(self, input: np.ndarray, lag: int) -> np.ndarray: ...
    @abstractmethod
    def propagate_difference(self, input: np.ndarray, lag: int) -> np.ndarray: ...


class StationaryDistribution(Statistic):
    def __init__(self, basis, weights, test_basis=None):
        if test_basis is None:
            test_basis = basis
        self.basis = basis
        self.weights = weights
        self.test_basis = test_basis

    def matrices(self, lag):
        assert lag > 0
        basis = self.basis
        weights = self.weights
        test_basis = self.test_basis

        n_basis = None
        a = 0.0
        b = 0.0
        for x, y, w in zip(test_basis, basis, weights, strict=True):
            n_frames = x.shape[0]
            n_basis = x.shape[1] if n_basis is None else n_basis
            assert x.shape == y.shape == (n_frames, n_basis)
            assert w.shape == (n_frames,)

            t0 = np.flatnonzero(w)  # initial time
            if len(t0) == 0:
                continue
            t1 = t0 + lag  # final time
            assert t1[-1] < n_frames  # all times < n_frames

            dx = (x[t1] - x[t0]).T
            a += dx @ linalg.scale_rows(w[t0], y[t0])
            b += dx @ w[t0]
        assert isinstance(a, (np.ndarray, sp.sparse.sparray, sp.sparse.spmatrix))
        assert isinstance(b, np.ndarray)
        return a, b

    def gram_matrix(self):
        return gram_matrix(self.basis, self.weights, test_basis=self.test_basis)

    def transform(self, coef):
        out = [
            w * (y @ coef + 1.0) for y, w in zip(self.basis, self.weights, strict=True)
        ]
        return _objarray(out, 1)

    def transform_difference(self, coef):
        out = [w * (y @ coef) for y, w in zip(self.basis, self.weights, strict=True)]
        return _objarray(out, 1)

    def propagate(self, input, lag):
        assert lag >= 0
        out = []
        for w in input:
            (n,) = w.shape
            ix = np.flatnonzero(w)
            iy = ix + lag
            assert iy[-1] < lag
            wt = np.zeros(n)
            wt[iy] = w[ix]
            out.append(wt)
        return _objarray(out, 1)

    def propagate_difference(self, input, lag):
        return self.propagate(input, lag)


class ForwardFeynmanKac(Statistic):
    def __init__(self, basis, weights, in_domain, function, guess, test_basis=None):
        if test_basis is None:
            test_basis = basis
        function = _broadcast_integrand(function, guess)
        self.basis = basis
        self.weights = weights
        self.in_domain = in_domain
        self.function = function
        self.guess = guess
        self.test_basis = test_basis

    def matrices(self, lag):
        assert lag > 0
        basis = self.basis
        weights = self.weights
        in_domain = self.in_domain
        function = self.function
        guess = self.guess
        test_basis = self.test_basis

        n_basis = None
        a = 0.0
        b = 0.0
        for x, y, w, d, f, g in zip(
            test_basis, basis, weights, in_domain, function, guess, strict=True
        ):
            n_frames = x.shape[0]
            n_basis = x.shape[1] if n_basis is None else n_basis
            assert x.shape == y.shape == (n_frames, n_basis)
            assert w.shape == d.shape == g.shape == (n_frames,)
            assert f.shape == (n_frames - 1,)

            t0 = np.flatnonzero(w)  # initial time
            if len(t0) == 0:
                continue
            t1 = np.minimum(t0 + lag, forward_stop(d)[t0])  # final time
            assert t1[-1] < n_frames  # all times < n_frames

            xw = linalg.scale_rows(w[t0], x[t0]).T
            a += xw @ (y[t1] - y[t0])
            b += xw @ (g[t1] - g[t0] + sum_windows(f, t0, t1))
        assert isinstance(a, (np.ndarray, sp.sparse.sparray, sp.sparse.spmatrix))
        assert isinstance(b, np.ndarray)
        return a, b

    def gram_matrix(self):
        return gram_matrix(self.basis, self.weights, test_basis=self.test_basis)

    def transform(self, coef):
        out = [y @ coef + g for y, g in zip(self.basis, self.guess, strict=True)]
        return _objarray(out, 1)

    def transform_difference(self, coef):
        out = [y @ coef for y in self.basis]
        return _objarray(out, 1)

    def propagate(self, input, lag):
        assert lag >= 0
        out = []
        for u, d, f in zip(input, self.in_domain, self.function, strict=True):
            n = len(u)
            assert u.shape == d.shape == (n,)
            assert f.shape == (n - 1,)
            ix = np.arange(n)
            iy = np.minimum(ix + lag, forward_stop(d))
            mask = iy < n
            ix = ix[mask]
            iy = iy[mask]
            # frames beyond endpoints of trajectory are nan
            ut = np.full(n, np.nan)
            ut[ix] = u[iy] + sum_windows(f, ix, iy)
            out.append(ut)
        return _objarray(out, 1)

    def propagate_difference(self, input, lag):
        assert lag >= 0
        out = []
        for u, d in zip(input, self.in_domain, strict=True):
            n = len(u)
            assert u.shape == d.shape == (n,)
            ix = np.arange(n)
            iy = np.minimum(ix + lag, forward_stop(d))
            mask = iy < n
            ix = ix[mask]
            iy = iy[mask]
            # frames beyond endpoints of trajectory are nan
            ut = np.full(n, np.nan)
            ut[ix] = u[iy]
            out.append(ut)
        return _objarray(out, 1)


class BackwardFeynmanKac(Statistic):
    def __init__(self, basis, weights, in_domain, function, guess, test_basis=None):
        if test_basis is None:
            test_basis = basis
        function = _broadcast_integrand(function, guess)
        self.basis = basis
        self.weights = weights
        self.in_domain = in_domain
        self.function = function
        self.guess = guess
        self.test_basis = test_basis

    def matrices(self, lag):
        assert lag > 0
        basis = self.basis
        weights = self.weights
        in_domain = self.in_domain
        function = self.function
        guess = self.guess
        test_basis = self.test_basis

        n_basis = None
        a = 0.0
        b = 0.0
        for x, y, w, d, f, g in zip(
            test_basis, basis, weights, in_domain, function, guess, strict=True
        ):
            n_frames = x.shape[0]
            n_basis = x.shape[1] if n_basis is None else n_basis
            assert x.shape == y.shape == (n_frames, n_basis)
            assert w.shape == d.shape == g.shape == (n_frames,)
            assert f.shape == (n_frames - 1,)

            t0 = np.flatnonzero(w)  # initial time
            if len(t0) == 0:
                continue
            t1 = np.maximum(t0 - lag, backward_stop(d)[t0])  # final time
            assert t1[0] >= 0  # all times >= 0

            xw = linalg.scale_rows(w[t0], x[t0]).T
            a += xw @ (y[t1] - y[t0])
            b += xw @ (g[t1] - g[t0] + sum_windows(f, t1, t0))
        assert isinstance(a, (np.ndarray, sp.sparse.sparray, sp.sparse.spmatrix))
        assert isinstance(b, np.ndarray)
        return a, b

    def gram_matrix(self):
        return gram_matrix(self.basis, self.weights, test_basis=self.test_basis)

    def transform(self, coef):
        out = [y @ coef + g for y, g in zip(self.basis, self.guess, strict=True)]
        return _objarray(out, 1)

    def transform_difference(self, coef):
        out = [y @ coef for y in self.basis]
        return _objarray(out, 1)

    def propagate(self, input, lag):
        assert lag >= 0
        out = []
        for u, d, f in zip(input, self.in_domain, self.function, strict=True):
            n = len(u)
            assert u.shape == d.shape == (n,)
            assert f.shape == (n - 1,)
            ix = np.arange(n)
            iy = np.maximum(ix - lag, backward_stop(d))
            mask = iy >= 0
            ix = ix[mask]
            iy = iy[mask]
            # frames beyond endpoints of trajectory are nan
            ut = np.full(n, np.nan)
            ut[ix] = u[iy] + sum_windows(f, iy, ix)
            out.append(ut)
        return _objarray(out, 1)

    def propagate_difference(self, input, lag):
        assert lag >= 0
        out = []
        for u, d in zip(input, self.in_domain, strict=True):
            n = len(u)
            assert u.shape == d.shape == (n,)
            ix = np.arange(n)
            iy = np.maximum(ix - lag, backward_stop(d))
            mask = iy >= 0
            ix = ix[mask]
            iy = iy[mask]
            # frames beyond endpoints of trajectory are nan
            ut = np.full(n, np.nan)
            ut[ix] = u[iy]
            out.append(ut)
        return _objarray(out, 1)


class SoftForwardCommittor(Statistic):
    def __init__(
        self, basis, weights, stop_rate, boundary, guess, *, dt=1.0, test_basis=None
    ):
        if test_basis is None:
            test_basis = basis
        self.basis = basis
        self.weights = weights
        self.stop_rate = stop_rate
        self.boundary = boundary
        self.guess = guess
        self.dt = dt
        self.test_basis = test_basis

        # continuous time equation is evaluated with discrete sampling
        # using nearest neighbor interpolation

        # currently, windows start/end at the center of the time steps
        # a better choice is to integrate the start/end time over the
        # time steps, but this is much more complicated to implement

        kernel = []
        for v, r in zip(stop_rate, boundary, strict=True):
            k_half = soft_forward_committor_kernel(v, r, dt / 2)
            k = k_half[:-1] @ k_half[1:]
            kernel.append(k)
        self._kernel = kernel

    def matrices(self, lag):
        assert lag > 0
        basis = self.basis
        weights = self.weights
        guess = self.guess
        test_basis = self.test_basis
        kernel = self._kernel

        n_basis = None
        a = 0.0
        b = 0.0
        for x, y, w, k, g in zip(
            test_basis, basis, weights, kernel, guess, strict=True
        ):
            n_frames = x.shape[0]
            n_basis = x.shape[1] if n_basis is None else n_basis
            assert x.shape == y.shape == (n_frames, n_basis)
            assert w.shape == g.shape == (n_frames,)
            assert k.shape == (n_frames - 1, 2, 2)

            t0 = np.flatnonzero(w)  # initial time
            if len(t0) == 0:
                continue
            t1 = t0 + lag  # final time
            assert t1[-1] < n_frames  # all times < n_frames

            k = moving_matmul(k.copy(), lag)[t0]
            wx = linalg.scale_rows(w[t0], x[t0])
            a += wx.T @ (linalg.scale_rows(k[:, 0, 0], y[t1]) - y[t0])
            b += wx.T @ (k[:, 0, 0] * g[t1] - g[t0] + k[:, 0, 1])
        assert isinstance(a, (np.ndarray, sp.sparse.sparray, sp.sparse.spmatrix))
        assert isinstance(b, np.ndarray)
        return a, b

    def gram_matrix(self):
        return gram_matrix(self.basis, self.weights, test_basis=self.test_basis)

    def transform(self, coef):
        out = [y @ coef + g for y, g in zip(self.basis, self.guess, strict=True)]
        return _objarray(out, 1)

    def transform_difference(self, coef):
        out = [y @ coef for y in self.basis]
        return _objarray(out, 1)

    def propagate(self, input, lag):
        assert lag >= 0
        out = []
        for u, k in zip(input, self._kernel, strict=True):
            n = len(u)
            assert u.shape == (n,)
            assert k.shape == (n - 1, 2, 2)
            k = moving_matmul(k.copy(), lag)
            t0 = np.arange(n - lag)
            t1 = t0 + lag
            # frames beyond endpoints of trajectory are nan
            ut = np.full(n, np.nan)
            ut[t0] = k[:, 0, 0] * u[t1] + k[:, 0, 1]
            out.append(ut)
        return _objarray(out, 1)

    def propagate_difference(self, input, lag):
        assert lag >= 0
        out = []
        for u, k in zip(input, self._kernel, strict=True):
            n = len(u)
            assert u.shape == (n,)
            assert k.shape == (n - 1, 2, 2)
            k = moving_matmul(k.copy(), lag)
            t0 = np.arange(n - lag)
            t1 = t0 + lag
            # frames beyond endpoints of trajectory are nan
            ut = np.full(n, np.nan)
            ut[t0] = k[:, 0, 0] * u[t1]
            out.append(ut)
        return _objarray(out, 1)


class SoftBackwardCommittor(Statistic):
    def __init__(
        self, basis, weights, stop_rate, boundary, guess, *, dt=1.0, test_basis=None
    ):
        if test_basis is None:
            test_basis = basis
        self.basis = basis
        self.weights = weights
        self.stop_rate = stop_rate
        self.boundary = boundary
        self.guess = guess
        self.dt = dt
        self.test_basis = test_basis

        # continuous time equation is evaluated with discrete sampling
        # using nearest neighbor interpolation

        # currently, windows start/end at the center of the time steps
        # a better choice is to integrate the start/end time over the
        # time steps, but this is much more complicated to implement

        kernel = []
        for v, r in zip(stop_rate, boundary, strict=True):
            k_half = soft_backward_committor_kernel(v, r, dt / 2)
            k = k_half[:-1] @ k_half[1:]
            kernel.append(k)
        self._kernel = kernel

    def matrices(self, lag):
        assert lag > 0
        basis = self.basis
        weights = self.weights
        guess = self.guess
        test_basis = self.test_basis
        kernel = self._kernel

        n_basis = None
        a = 0.0
        b = 0.0
        for x, y, w, k, g in zip(
            test_basis, basis, weights, kernel, guess, strict=True
        ):
            n_frames = x.shape[0]
            n_basis = x.shape[1] if n_basis is None else n_basis
            assert x.shape == y.shape == (n_frames, n_basis)
            assert w.shape == g.shape == (n_frames,)
            assert k.shape == (n_frames - 1, 2, 2)

            t0 = np.flatnonzero(w)  # initial time
            if len(t0) == 0:
                continue
            t1 = t0 - lag  # final time
            assert t1[0] >= 0  # all times >= 0

            k = moving_matmul(k.copy(), lag)[t1]
            wx = linalg.scale_rows(w[t0], x[t0])
            a += wx.T @ (linalg.scale_rows(k[:, 0, 0], y[t1]) - y[t0])
            b += wx.T @ (k[:, 0, 0] * g[t1] - g[t0] + k[:, 1, 0])
        assert isinstance(a, (np.ndarray, sp.sparse.sparray, sp.sparse.spmatrix))
        assert isinstance(b, np.ndarray)
        return a, b

    def gram_matrix(self):
        return gram_matrix(self.basis, self.weights, test_basis=self.test_basis)

    def transform(self, coef):
        out = [y @ coef + g for y, g in zip(self.basis, self.guess, strict=True)]
        return _objarray(out, 1)

    def transform_difference(self, coef):
        out = [y @ coef for y in self.basis]
        return _objarray(out, 1)

    def propagate(self, input, lag):
        assert lag >= 0
        out = []
        for u, k in zip(input, self._kernel, strict=True):
            n = len(u)
            assert u.shape == (n,)
            assert k.shape == (n - 1, 2, 2)
            k = moving_matmul(k.copy(), lag)
            t0 = np.arange(lag, n)
            t1 = t0 - lag
            # frames beyond endpoints of trajectory are nan
            ut = np.full(n, np.nan)
            ut[t0] = k[:, 0, 0] * u[t1] + k[:, 1, 0]
            out.append(ut)
        return _objarray(out, 1)

    def propagate_difference(self, input, lag):
        assert lag >= 0
        out = []
        for u, k in zip(input, self._kernel, strict=True):
            n = len(u)
            assert u.shape == (n,)
            assert k.shape == (n - 1, 2, 2)
            k = moving_matmul(k.copy(), lag)
            t0 = np.arange(lag, n)
            t1 = t0 - lag
            # frames beyond endpoints of trajectory are nan
            ut = np.full(n, np.nan)
            ut[t0] = k[:, 0, 0] * u[t1]
            out.append(ut)
        return _objarray(out, 1)


def gram_matrix(basis, weights, test_basis=None):
    """
    Compute the Gram matrix between basis and test basis functions.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis functions evaluated at each frame.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    test_basis : list of (n_frames[i], n_basis) ndarray of float, optional
        Test basis functions evaluated at each frame. Must have the
        same dimensions as `basis`. If None, `basis` is used.

    Returns
    -------
    c : (n_basis, n_basis) ndarray or sparse array of float
        The Gram matrix.

    """
    if test_basis is None:
        test_basis = basis
    n_basis = None
    c = 0.0
    for x, y, w in zip(test_basis, basis, weights, strict=True):
        n_frames = x.shape[0]
        n_basis = x.shape[1] if n_basis is None else n_basis
        assert x.shape == y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)
        t = np.flatnonzero(w)
        if len(t) == 0:
            continue
        xw = linalg.scale_rows(w[t], x[t]).T
        c += xw @ y[t]
    assert isinstance(c, (np.ndarray, sp.sparse.sparray, sp.sparse.spmatrix))
    return c


def _broadcast_integrand(f, trajs):
    if not np.iterable(f):
        f = [np.broadcast_to(f, traj.shape[0] - 1) for traj in trajs]
    return f


def _objarray(a, ndim):
    shape = _shape(a, ndim)
    out = np.full(shape, None)
    for index in np.ndindex(shape):
        x = a
        for i in index:
            x = x[i]
        out[index] = x
    return out


def _shape(a, ndim):
    assert ndim >= 0
    if ndim == 0:
        return ()
    shapes = [_shape(ai, ndim - 1) for ai in a]
    assert all(shape == shapes[0] for shape in shapes)
    return (len(a), *shapes[0])
