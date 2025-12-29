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
        self.basis = _objarray(basis, 1)
        self.weights = _objarray(weights, 1)
        self.test_basis = _objarray(test_basis, 1)

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
        return self.weights * (_basis_coef(self.basis, coef) + 1.0)

    def transform_difference(self, coef):
        return self.weights * _basis_coef(self.basis, coef)

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
        self.basis = _objarray(basis, 1)
        self.weights = _objarray(weights, 1)
        self.in_domain = _objarray(in_domain, 1)
        self.function = _objarray(function, 1)
        self.guess = _objarray(guess, 1)
        self.test_basis = _objarray(test_basis, 1)

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
        return _basis_coef(self.basis, coef) + self.guess

    def transform_difference(self, coef):
        return _basis_coef(self.basis, coef)

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
        self.basis = _objarray(basis, 1)
        self.weights = _objarray(weights, 1)
        self.in_domain = _objarray(in_domain, 1)
        self.function = _objarray(function, 1)
        self.guess = _objarray(guess, 1)
        self.test_basis = _objarray(test_basis, 1)

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
        return _basis_coef(self.basis, coef) + self.guess

    def transform_difference(self, coef):
        return _basis_coef(self.basis, coef)

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
        self.basis = _objarray(basis, 1)
        self.weights = _objarray(weights, 1)
        self.stop_rate = _objarray(stop_rate, 1)
        self.boundary = _objarray(boundary, 1)
        self.guess = _objarray(guess, 1)
        self.dt = dt
        self.test_basis = _objarray(test_basis, 1)

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
        return _basis_coef(self.basis, coef) + self.guess

    def transform_difference(self, coef):
        return _basis_coef(self.basis, coef)

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
        self.basis = _objarray(basis, 1)
        self.weights = _objarray(weights, 1)
        self.stop_rate = _objarray(stop_rate, 1)
        self.boundary = _objarray(boundary, 1)
        self.guess = _objarray(guess, 1)
        self.dt = dt
        self.test_basis = _objarray(test_basis, 1)

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
        return _basis_coef(self.basis, coef) + self.guess

    def transform_difference(self, coef):
        return _basis_coef(self.basis, coef)

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


class ForwardVectorFeynmanKac(Statistic):
    def __init__(
        self, basis, weights, transitions, in_domain, function, guess, test_basis=None
    ):
        if test_basis is None:
            test_basis = basis
        function = _broadcast_integrand_vector(function, transitions)
        self.basis = _objarray(basis, 2)
        self.weights = _objarray(weights, 1)
        self.transitions = _objarray(transitions, 3)
        self.in_domain = _objarray(in_domain, 2)
        self.function = _objarray(function, 3)
        self.guess = _objarray(guess, 2)
        self.test_basis = _objarray(test_basis, 2)
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        transitions = self.transitions
        in_domain = self.in_domain
        function = self.function
        guess = self.guess

        n_indices, n_trajs = in_domain.shape
        assert in_domain.shape == guess.shape == (n_indices, n_trajs)
        assert transitions.shape == function.shape == (n_indices, n_indices, n_trajs)

        kernel = []
        for k, d, f, g in zip(
            _adapt_steps(transitions),
            _adapt_frames(in_domain),
            _adapt_steps(function),
            _adapt_frames(guess),
            strict=True,
        ):
            n_frames = d.shape[1]
            assert d.shape == g.shape == (n_indices, n_frames)
            assert k.shape == f.shape == (n_indices, n_indices, n_frames - 1)

            m = np.zeros((n_indices + 1, n_indices + 1, n_frames - 1))
            m[:-1, :-1] = np.where(d[:, None, :-1], k, 0)
            m[:-1, -1] = np.where(d[:, :-1], np.sum(k * f, axis=1), g[:, :-1])
            m[-1, -1] = 1
            kernel.append(m)
        return np.moveaxis(_objarray(kernel, 3), 0, -1)

    def matrices(self, lag):
        assert lag > 0
        basis = self.basis
        weights = self.weights
        guess = self.guess
        test_basis = self.test_basis
        kernel = self._kernel

        n_indices = None
        n_basis = None
        a = 0.0
        b = 0.0
        for x, y, w, g, m in zip(
            _adapt_basis(test_basis),
            _adapt_basis(basis),
            weights,
            _adapt_frames(guess),
            _adapt_steps(kernel),
            strict=True,
        ):
            n_frames = x[0].shape[0]
            n_indices = len(x) if n_indices is None else n_indices
            n_basis = x[0].shape[1] if n_basis is None else n_basis

            assert len(x) == len(y) == n_indices
            assert all(xi.shape == (n_frames, n_basis) for xi in x)
            assert all(yi.shape == (n_frames, n_basis) for yi in y)
            assert w.shape == (n_frames,)
            assert g.shape == (n_indices, n_frames)
            assert m.shape == (n_indices + 1, n_indices + 1, n_frames - 1)

            t0 = np.flatnonzero(w)
            if len(t0) == 0:
                continue
            t1 = t0 + lag
            assert t1[-1] < n_frames  # all frames < n_frames

            m = _kernel_lag(m, lag)[t0]

            for i in range(n_indices):
                wx = linalg.scale_rows(w[t0], x[i][t0])

                yi = 0.0
                gi = 0.0

                for j in range(n_indices):
                    yi += linalg.scale_rows(m[i, j], y[j][t1])  # type: ignore
                    gi += linalg.scale_rows(m[i, j], g[j][t1])  # type: ignore
                gi += m[i, -1]  # integral and boundary conditions

                yi -= y[i][t0]
                gi -= g[i][t0]

                a += wx.T @ yi
                b += wx.T @ gi
        assert isinstance(a, (np.ndarray, sp.sparse.sparray, sp.sparse.spmatrix))
        assert isinstance(b, np.ndarray)
        return a, b

    def gram_matrix(self):
        basis = self.basis
        weights = self.weights
        test_basis = self.test_basis
        n_indices = None
        n_basis = None
        c = 0.0
        for x, y, w in zip(
            _adapt_basis(test_basis), _adapt_basis(basis), weights, strict=True
        ):
            n_frames = x[0].shape[0]
            n_indices = len(x) if n_indices is None else n_indices
            n_basis = x[0].shape[1] if n_basis is None else n_basis

            assert len(x) == len(y) == n_indices
            assert all(xi.shape == (n_frames, n_basis) for xi in x)
            assert all(yi.shape == (n_frames, n_basis) for yi in y)
            assert w.shape == (n_frames,)

            t = np.flatnonzero(w)
            if len(t) == 0:
                continue
            for i in range(n_indices):
                wx = linalg.scale_rows(w[t], x[i][t])
                c += wx.T @ y[i][t]
        assert isinstance(c, (np.ndarray, sp.sparse.sparray, sp.sparse.spmatrix))
        return c

    def transform(self, coef):
        return _basis_coef(self.basis, coef) + self.guess

    def transform_difference(self, coef):
        return _basis_coef(self.basis, coef)

    def propagate(self, input, lag):
        assert lag >= 0
        kernel = self._kernel
        out = []
        for u, m in zip(_adapt_frames(input), _adapt_steps(kernel), strict=True):
            n_indices, n_frames = u.shape
            assert m.shape == (n_indices + 1, n_indices + 1, n_frames - 1)
            m = _kernel_lag(m, lag)
            t0 = np.arange(n_frames - lag)
            t1 = t0 + lag
            # frames beyond endpoints of trajectory are nan
            ut = np.full((n_indices, n_frames), np.nan)
            ut[t0] = np.einsum("ijt,jt->it", m[:-1, :-1], u[:, t1]) + m[:-1, -1]
            out.append(ut)
        return _objarray(out, 2).T

    def propagate_difference(self, input, lag):
        assert lag >= 0
        kernel = self._kernel
        out = []
        for u, m in zip(_adapt_frames(input), _adapt_steps(kernel), strict=True):
            n_indices, n_frames = u.shape
            assert m.shape == (n_indices + 1, n_indices + 1, n_frames - 1)
            m = _kernel_lag(m, lag)
            t0 = np.arange(n_frames - lag)
            t1 = t0 + lag
            # frames beyond endpoints of trajectory are nan
            ut = np.full((n_indices, n_frames), np.nan)
            ut[t0] = np.einsum("ijt,jt->it", m[:-1, :-1], u[:, t1])
            out.append(ut)
        return _objarray(out, 2).T


class BackwardVectorFeynmanKac(Statistic):
    def __init__(
        self, basis, weights, transitions, in_domain, function, guess, test_basis=None
    ):
        if test_basis is None:
            test_basis = basis
        function = _broadcast_integrand_vector(function, transitions)
        self.basis = _objarray(basis, 2)
        self.weights = _objarray(weights, 1)
        self.transitions = _objarray(transitions, 3)
        self.in_domain = _objarray(in_domain, 2)
        self.function = _objarray(function, 3)
        self.guess = _objarray(guess, 2)
        self.test_basis = _objarray(test_basis, 2)
        self._kernel = self._make_kernel()

    def _make_kernel(self):
        transitions = self.transitions
        in_domain = self.in_domain
        function = self.function
        guess = self.guess

        n_indices, n_trajs = in_domain.shape
        assert in_domain.shape == guess.shape == (n_indices, n_trajs)
        assert transitions.shape == function.shape == (n_indices, n_indices, n_trajs)

        kernel = []
        for k, d, f, g in zip(
            _adapt_steps(transitions),
            _adapt_frames(in_domain),
            _adapt_steps(function),
            _adapt_frames(guess),
            strict=True,
        ):
            n_frames = d.shape[1]
            assert d.shape == g.shape == (n_indices, n_frames)
            assert k.shape == f.shape == (n_indices, n_indices, n_frames - 1)

            m = np.zeros((n_indices + 1, n_indices + 1, n_frames - 1))
            m[:-1, :-1] = np.where(d[None, :, 1:], k, 0)
            m[-1, :-1] = np.where(d[:, 1:], np.sum(k * f, axis=0), g[:, 1:])
            m[-1, -1] = 1
            kernel.append(m)
        return np.moveaxis(_objarray(kernel, 3), 0, -1)

    def matrices(self, lag):
        assert lag > 0
        basis = self.basis
        weights = self.weights
        guess = self.guess
        test_basis = self.test_basis
        kernel = self._kernel

        n_indices = None
        n_basis = None
        a = 0.0
        b = 0.0
        for x, y, w, g, m in zip(
            _adapt_basis(test_basis),
            _adapt_basis(basis),
            weights,
            _adapt_frames(guess),
            _adapt_steps(kernel),
            strict=True,
        ):
            n_frames = x[0].shape[0]
            n_indices = len(x) if n_indices is None else n_indices
            n_basis = x[0].shape[1] if n_basis is None else n_basis

            assert len(x) == len(y) == n_indices
            assert all(xi.shape == (n_frames, n_basis) for xi in x)
            assert all(yi.shape == (n_frames, n_basis) for yi in y)
            assert w.shape == (n_frames,)
            assert g.shape == (n_indices, n_frames)
            assert m.shape == (n_indices + 1, n_indices + 1, n_frames - 1)

            t0 = np.flatnonzero(w)
            if len(t0) == 0:
                continue
            t1 = t0 - lag
            assert t1[0] >= 0  # all frames >= 0

            m = _kernel_lag(m, lag)[t1]

            for i in range(n_indices):
                wx = linalg.scale_rows(w[t0], x[i][t0])

                yi = 0.0
                gi = 0.0

                for j in range(n_indices):
                    yi += linalg.scale_rows(m[j, i], y[j][t1])  # type: ignore
                    gi += linalg.scale_rows(m[j, i], g[j][t1])  # type: ignore
                gi += m[-1, i]  # integral and boundary conditions

                yi -= y[i][t0]
                gi -= g[i][t0]

                a += wx.T @ yi
                b += wx.T @ gi
        assert isinstance(a, (np.ndarray, sp.sparse.sparray, sp.sparse.spmatrix))
        assert isinstance(b, np.ndarray)
        return a, b

    def gram_matrix(self):
        basis = self.basis
        weights = self.weights
        test_basis = self.test_basis
        n_indices = None
        n_basis = None
        c = 0.0
        for x, y, w in zip(
            _adapt_basis(test_basis), _adapt_basis(basis), weights, strict=True
        ):
            n_frames = x[0].shape[0]
            n_indices = len(x) if n_indices is None else n_indices
            n_basis = x[0].shape[1] if n_basis is None else n_basis

            assert len(x) == len(y) == n_indices
            assert all(xi.shape == (n_frames, n_basis) for xi in x)
            assert all(yi.shape == (n_frames, n_basis) for yi in y)
            assert w.shape == (n_frames,)

            t = np.flatnonzero(w)
            if len(t) == 0:
                continue
            for i in range(n_indices):
                wx = linalg.scale_rows(w[t], x[i][t])
                c += wx.T @ y[i][t]
        assert isinstance(c, (np.ndarray, sp.sparse.sparray, sp.sparse.spmatrix))
        return c

    def transform(self, coef):
        return _basis_coef(self.basis, coef) + self.guess

    def transform_difference(self, coef):
        return _basis_coef(self.basis, coef)

    def propagate(self, input, lag):
        assert lag >= 0
        kernel = self._kernel
        out = []
        for u, m in zip(_adapt_frames(input), _adapt_steps(kernel), strict=True):
            n_indices, n_frames = u.shape
            assert m.shape == (n_indices + 1, n_indices + 1, n_frames - 1)
            m = _kernel_lag(m, lag)
            t0 = np.arange(lag, n_frames)
            t1 = t0 - lag
            # frames beyond endpoints of trajectory are nan
            ut = np.full((n_indices, n_frames), np.nan)
            ut[t0] = np.einsum("jit,jt->it", m[:-1, :-1], u[:, t1]) + m[-1, :-1]
            out.append(ut)
        return _objarray(out, 2).T

    def propagate_difference(self, input, lag):
        assert lag >= 0
        kernel = self._kernel
        out = []
        for u, m in zip(_adapt_frames(input), _adapt_steps(kernel), strict=True):
            n_indices, n_frames = u.shape
            assert m.shape == (n_indices + 1, n_indices + 1, n_frames - 1)
            m = _kernel_lag(m, lag)
            t0 = np.arange(lag, n_frames)
            t1 = t0 - lag
            # frames beyond endpoints of trajectory are nan
            ut = np.full((n_indices, n_frames), np.nan)
            ut[t0] = np.einsum("jit,jt->it", m[:-1, :-1], u[:, t1])
            out.append(ut)
        return _objarray(out, 2).T


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


def _basis_coef(basis, coef):
    return np.vectorize(lambda y: y @ coef)(basis)


def _kernel_lag(m, lag):
    return np.moveaxis(moving_matmul(np.moveaxis(m, -1, 0).copy(order="C"), lag), 0, -1)


def _adapt_basis(a):
    return np.moveaxis(_objarray(a, 2), 1, 0).tolist()


def _adapt_frames(a):
    return map(np.array, np.moveaxis(_objarray(a, 2), 1, 0).tolist())


def _adapt_steps(a):
    return map(np.array, np.moveaxis(_objarray(a, 3), 2, 0).tolist())


def _broadcast_integrand(f, trajs):
    if not np.iterable(f):
        f = [np.broadcast_to(f, traj.shape[0] - 1) for traj in trajs]
    return f


def _broadcast_integrand_vector(f, transitions):
    if not np.iterable(f):
        func = np.vectorize(lambda k: np.broadcast_to(f, k))
        transitions = _objarray(transitions, 3)
        f = func(transitions)
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
