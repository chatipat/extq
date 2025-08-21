from abc import ABC, abstractmethod

import numpy as np
import scipy as sp

from . import linalg

__all__ = [
    "StationaryDistribution",
    "ForwardFeynmanKac",
    "BackwardFeynmanKac",
]


class Statistic(ABC):
    @abstractmethod
    def matrices(
        self, lag: int | None
    ) -> tuple[np.ndarray | sp.sparse.sparray | sp.sparse.spmatrix, np.ndarray]: ...
    @abstractmethod
    def gram_matrix(self) -> np.ndarray | sp.sparse.sparray | sp.sparse.spmatrix: ...
    @abstractmethod
    def transform(self, coef: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def transform_difference(self, coef: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def propagate(self, u: np.ndarray, lag: int) -> np.ndarray: ...
    @abstractmethod
    def propagate_difference(self, u: np.ndarray, lag: int) -> np.ndarray: ...


class StationaryDistribution(Statistic):
    def __init__(self, generator, basis, weights, test_basis=None):
        if test_basis is None:
            test_basis = basis
        self.generator = generator
        self.basis = basis
        self.weights = weights
        self.test_basis = test_basis

    def matrices(self, lag):
        L = self.generator.T
        x = self.test_basis
        y = self.basis
        w = self.weights

        if lag is None:
            wy = w[:, None] * y
            a = x.T @ L @ wy
            b = x.T @ L @ w
        else:
            assert lag > 0
            T = sp.linalg.expm(L * lag)
            wy = w[:, None] * y
            a = x.T @ (T @ wy - wy)
            b = x.T @ (T @ w - w)
        return a, b

    def gram_matrix(self):
        x = self.test_basis
        y = self.basis
        w = self.weights

        c0 = x.T * w @ y
        return c0

    def transform(self, coef):
        return self.weights * (self.basis @ coef + 1.0)

    def transform_difference(self, coef):
        return self.weights * (self.basis @ coef)

    def propagate(self, u, lag):
        assert lag >= 0
        L = self.generator.T
        T = sp.linalg.expm(L * lag)
        return T @ u

    def propagate_difference(self, u, lag):
        assert lag >= 0
        L = self.generator.T
        T = sp.linalg.expm(L * lag)
        return T @ u


class ForwardFeynmanKac(Statistic):
    def __init__(
        self,
        generator,
        basis,
        weights,
        in_domain,
        function,
        guess,
        test_basis=None,
    ):
        if test_basis is None:
            test_basis = basis
        self.generator = generator
        self.basis = basis
        self.weights = weights
        self.in_domain = in_domain
        self.function = function
        self.guess = guess
        self.test_basis = test_basis

    def matrices(self, lag):
        L = self.generator
        x = self.test_basis
        y = self.basis
        w = self.weights
        d = self.in_domain
        f = self.function
        g = self.guess

        Ld = L[np.ix_(d, d)]
        xwd = x[d].T * w[d]
        yd = y[d]
        if lag is None:
            a = xwd @ (Ld @ yd)
            b = xwd @ (L[d] @ g + f[d])
        else:
            assert lag > 0
            Sd = sp.linalg.expm(Ld * lag)
            rd = sp.linalg.solve(Ld, L[d] @ g + f[d])
            a = xwd @ (Sd @ yd - yd)
            b = xwd @ (Sd @ rd - rd)
        return a, b

    def gram_matrix(self):
        x = self.test_basis
        y = self.basis
        w = self.weights
        d = self.in_domain

        c0 = x[d].T * w[d] @ y[d]
        return c0

    def transform(self, coef):
        return self.basis @ coef + self.guess

    def transform_difference(self, coef):
        return self.basis @ coef

    def propagate(self, u, lag):
        assert lag >= 0
        L = self.generator
        d = self.in_domain
        f = self.function

        Ld = L[np.ix_(d, d)]
        Sd = sp.linalg.expm(Ld * lag)
        rd = sp.linalg.solve(Ld, L[d] @ u + f[d])

        out = u.copy()
        out[d] += Sd @ rd - rd
        return out

    def propagate_difference(self, u, lag):
        assert lag >= 0
        L = self.generator
        d = self.in_domain

        Ld = L[np.ix_(d, d)]
        Sd = sp.linalg.expm(Ld * lag)

        out = np.zeros(u.shape)
        out[d] = Sd @ u[d]
        return out


class BackwardFeynmanKac(Statistic):
    def __init__(
        self,
        generator,
        basis,
        weights,
        in_domain,
        function,
        guess,
        test_basis=None,
    ):
        if test_basis is None:
            test_basis = basis
        self.generator = generator
        self.basis = basis
        self.weights = weights
        self.in_domain = in_domain
        self.function = function
        self.guess = guess
        self.test_basis = test_basis

    def matrices(self, lag):
        L = self.generator.T
        x = self.test_basis
        y = self.basis
        w = self.weights
        d = self.in_domain
        f = self.function
        g = self.guess

        Ld = L[np.ix_(d, d)]
        xd = x[d].T
        yd = y[d]
        if lag is None:
            dw = L @ w
            a = xd @ (Ld @ (w[d, None] * yd) - dw[d, None] * yd)
            b = xd @ (L[d] @ (w * g) - dw[d] * g[d] + w[d] * f[d])
        else:
            assert lag > 0
            Sd = sp.linalg.expm(Ld * lag)
            T = sp.linalg.expm(L * lag)
            rd = L[d] * g - g[d, None] * L[d] + np.diag(f)[d]
            F = sp.linalg.solve_sylvester(Ld, -L, rd)
            Rd = Sd @ F - F @ T
            wt = sp.linalg.solve(T, w)
            a = xd @ (Sd @ (wt[d, None] * yd) - w[d, None] * yd)
            b = xd @ (Rd @ wt)
        return a, b

    def gram_matrix(self):
        x = self.test_basis
        y = self.basis
        w = self.weights
        d = self.in_domain

        c0 = x[d].T * w[d] @ y[d]
        return c0

    def transform(self, coef):
        return self.basis @ coef + self.guess

    def transform_difference(self, coef):
        return self.basis @ coef

    def propagate(self, u, lag):
        assert lag >= 0
        L = self.generator.T
        w = self.weights
        d = self.in_domain
        f = self.function

        Ld = L[np.ix_(d, d)]
        Sd = sp.linalg.expm(Ld * lag)
        T = sp.linalg.expm(L * lag)
        rd = L[d] * u - u[d, None] * L[d] + np.diag(f)[d]
        F = sp.linalg.solve_sylvester(Ld, -L, rd)
        Rd = Sd @ F - F @ T

        wt = linalg.solve(T, w)
        out = u.copy()
        out[d] += (Rd @ wt) / w[d]
        return out

    def propagate_difference(self, u, lag):
        assert lag >= 0
        L = self.generator.T
        w = self.weights
        d = self.in_domain

        Ld = L[np.ix_(d, d)]
        Sd = sp.linalg.expm(Ld * lag)
        T = sp.linalg.expm(L * lag)

        wt = linalg.solve(T, w)
        out = np.zeros(u.shape)
        out[d] = (Sd @ (wt * u)[d]) / w[d]
        return out
