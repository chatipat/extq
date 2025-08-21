"""Finite difference reference calculation for DGA with memory."""

from abc import ABC, abstractmethod

import numpy as np
import scipy as sp

from . import linalg

__all__ = [
    "reweight",
    "forward_committor",
    "forward_mfpt",
    "forward_feynman_kac",
    "backward_committor",
    "backward_mfpt",
    "backward_feynman_kac",
    "solve",
    "DGAWithMemory",
    "StationaryDistribution",
    "ForwardFeynmanKac",
    "BackwardFeynmanKac",
]


def reweight(
    generator,
    basis,
    weights,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    assert return_projection or return_solution or return_coef or return_mem_coef
    stat = StationaryDistribution(generator, basis, weights, test_basis=test_basis)
    algo = DGAWithMemory(lag, mem).fit(stat)
    coef, mem_coef = algo.get_parameters()
    out = []
    if return_projection:
        out.append(algo.projection(stat))
    if return_solution:
        out.append(algo.solution(stat))
    if return_coef:
        out.append(coef)
    if return_mem_coef:
        out.append(mem_coef)
    if len(out) == 1:
        out = out[0]
    return out


def forward_committor(
    generator,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    return forward_feynman_kac(
        generator,
        basis,
        weights,
        in_domain,
        0.0,
        guess,
        lag,
        mem,
        test_basis,
        return_projection=return_projection,
        return_solution=return_solution,
        return_coef=return_coef,
        return_mem_coef=return_mem_coef,
    )


def forward_mfpt(
    generator,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    return forward_feynman_kac(
        generator,
        basis,
        weights,
        in_domain,
        1.0,
        guess,
        lag,
        mem,
        test_basis,
        return_projection=return_projection,
        return_solution=return_solution,
        return_coef=return_coef,
        return_mem_coef=return_mem_coef,
    )


def forward_feynman_kac(
    generator,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    assert return_projection or return_solution or return_coef or return_mem_coef
    stat = ForwardFeynmanKac(
        generator,
        basis,
        weights,
        in_domain,
        function,
        guess,
        test_basis=test_basis,
    )
    algo = DGAWithMemory(lag, mem).fit(stat)
    coef, mem_coef = algo.get_parameters()
    out = []
    if return_projection:
        out.append(algo.projection(stat))
    if return_solution:
        out.append(algo.solution(stat))
    if return_coef:
        out.append(coef)
    if return_mem_coef:
        out.append(mem_coef)
    if len(out) == 1:
        out = out[0]
    return out


def backward_committor(
    generator,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    return backward_feynman_kac(
        generator,
        basis,
        weights,
        in_domain,
        0.0,
        guess,
        lag,
        mem,
        test_basis,
        return_projection=return_projection,
        return_solution=return_solution,
        return_coef=return_coef,
        return_mem_coef=return_mem_coef,
    )


def backward_mfpt(
    generator,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    return backward_feynman_kac(
        generator,
        basis,
        weights,
        in_domain,
        1.0,
        guess,
        lag,
        mem,
        test_basis,
        return_projection=return_projection,
        return_solution=return_solution,
        return_coef=return_coef,
        return_mem_coef=return_mem_coef,
    )


def backward_feynman_kac(
    generator,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=False,
    return_solution=True,
    return_coef=False,
    return_mem_coef=False,
):
    assert return_projection or return_solution or return_coef or return_mem_coef
    weights = sp.linalg.expm(generator.T * lag) @ weights
    stat = BackwardFeynmanKac(
        generator,
        basis,
        weights,
        in_domain,
        function,
        guess,
        test_basis=test_basis,
    )
    algo = DGAWithMemory(lag, mem).fit(stat)
    coef, mem_coef = algo.get_parameters()
    out = []
    if return_projection:
        out.append(algo.projection(stat))
    if return_solution:
        out.append(algo.solution(stat))
    if return_coef:
        out.append(coef)
    if return_mem_coef:
        out.append(mem_coef)
    if len(out) == 1:
        out = out[0]
    return out


def solve(a, b, c0):
    """
    Solve DGA with memory for projection and memory-correction
    coefficients.

    Parameters
    ----------
    a : (mem + 1, n_basis, n_basis) ndarray of float
        DGA matrices for the homogeneous term.
    b : (mem + 1, n_basis) ndarray of float
        DGA matrices for the nonhomogeneous term.
    c0 : (n_basis, n_basis) ndarray of float
        Matrix of inner products of basis functions.

    Returns
    -------
    coef : (n_basis,) ndarray of float
        Projection coefficients.
    mem_coef : (mem, n_basis) ndarray of float
        Memory-correction coefficients.

    """
    mem = a.shape[0] - 1
    n_basis = a.shape[1]
    assert a.shape == (mem + 1, n_basis, n_basis)
    assert b.shape == (mem + 1, n_basis)

    b = b[..., None]

    inv = sp.linalg.inv(c0)
    a = inv @ a
    b = inv @ b
    c = a[::-1] + np.identity(n_basis)
    for n in range(1, mem + 1):
        a[n] -= np.sum(c[-n:] @ a[:n], axis=0)
        b[n] -= np.sum(c[-n:] @ b[:n], axis=0)

    b = b.reshape(b.shape[:2])

    coef = sp.linalg.solve(a[-1], -b[-1])
    mem_coef = a[:-1] @ coef + b[:-1]
    return coef, mem_coef


class DGAWithMemory:
    def __init__(self, lag, mem):
        assert lag % (mem + 1) == 0
        self.lag = lag
        self.mem = mem
        self._dlag = lag // (mem + 1)
        self.params = None

    def get_parameters(self):
        if self.params is None:
            raise ValueError
        return self.params

    def set_parameters(self, params):
        self.params = params
        return self

    def fit(self, stat):
        a, b, c0 = self.matrices(stat)
        self.set_parameters(solve(a, b, c0))
        return self

    def matrices(self, stat):
        a = []
        b = []
        for n in range(self.mem + 1):
            lag = (n + 1) * self._dlag
            a_n, b_n = stat.matrices(lag)
            a.append(linalg.as_dense(a_n))
            b.append(linalg.as_dense(b_n))
        a = np.array(a)
        b = np.array(b)
        c0 = linalg.as_dense(stat.gram_matrix())
        return a, b, c0

    def projection(self, stat):
        coef, _ = self.get_parameters()
        return stat.transform(coef)

    def solution(self, stat):
        lag = self.lag
        dlag = self._dlag
        mem = self.mem
        coef, mem_coef = self.get_parameters()
        out = stat.propagate(stat.transform(coef), lag)
        for m in range(mem):
            out -= stat.propagate_difference(
                stat.transform_difference(mem_coef[m]), lag - dlag * (m + 1)
            )
        return out


class Statistic(ABC):
    @abstractmethod
    def matrices(self, lag: int) -> tuple[np.ndarray, np.ndarray]: ...
    @abstractmethod
    def gram_matrix(self) -> np.ndarray: ...
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
        assert lag >= 0
        L = self.generator.T
        x = self.test_basis
        y = self.basis
        w = self.weights

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
        assert lag >= 0
        L = self.generator
        x = self.test_basis
        y = self.basis
        w = self.weights
        d = self.in_domain
        f = self.function
        g = self.guess

        Ld = L[np.ix_(d, d)]
        Sd = sp.linalg.expm(Ld * lag)
        rd = sp.linalg.solve(Ld, L[d] @ g + f[d])

        xwd = x[d].T * w[d]
        yd = y[d]
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
        assert lag >= 0
        L = self.generator.T
        x = self.test_basis
        y = self.basis
        w = self.weights
        d = self.in_domain
        f = self.function
        g = self.guess

        Ld = L[np.ix_(d, d)]
        Sd = sp.linalg.expm(Ld * lag)
        T = sp.linalg.expm(L * lag)
        rd = L[d] * g - g[d, None] * L[d] + np.diag(f)[d]
        F = sp.linalg.solve_sylvester(Ld, -L, rd)
        Rd = Sd @ F - F @ T

        xd = x[d].T
        yd = y[d]
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
