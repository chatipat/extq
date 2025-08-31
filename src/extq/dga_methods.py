from abc import ABC, abstractmethod

import numpy as np

from . import linalg

__all__ = [
    "DGA",
    "DGAMemMZ",
    "DGAMemTM",
    "HankelDGA",
]


class DGAMethod(ABC):
    def __init__(self):
        self.state = None
        self.parameters = None

    def fit(self, stat):
        self.state = None
        self.parameters = None
        self.partial_fit(stat)
        self.solve()
        return self

    def fit_transform(self, stat, *, output="projection"):
        return self.fit(stat).transform(stat, output=output)

    def partial_fit(self, stat):
        self.parameters = None
        self.state = self._partial_fit(self.state, stat)
        return self

    def solve(self):
        if self.state is None:
            raise ValueError
        self.parameters = self._solve(self.state)
        return self

    def transform(self, stat, *, output="projection"):
        if self.parameters is None:
            raise ValueError
        return self._transform(self.parameters, stat, output)

    @abstractmethod
    def _partial_fit(self, state, stat) -> object: ...

    @abstractmethod
    def _solve(self, state) -> object: ...

    @abstractmethod
    def _transform(self, parameters, stat, output) -> object: ...


class DGA(DGAMethod):
    def __init__(self, lag):
        super().__init__()
        self.lag = lag

    def _partial_fit(self, state, stat):
        lag = self.lag

        a, b = stat.matrices(lag)
        if state is not None:
            a_old, b_old = state
            a = a + a_old
            b = b + b_old
        return a, b

    def _solve(self, state):
        a, b = state
        coef = -linalg.solve(a, b)
        return coef

    def _transform(self, parameters, stat, output):
        lag = self.lag

        coef = parameters
        if output == "projection":
            out = stat.transform(coef)
        elif output == "solution":
            out = stat.transform(coef)
            out = stat.propagate(out, lag)
        else:
            raise ValueError
        return out


class DGAMemMZ(DGAMethod):
    def __init__(self, lag, mem):
        super().__init__()
        assert lag % (mem + 1) == 0
        self.lag = lag
        self.mem = mem
        self._dlag = lag // (mem + 1)

    def _partial_fit(self, state, stat):
        dlag = self._dlag
        mem = self.mem

        a = np.full(mem + 1, None)
        b = np.full(mem + 1, None)
        for n in range(mem + 1):
            lag = (n + 1) * dlag
            a_n, b_n = stat.matrices(lag)
            a[n] = a_n
            b[n] = b_n
        c0 = stat.gram_matrix()
        if state is not None:
            a_old, b_old, c0_old = state
            a = a + a_old
            b = b + b_old
            c0 = c0 + c0_old
        return a, b, c0

    def _solve(self, state):
        mem = self.mem

        a, b, c0 = state

        n_basis = c0.shape[0]
        assert len(a) == len(b) == mem + 1
        assert all(ai.shape == (n_basis, n_basis) for ai in a)
        assert all(bi.shape == (n_basis,) for bi in b)
        assert c0.shape == (n_basis, n_basis)

        da = np.full(mem + 1, None)
        db = np.full(mem + 1, None)
        for i in range(mem + 1):
            if i == 0:
                da[i] = a[i]
                db[i] = b[i]
            else:
                da[i] = a[i] - a[i - 1]
                db[i] = b[i] - b[i - 1]

        lhs = np.full((mem + 1, mem + 1), None)
        for i in range(mem + 1):
            for j in range(mem + 1):
                if j <= i:
                    lhs[i, j] = da[i - j]
                elif j == i + 1:
                    lhs[i, j] = c0
        lhs = linalg.block(lhs)
        rhs = np.concatenate(db)

        coef = -linalg.solve(lhs, rhs)
        coef = coef.reshape(mem + 1, n_basis)
        return coef

    def _transform(self, parameters, stat, output):
        lag = self.lag
        dlag = self._dlag
        mem = self.mem

        coef = parameters
        if output == "projection":
            out = stat.transform(coef[0])
        elif output == "solution":
            out = stat.propagate(stat.transform(coef[0]), lag)
            for m in range(1, mem + 1):
                diff = stat.transform_difference(coef[m])
                out = out + stat.propagate_difference(diff, lag - dlag * m)
        else:
            raise ValueError
        return out


class DGAMemTM(DGAMethod):
    def __init__(self, lag1, lag2):
        super().__init__()
        self.lag1 = lag1
        self.lag2 = lag2

    def _partial_fit(self, state, stat):
        lag1 = self.lag1
        lag2 = self.lag2

        a1, b1 = stat.matrices(lag1)
        a2, b2 = stat.matrices(lag2)
        c0 = stat.gram_matrix()
        if state is not None:
            (a1_old, a2_old), (b1_old, b2_old), c0_old = state
            a1 = a1 + a1_old
            a2 = a2 + a2_old
            b1 = b1 + b1_old
            b2 = b2 + b2_old
            c0 = c0 + c0_old
        return (a1, a2), (b1, b2), c0

    def _solve(self, state):
        (a1, a2), (b1, b2), c0 = state
        s_coef = -linalg.solve(a2 - a1, b2 - b1)
        p_coef = s_coef + linalg.solve(c0, a1 @ s_coef + b1)
        return p_coef, s_coef

    def _transform(self, parameters, stat, output):
        lag1 = self.lag1
        lag2 = self.lag2

        p_coef, s_coef = parameters
        if output == "projection":
            out = stat.transform(p_coef)
        elif output == "solution1":
            out = stat.transform(s_coef)
            out = stat.propagate(out, lag1)
        elif output == "solution1":
            out = stat.transform(s_coef)
            out = stat.propagate(out, lag2)
        else:
            raise ValueError
        return out


class HankelDGA(DGAMethod):
    def __init__(self, delay, n_delays):
        assert delay > 0
        assert n_delays > 0 and n_delays % 2 == 1
        self.delay = delay
        self.n_delays = n_delays

    def _partial_fit(self, state, stat):
        a = []
        b = []
        for n in range(self.n_delays):
            # lag = 0 can be skipped because DGA matrices are zero
            lag = self.delay * (n + 1)
            a_n, b_n = stat.matrices(lag)
            a.append(a_n)
            b.append(b_n)
        if state is not None:
            a_old, b_old = state
            a = [a_n + a_n_old for a_n, a_n_old in zip(a, a_old, strict=True)]
            b = [b_n + b_n_old for b_n, b_n_old in zip(b, b_old, strict=True)]
        return a, b

    def _solve(self, state):
        a_mats, b_mats = state

        a_mats = list(a_mats)
        b_mats = list(b_mats)
        assert len(a_mats) == len(b_mats)
        n_delays = len(a_mats)
        assert n_delays % 2 == 1
        n_blocks = (n_delays + 1) // 2

        a_diff = np.full(n_delays + 1, None)
        a_diff[0] = 0
        a_diff[1:] = a_mats
        a_diff = np.diff(a_diff)

        a = np.full((n_blocks, n_blocks), None)
        for i in range(n_blocks):
            for j in range(n_blocks):
                a[i, j] = a_diff[i + j]
        a = linalg.block(a.tolist())

        b = np.full(n_delays + 1, None)
        b[0] = 0
        b[1:] = b_mats
        b = b[n_blocks:] - b[:n_blocks]
        b = np.concatenate(b.tolist())

        coef = -linalg.solve(a, b)
        return coef.reshape(n_blocks, -1)

    def _transform(self, parameters, stat, output):
        assert output == "solution"
        coef = parameters
        n_blocks, _ = coef.shape
        out = None
        for n in range(n_blocks):
            lag = self.delay * n
            u = stat.transform(coef[n])
            u = stat.propagate(u, lag)
            if out is None:
                out = u
            else:
                out = out + u
        return out
