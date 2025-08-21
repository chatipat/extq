"""DGA with memory estimators for statistics."""

import numpy as np

from . import dgastat, linalg, utils

__all__ = [
    "reweight",
    "forward_committor",
    "forward_mfpt",
    "forward_feynman_kac",
    "backward_committor",
    "backward_mfpt",
    "backward_feynman_kac",
    "DGAWithMemory",
]


def reweight(
    basis,
    weights,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=True,
    return_solution=False,
    return_coef=False,
    return_mem_coef=False,
):
    """
    Estimate the invariant distribution using DGA with memory.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the invariant distribution. The span of
        `basis` must *not* contain the constant function.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.
    return_projection : bool, optional
        If True, return the projected invariant distribution.
    return_solution : bool, optional
        If True (default), return a stochastic approximation of the invariant
        distribution.
    return_coef : bool, optional
        If True, return the projection coefficients.
    return_mem_coef : bool, optional
        If True, return the memory-correction coefficients.

    Returns
    -------
    projection : list of (n_frames[i],) ndarray of float
        Estimate of the projected invariant distribution.
    solution : list of (n_frames[i],) ndarray of float
        Estimate of the invariant distribution.
    coef : (n_basis,) ndarray of float
        Projection coefficients.
    mem_coef : (mem, n_basis) ndarray of float
        Memory-correction coefficients.

    """
    assert return_projection or return_solution or return_coef or return_mem_coef
    stat = dgastat.StationaryDistribution(basis, weights, test_basis=test_basis)
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
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=True,
    return_solution=False,
    return_coef=False,
    return_mem_coef=False,
):
    """
    Estimate the forward committor using DGA with memory.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the committor. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.
    return_projection : bool, optional
        If True, return the projected committor.
    return_solution : bool, optional
        If True (default), return a stochastic approximation of the committor.
    return_coef : bool, optional
        If True, return the projection coefficients.
    return_mem_coef : bool, optional
        If True, return the memory-correction coefficients.

    Returns
    -------
    projection : list of (n_frames[i],) ndarray of float
        Estimate of the projected committor.
    solution : list of (n_frames[i],) ndarray of float
        Estimate of the committor.
    coef : (n_basis,) ndarray of float
        Projection coefficients.
    mem_coef : (mem, n_basis) ndarray of float
        Memory-correction coefficients.

    """
    return forward_feynman_kac(
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
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=True,
    return_solution=False,
    return_coef=False,
    return_mem_coef=False,
):
    """
    Estimate the forward mean first passage time (MFPT) using DGA
    with memory.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the MFPT. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the MFPT. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.
    return_projection : bool, optional
        If True, return the projected MFPT.
    return_solution : bool, optional
        If True (default), return a stochastic approximation of the MFPT.
    return_coef : bool, optional
        If True, return the projection coefficients.
    return_mem_coef : bool, optional
        If True, return the memory-correction coefficients.

    Returns
    -------
    projection : list of (n_frames[i],) ndarray of float
        Estimate of the projected MFPT.
    solution : list of (n_frames[i],) ndarray of float
        Estimate of the MFPT.
    coef : (n_basis,) ndarray of float
        Projection coefficients.
    mem_coef : (mem, n_basis) ndarray of float
        Memory-correction coefficients.

    """
    return forward_feynman_kac(
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
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=True,
    return_solution=False,
    return_coef=False,
    return_mem_coef=False,
):
    """
    Solve a forward Feynman-Kac problem using DGA with memory.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame. The last `lag` frames of each trajectory
        must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    function : sequence of (n_frames[i] - 1,) ndarray of float
        Function to integrate. This is defined over *steps*, not frames.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.
    return_projection : bool, optional
        If True, return the projected solution.
    return_solution : bool, optional
        If True (default), return a stochastic approximation of the solution.
    return_coef : bool, optional
        If True, return the projection coefficients.
    return_mem_coef : bool, optional
        If True, return the memory-correction coefficients.

    Returns
    -------
    projection : list of (n_frames[i],) ndarray of float
        Estimate of the projected solution.
    solution : list of (n_frames[i],) ndarray of float
        Estimate of the solution.
    coef : (n_basis,) ndarray of float
        Projection coefficients.
    mem_coef : (mem, n_basis) ndarray of float
        Memory-correction coefficients.

    """
    assert return_projection or return_solution or return_coef or return_mem_coef
    stat = dgastat.ForwardFeynmanKac(
        basis, weights, in_domain, function, guess, test_basis=test_basis
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
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=True,
    return_solution=False,
    return_coef=False,
    return_mem_coef=False,
):
    """
    Estimate the backward committor using DGA with memory.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the committor. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution. The last `lag`
        frames of each trajectory must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the committor. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.
    return_projection : bool, optional
        If True, return the projected committor.
    return_solution : bool, optional
        If True (default), return a stochastic approximation of the committor.
    return_coef : bool, optional
        If True, return the projection coefficients.
    return_mem_coef : bool, optional
        If True, return the memory-correction coefficients.

    Returns
    -------
    projection : list of (n_frames[i],) ndarray of float
        Estimate of the projected committor.
    solution : list of (n_frames[i],) ndarray of float
        Estimate of the committor.
    coef : (n_basis,) ndarray of float
        Projection coefficients.
    mem_coef : (mem, n_basis) ndarray of float
        Memory-correction coefficients.

    """
    return backward_feynman_kac(
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
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=True,
    return_solution=False,
    return_coef=False,
    return_mem_coef=False,
):
    """
    Estimate the backward mean first passage time (MFPT) using DGA
    with memory.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the MFPT. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution. The last `lag`
        frames of each trajectory must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the MFPT. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.
    return_projection : bool, optional
        If True, return the projected MFPT.
    return_solution : bool, optional
        If True (default), return a stochastic approximation of the MFPT.
    return_coef : bool, optional
        If True, return the projection coefficients.
    return_mem_coef : bool, optional
        If True, return the memory-correction coefficients.

    Returns
    -------
    projection : list of (n_frames[i],) ndarray of float
        Estimate of the projected MFPT.
    solution : list of (n_frames[i],) ndarray of float
        Estimate of the MFPT.
    coef : (n_basis,) ndarray of float
        Projection coefficients.
    mem_coef : (mem, n_basis) ndarray of float
        Memory-correction coefficients.

    """
    return backward_feynman_kac(
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
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    test_basis=None,
    *,
    return_projection=True,
    return_solution=False,
    return_coef=False,
    return_mem_coef=False,
):
    """
    Solve a backward Feynman-Kac problem using DGA with memory.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    weights : sequence of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution. The last `lag`
        frames of each trajectory must be zero.
    in_domain : sequence of (n_frames[i],) ndarray of bool
        Whether each frame is in the domain.
    function : sequence of (n_frames[i] - 1,) ndarray of float
        Function to integrate. This is defined over *steps*, not frames.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.
    lag : int
        Maximum lag time in units of frames.
    mem : int
        Number of memory terms to use. These are evaluated at equally
        spaced times between time 0 and time `lag`, so `mem+1` must
        evenly divide `lag`. For example, with a `lag=32`, `mem=3` and
        `mem=7` are fine since 7+1=8 and 3+1=4 evenly divide 32. Setting
        `mem=0` corresponds to not using memory.
    test_basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as `basis`. If `None`, use `basis`.
    return_projection : bool, optional
        If True, return the projected solution.
    return_solution : bool, optional
        If True (default), return a stochastic approximation of the solution.
    return_coef : bool, optional
        If True, return the projection coefficients.
    return_mem_coef : bool, optional
        If True, return the memory-correction coefficients.

    Returns
    -------
    projection : list of (n_frames[i],) ndarray of float
        Estimate of the projected solution.
    solution : list of (n_frames[i],) ndarray of float
        Estimate of the solution.
    coef : (n_basis,) ndarray of float
        Projection coefficients.
    mem_coef : (mem, n_basis) ndarray of float
        Memory-correction coefficients.

    """
    assert return_projection or return_solution or return_coef or return_mem_coef
    weights = utils.shift_weights(weights, lag)
    stat = dgastat.BackwardFeynmanKac(
        basis, weights, in_domain, function, guess, test_basis=test_basis
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


class DGAWithMemory:
    def __init__(self, lag, mem):
        assert lag % (mem + 1) == 0
        self.lag = lag
        self.mem = mem
        self._dlag = lag // (mem + 1)
        self.params = None

    def get_parameters(self):
        """
        Return fit parameters.

        Returns
        -------
        coef : (n_basis,) ndarray of float
            Projection coefficients.
        mem_coef : (mem, n_basis) ndarray of float
            Memory-correction coefficients.

        """
        if self.params is None:
            raise ValueError
        return self.params

    def set_parameters(self, coef, mem_coef):
        """
        Set fit parameters.

        Parameters
        ----------
        coef : (n_basis,) ndarray of float
            Projection coefficients.
        mem_coef : (mem, n_basis) ndarray of float
            Memory-correction coefficients.

        Returns
        -------
        self

        """
        self.params = (coef, mem_coef)
        return self

    def fit(self, stat):
        """
        Fit statistic to data.

        Parameters
        ----------
        stat
            DGA statistic.

        Returns
        -------
        self

        """
        a, b, c0 = self.matrices(stat)
        coef, mem_coef = solve(a, b, c0)
        self.set_parameters(coef, mem_coef)
        return self

    def matrices(self, stat):
        """
        Compute DGA matrices.

        Parameters
        ----------
        stat
            DGA statistic.

        Returns
        -------
        a : (mem + 1, n_basis, n_basis) ndarray of float
            DGA matrices for the homogeneous term.
        b : (mem + 1, n_basis) ndarray of float
            DGA matrices for the nonhomogeneous term.
        c0 : (n_basis, n_basis) ndarray of float
            Matrix of inner products of basis functions.

        """
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
        """
        Returns the projected solution.

        Parameters
        ----------
        stat
            DGA statistic.

        Returns
        -------
        list of (n_frames[i],) ndarray of float
            Estimate of the projected solution.

        """
        coef, _ = self.get_parameters()
        return stat.transform(coef)

    def solution(self, stat):
        """
        Returns a stochastic approximation of the solution.

        Parameters
        ----------
        stat
            DGA statistic.

        Returns
        -------
        list of (n_frames[i],) ndarray of float
            Estimate of the solution.

        """
        lag = self.lag
        dlag = self._dlag
        mem = self.mem
        coef, mem_coef = self.get_parameters()
        out = stat.propagate(stat.transform(coef), lag)
        for m in range(mem):
            out = out - stat.propagate_difference(
                stat.transform_difference(mem_coef[m]), lag - dlag * (m + 1)
            )
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

    inv = linalg.inv(c0)
    a = inv @ a
    b = inv @ b
    c = a[::-1] + np.identity(n_basis)
    for n in range(1, mem + 1):
        a[n] -= np.sum(c[-n:] @ a[:n], axis=0)
        b[n] -= np.sum(c[-n:] @ b[:n], axis=0)

    b = b.reshape(b.shape[:2])

    coef = linalg.solve(a[-1], -b[-1])
    mem_coef = a[:-1] @ coef + b[:-1]
    return coef, mem_coef
