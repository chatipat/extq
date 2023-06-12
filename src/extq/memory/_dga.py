"""DGA with memory estimators for statistics."""


import numpy as np
import scipy.sparse
from more_itertools import zip_equal

from .. import linalg
from ..stop import backward_stop, forward_stop

__all__ = [
    "reweight",
    "reweight_matrices",
    "reweight_projection",
    "reweight_solution",
    "forward_committor",
    "forward_mfpt",
    "forward_feynman_kac",
    "forward_feynman_kac_matrices",
    "forward_feynman_kac_projection",
    "forward_feynman_kac_solution",
    "backward_committor",
    "backward_mfpt",
    "backward_feynman_kac",
    "backward_feynman_kac_matrices",
    "backward_feynman_kac_projection",
    "backward_feynman_kac_solution",
]


def reweight(
    basis,
    weights,
    lag,
    mem,
    test_basis=None,
    return_projection=False,
    return_solution=True,
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
    assert (
        return_projection or return_solution or return_coef or return_mem_coef
    )
    a, b, c0 = reweight_matrices(
        basis, weights, lag, mem, test_basis=test_basis
    )
    coef, mem_coef = _dga_mem(a, b, c0)
    out = []
    if return_projection:
        out.append(reweight_projection(basis, weights, coef))
    if return_solution:
        out.append(reweight_solution(basis, weights, lag, mem, coef, mem_coef))
    if return_coef:
        out.append(coef)
    if return_mem_coef:
        out.append(mem_coef)
    if len(out) == 1:
        out = out[0]
    return out


def reweight_matrices(basis, weights, lag, mem, test_basis=None):
    """
    Compute DGA matrices for estimating the invariant distribution.

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

    Returns
    -------
    a : (mem + 1, n_basis, n_basis) ndarray of float
        DGA matrices for the homogeneous term.
    b : (mem + 1, n_basis) ndarray of float
        DGA matrices for the nonhomogeneous term.
    c0 : (n_basis, n_basis) ndarray of float
        Matrix of inner products of basis functions.

    """
    if test_basis is None:
        test_basis = basis

    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    a = np.zeros((mem + 1, n_basis, n_basis))
    b = np.zeros((mem + 1, n_basis))
    c0 = np.zeros((n_basis, n_basis))

    for x, y, w in zip_equal(test_basis, basis, weights):
        n_frames = len(w)
        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)

        if n_frames <= lag:
            assert np.all(w == 0.0)
            continue
        end = n_frames - lag
        assert np.all(w[end:] == 0.0)

        wy = linalg.scale_rows(w[:end], y[:end])
        for n in range(mem + 1):
            dx = (x[(n + 1) * dlag : end + (n + 1) * dlag] - x[:end]).T
            a[n] += _densify(dx @ wy)
            b[n] += dx @ w[:end]
        c0 += x[:end].T @ wy

    return a, b, c0


def reweight_projection(basis, weights, coef):
    """
    Returns the projected invariant distribution.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the invariant distribution. The span of
        `basis` must *not* contain the constant function.
    weights : sequence of (n_frames[i],) ndarray of float
        Weight of each frame.
    coef : (n_basis,) ndarray of float
        Projection coefficients.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the projected invariant distribution.

    """
    return [w * (y @ coef + 1.0) for y, w in zip_equal(basis, weights)]


def reweight_solution(basis, weights, lag, mem, coef, mem_coef):
    """
    Returns a stochastic approximation of the invariant distribution.

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
    coef : (n_basis,) ndarray of float
        Projection coefficients.
    mem_coef : (mem, n_basis) ndarray of float
        Memory-correction coefficients.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the invariant distribution.

    """
    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    out = []
    for y, w in zip_equal(basis, weights):
        n_frames = y.shape[0]
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)

        if n_frames <= lag:
            assert np.all(w == 0.0)
            out.append(np.zeros(n_frames))
            continue
        assert np.all(w[-lag:] == 0.0)

        pad = np.zeros(dlag)
        u = w * (y @ coef + 1.0)
        for v in mem_coef:
            u = np.concatenate([pad, u[:-dlag]])
            u -= w * (y @ v)
        u = np.concatenate([pad, u[:-dlag]])
        out.append(u)
    return out


def forward_committor(
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    return_projection=False,
    return_solution=True,
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
    function = np.zeros(len(weights))
    return forward_feynman_kac(
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem,
        test_basis,
        return_projection,
        return_solution,
        return_coef,
        return_mem_coef,
    )


def forward_mfpt(
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    return_projection=False,
    return_solution=True,
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
    function = np.ones(len(weights))
    return forward_feynman_kac(
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem,
        test_basis,
        return_projection,
        return_solution,
        return_coef,
        return_mem_coef,
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
    return_projection=False,
    return_solution=True,
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
    assert (
        return_projection or return_solution or return_coef or return_mem_coef
    )
    a, b, c0 = forward_feynman_kac_matrices(
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem,
        test_basis=test_basis,
    )
    coef, mem_coef = _dga_mem(a, b, c0)
    out = []
    if return_projection:
        out.append(forward_feynman_kac_projection(basis, guess, coef))
    if return_solution:
        out.append(
            forward_feynman_kac_solution(
                basis, in_domain, function, guess, lag, mem, coef, mem_coef
            )
        )
    if return_coef:
        out.append(coef)
    if return_mem_coef:
        out.append(mem_coef)
    if len(out) == 1:
        out = out[0]
    return out


def forward_feynman_kac_matrices(
    basis, weights, in_domain, function, guess, lag, mem, test_basis=None
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

    Returns
    -------
    a : (mem + 1, n_basis, n_basis) ndarray of float
        DGA matrices for the homogeneous term.
    b : (mem + 1, n_basis) ndarray of float
        DGA matrices for the nonhomogeneous term.
    c0 : (n_basis, n_basis) ndarray of float
        Matrix of inner products of basis functions.

    """
    if test_basis is None:
        test_basis = basis

    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    a = np.zeros((mem + 1, n_basis, n_basis))
    b = np.zeros((mem + 1, n_basis))
    c0 = np.zeros((n_basis, n_basis))

    for x, y, w, d, f, g in zip_equal(
        test_basis, basis, weights, in_domain, function, guess
    ):
        n_frames = len(w)
        f = np.broadcast_to(f, n_frames - 1)
        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)

        if n_frames <= lag:
            assert np.all(w == 0.0)
            continue
        end = n_frames - lag
        assert np.all(w[end:] == 0.0)

        ix = np.arange(end)
        stop = forward_stop(d)[:end]
        intf = np.insert(np.cumsum(f), 0, 0.0)
        xw = linalg.scale_rows(w[:end], x[:end]).T
        for n in range(mem + 1):
            iy = np.minimum(ix + (n + 1) * dlag, stop)
            a[n] += _densify(xw @ (y[iy] - y[:end]))
            b[n] += xw @ ((g[iy] - g[:end]) + (intf[iy] - intf[:end]))
        c0 += _densify(xw @ y[:end])

    return a, b, c0


def forward_feynman_kac_projection(basis, guess, coef):
    """
    Returns the projected solution of a forward Feynman-Kac problem.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.
    coef : (n_basis,) ndarray of float
        Projection coefficients.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the projected solution.

    """
    return [y @ coef + g for y, g in zip_equal(basis, guess)]


def forward_feynman_kac_solution(
    basis, in_domain, function, guess, lag, mem, coef, mem_coef
):
    """
    Returns a stochastic approximation of the solution of a forward
    Feynman-Kac problem.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
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
    coef : (n_basis,) ndarray of float
        Projection coefficients.
    mem_coef : (mem, n_basis) ndarray of float
        Memory-correction coefficients.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the solution.

    """
    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    out = []
    for y, d, f, g in zip_equal(basis, in_domain, function, guess):
        n_frames = y.shape[0]
        f = np.broadcast_to(f, n_frames - 1)
        assert y.shape == (n_frames, n_basis)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)

        if n_frames <= lag:
            out.append(np.full(n_frames, np.nan))
            continue

        stop = np.minimum(np.arange(dlag, len(d)), forward_stop(d)[:-dlag])
        intf = np.insert(np.cumsum(f), 0, 0.0)
        r = intf[stop] - intf[:-dlag]
        pad = np.full(dlag, np.nan)
        u = y @ coef + g
        for v in mem_coef:
            u = np.concatenate([u[stop] + r, pad])
            u -= y @ v
        u = np.concatenate([u[stop] + r, pad])
        out.append(u)
    return out


def backward_committor(
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    return_projection=False,
    return_solution=True,
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
    function = np.zeros(len(weights))
    return backward_feynman_kac(
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem,
        test_basis,
        return_projection,
        return_solution,
        return_coef,
        return_mem_coef,
    )


def backward_mfpt(
    basis,
    weights,
    in_domain,
    guess,
    lag,
    mem,
    test_basis=None,
    return_projection=False,
    return_solution=True,
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
    function = np.ones(len(weights))
    return backward_feynman_kac(
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem,
        test_basis,
        return_projection,
        return_solution,
        return_coef,
        return_mem_coef,
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
    return_projection=False,
    return_solution=True,
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
    assert (
        return_projection or return_solution or return_coef or return_mem_coef
    )
    a, b, c0 = backward_feynman_kac_matrices(
        basis,
        weights,
        in_domain,
        function,
        guess,
        lag,
        mem,
        test_basis=test_basis,
    )
    coef, mem_coef = _dga_mem(a, b, c0)
    out = []
    if return_projection:
        out.append(backward_feynman_kac_projection(basis, guess, coef))
    if return_solution:
        out.append(
            backward_feynman_kac_solution(
                basis, in_domain, function, guess, lag, mem, coef, mem_coef
            )
        )
    if return_coef:
        out.append(coef)
    if return_mem_coef:
        out.append(mem_coef)
    if len(out) == 1:
        out = out[0]
    return out


def backward_feynman_kac_matrices(
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    mem,
    test_basis=None,
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

    Returns
    -------
    a : (mem + 1, n_basis, n_basis) ndarray of float
        DGA matrices for the homogeneous term.
    b : (mem + 1, n_basis) ndarray of float
        DGA matrices for the nonhomogeneous term.
    c0 : (n_basis, n_basis) ndarray of float
        Matrix of inner products of basis functions.

    """
    if test_basis is None:
        test_basis = basis

    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    a = np.zeros((mem + 1, n_basis, n_basis))
    b = np.zeros((mem + 1, n_basis))
    c0 = np.zeros((n_basis, n_basis))

    for x, y, w, d, f, g in zip_equal(
        test_basis, basis, weights, in_domain, function, guess
    ):
        n_frames = len(w)
        f = np.broadcast_to(f, n_frames - 1)
        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)

        if n_frames <= lag:
            assert np.all(w == 0.0)
            continue
        end = n_frames - lag
        assert np.all(w[end:] == 0.0)

        ix = np.arange(lag, n_frames)
        stop = backward_stop(d)[lag:]
        intf = np.insert(np.cumsum(f), 0, 0.0)
        xw = linalg.scale_rows(w[:end], x[lag:]).T
        for n in range(mem + 1):
            iy = np.maximum(ix - (n + 1) * dlag, stop)
            a[n] += _densify(xw @ (y[iy] - y[lag:]))
            b[n] += xw @ ((g[iy] - g[lag:]) + (intf[lag:] - intf[iy]))
        c0 += xw @ y[lag:]

    return a, b, c0


def backward_feynman_kac_projection(basis, guess, coef):
    """
    Returns the projected solution of a backward Feynman-Kac problem.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
    guess : sequence of (n_frames[i],) ndarray of float
        Guess for the solution. Must satisfy boundary conditions.
    coef : (n_basis,) ndarray of float
        Projection coefficients.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the projected solution.

    """
    return [y @ coef + g for y, g in zip_equal(basis, guess)]


def backward_feynman_kac_solution(
    basis, in_domain, function, guess, lag, mem, coef, mem_coef
):
    """
    Returns a stochastic approximation of the solution of a backward
    Feynman-Kac problem.

    Parameters
    ----------
    basis : sequence of (n_frames[i], n_basis) {ndarray, sparse matrix} of float
        Basis for estimating the solution. Must be zero outside of the
        domain.
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
    coef : (n_basis,) ndarray of float
        Projection coefficients.
    mem_coef : (mem, n_basis) ndarray of float
        Memory-correction coefficients.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimate of the solution.

    """
    assert lag % (mem + 1) == 0
    dlag = lag // (mem + 1)
    n_basis = basis[0].shape[1]

    out = []
    for y, d, f, g in zip_equal(basis, in_domain, function, guess):
        n_frames = y.shape[0]
        f = np.broadcast_to(f, n_frames - 1)
        assert y.shape == (n_frames, n_basis)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)

        if n_frames <= lag:
            out.append(np.full(n_frames, np.nan))
            continue

        stop = np.maximum(np.arange(len(d) - dlag), backward_stop(d)[dlag:])
        intf = np.insert(np.cumsum(f), 0, 0.0)
        r = intf[dlag:] - intf[stop]
        pad = np.full(dlag, np.nan)
        u = y @ coef + g
        for v in mem_coef:
            u = np.concatenate([pad, u[stop] + r])
            u -= y @ v
        u = np.concatenate([pad, u[stop] + r])
        out.append(u)
    return out


def _densify(a):
    """Convert the input to a dense array."""
    if scipy.sparse.issparse(a):
        a = a.toarray()
    return a


def _dga_mem(a, b, c0):
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
