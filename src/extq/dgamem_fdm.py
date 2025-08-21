"""Finite difference reference calculation for DGA with memory."""

import scipy as sp

from . import dgastat_fdm
from .dgamem import DGAWithMemory

__all__ = [
    "reweight",
    "forward_committor",
    "forward_mfpt",
    "forward_feynman_kac",
    "backward_committor",
    "backward_mfpt",
    "backward_feynman_kac",
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
    stat = dgastat_fdm.StationaryDistribution(
        generator, basis, weights, test_basis=test_basis
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
    stat = dgastat_fdm.ForwardFeynmanKac(
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
    stat = dgastat_fdm.BackwardFeynmanKac(
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
