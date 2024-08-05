"""Perturbative DGA"""

from collections.abc import Callable

import numpy as np
import scipy as sp

from . import linalg
from .stop import forward_stop


def forecast_step1(
    trajs: np.ndarray,
    *,
    weights: np.ndarray = None,
    in_domain_fn: Callable[[np.ndarray], np.ndarray],
    lag_p: int,
    n_trajs_p: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perturbative DGA step 1: select initial frames to run trajectories.

    This function takes a dataset of unperturbed trajectories and
    returns initial frames from which to run the perturbed trajectories.
    The output `initial` should be passed to `forecast_step2` as is,
    while each frame of `initial_p` should be mutated to the perturbed
    system (if necessary) and then run for `lag_p` steps.

    Parameters
    ----------
    trajs : (n_trajs, lag + 1, *frame_shape) ndarray
        Unperturbed trajectories.
    weights : (n_trajs,) ndarray of float, optional
        Weight of each trajectory. Default is uniform weights.
    in_domain_fn : callable (..., *frame_shape) ndarray -> (...) ndarray of bool
        Vectorized callable that takes frames from the unperturbed
        system and returns whether each frame is in the domain.
    lag_p : int
        Number of steps to run each perturbed trajectory.
        This function assumes that ``lag % lag_p == 0``.
    n_trajs_p : int
        Number of perturbed trajectories to run.
        This function assumes that ``n_trajs_p % (lag // lag_p) == 0``
        and ``n_trajs_p <= n_trajs * (lag // lag_p)``.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    initial : (n_trajs_p, *frame_shape) ndarray
        Initial frames of the unperturbed trajectories from which the
        initial frames of the perturbed trajectories are taken.
    initial_p : (n_trajs_p, *frame_shape) ndarray
        Initial frames from which to run the perturbed trajectories.
        The shape of each perturbed trajectory should be
        ``(lag_p + 1, *frames_shape_p)``.

    """
    n_trajs = trajs.shape[0]
    lag = trajs.shape[1] - 1
    assert lag % lag_p == 0
    assert n_trajs_p % (lag // lag_p) == 0

    if weights is None:
        weights = np.ones(n_trajs)
    if rng is None:
        rng = np.random.default_rng()

    # stopping time of each trajectory
    stop = np.array([forward_stop(in_domain_fn(traj))[0] for traj in trajs])

    traj_idx = []
    frame_idx = []
    size = n_trajs_p // (lag // lag_p)  # candidates per lag time
    p = weights / np.sum(weights)  # probability of each trajectory
    for t in range(0, lag, lag_p):  # lag times
        # sample an equal number of initial points per lag time
        # sample without replacement to reduce variance
        idx = rng.choice(n_trajs, size=size, p=p, replace=False)
        traj_idx.append(idx)
        frame_idx.append(np.minimum(t, stop[idx]))
    traj_idx = np.concatenate(traj_idx)
    frame_idx = np.concatenate(frame_idx)

    # # probability of each candidate
    # p = np.repeat(weights, lag // lag_p)
    # p /= np.sum(p)
    #
    # # candidates to run
    # idx = rng.choice(n_trajs * (lag // lag_p), size=n_trajs_p, p=p)
    #
    # # index of original trajectory and frame of each candidate
    # traj_idx, frame_idx = np.unravel_index(idx, (n_trajs, lag // lag_p))
    # traj_idx = np.repeat(np.arange(n_trajs), lag // lag_p)
    # frame_idx = np.minimum(frame_idx * lag_p, stop[traj_idx])

    initial = trajs[traj_idx, 0]
    initial_p = trajs[traj_idx, frame_idx]
    return initial, initial_p


def forecast_step2(
    initial: np.ndarray,
    trajs_p: np.ndarray,
    *,
    basis_fn: Callable[[np.ndarray], np.ndarray | sp.sparse.sparray],
    basis_fn_p: Callable[[np.ndarray], np.ndarray | sp.sparse.sparray],
    guess_fn_p: Callable[[np.ndarray], np.ndarray],
    in_domain_fn_p: Callable[[np.ndarray], np.ndarray],
    integrand_fn_p: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> callable:
    """
    Perturbative DGA step 2: calculate forecast from trajectories.

    This function takes a dataset of perturbed trajectories and the
    output `initial` from `forecast_step1`, and returns a forecast
    function.

    Parameters
    ---------
    initial : (n_trajs_p, *frame_shape) ndarray
        Initial frames of the unperturbed trajectories from which the
        initial frames of the perturbed trajectories are taken.
    trajs_p : (n_trajs_p, lag_p + 1, *frame_shape_p) ndarray
        Perturbed trajectories.
    basis_fn : callable (..., *frame_shape) ndarray -> (..., n_basis) {ndarray, sparray} of float
        Vectorized callable that takes frames from the unperturbed
        system and returns the basis evaluated at each frame.
    basis_fn_p : callable (..., *frame_shape_p) ndarray -> (..., n_basis) {ndarray, sparray} of float
        Vectorized callable that takes frames from the perturbed system
        and returns the basis evaluated at each frame. This basis can be
        be different from `basis_fn_p`, but the basis dimensions
        (`n_basis`) must be the same.
    guess_fn_p : callable (..., *frame_shape_p) ndarray -> (...) ndarray of float
        Vectorized callable that takes frames from the perturbed system
        and returns a guess for the forecast at each frame. This guess
        must obey boundary conditions.
    in_domain_fn_p : callable (..., *frame_shape_p) ndarray -> (...) ndarray of bool
        Vectorized callable that takes frames from the perturbed system
        and returns whether each frame is in the domain.
    integrand_fn_p : callable (..., *frame_shape_p) ndarray, (..., *frame_shape_p) ndarray -> (...) ndarray of float
        Vectorized callable that takes pairs of adjacent frames from the
        perturbed system and, for each pair, returns the integral of
        some function from the first frame to the second frame.

    Returns
    -------
    callable (..., *frame_shape_p) ndarray -> (...) ndarray of float
        Vectorized function that takes frames from the perturbed system
        and returns the forecast at each frame.

    """
    assert initial.shape[0] == trajs_p.shape[0]
    lag_p = trajs_p.shape[1] - 1

    stop = []
    r = []
    for traj in trajs_p:
        t = min(lag_p, forward_stop(in_domain_fn_p(traj))[0])
        stop.append(t)
        r.append(np.sum(integrand_fn_p(traj[:t], traj[1 : t + 1])))

    x = basis_fn(initial)
    y0 = basis_fn_p(trajs_p[:, 0])
    y1 = basis_fn(trajs_p[:, stop])
    g0 = guess_fn_p(trajs_p[:, 0])
    g1 = guess_fn_p(trajs_p[:, stop])

    a = x.T @ (y1 - y0)
    b = x.T @ (g1 - g0 + r)
    coef = -linalg.solve(a, b)

    def transform(frame):
        return guess_fn_p(frame) + basis_fn_p(frame) @ coef

    return transform
