import numpy as np
from more_itertools import zip_equal

from .integral import integral_coeffs, integral_windows
from .utils import normalize_weights

__all__ = [
    "extended_rate",
    "extended_density",
    "extended_current",
]


def extended_rate(
    forward_q,
    backward_q,
    weights,
    transitions,
    in_domain,
    rxn_coord,
    lag,
    *,
    normalize=True,
):
    """Estimate the TPT rate with extended committors.

    Parameters
    ----------
    forward_q : list of (n_indices, n_frames[i]) ndarray of float
        Forward extended committor for each frame.
    backward_q : list of (n_indices, n_frames[i]) ndarray of float
        Backward extended committor for each frame.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames.
    in_domain : list of (n_indices, n_frames[i]) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    rxn_coord : list of (n_indices, n_frames[i]) ndarray of float
        Reaction coordinate at each frame. Must obey boundary
        conditions.
    lag : int
        Lag time in units of frames.
    normalize : bool, optional
        If True (default), normalize `weights` to one.

    Returns
    -------
    float
        Estimated TPT rate.

    """
    assert lag > 0
    if normalize:
        weights = normalize_weights(weights)
    n_indices = None
    out = 0.0
    for qp, qm, w, m, d, h in zip_equal(
        forward_q, backward_q, weights, transitions, in_domain, rxn_coord
    ):
        n_frames = w.shape[0]
        n_indices = m.shape[0] if n_indices is None else n_indices
        assert qp.shape == (n_indices, n_frames)
        assert qm.shape == (n_indices, n_frames)
        assert w.shape == (n_frames,)
        assert m.shape == (n_indices, n_indices, n_frames - 1)
        assert d.shape == (n_indices, n_frames)
        assert h.shape == (n_indices, n_frames)
        assert np.all(w[max(0, n_frames - lag) :] == 0.0)
        if n_frames <= lag:
            continue
        u = _extended_committor_outer_kernel(qp, qm, w, lag)
        kl = _extended_backward_committor_kernel(qm, m, d)
        kr = _extended_forward_committor_kernel(qp, m, d)
        obs = np.zeros((n_indices + 1, n_indices + 1, n_frames - 1))
        obs[:-1, :-1] = m * (h[None, :, 1:] - h[:, None, :-1])
        c = _integral_windows(kl, kr, obs, lag)
        out += np.sum(c * u) / lag
    return out


def extended_density(
    forward_q,
    backward_q,
    weights,
    transitions,
    in_domain,
    lag,
    *,
    normalize=True,
):
    """Estimate the reactive density with extended committors.

    Parameters
    ----------
    forward_q : list of (n_indices, n_frames[i]) ndarray of float
        Forward extended committor for each frame.
    backward_q : list of (n_indices, n_frames[i]) ndarray of float
        Backward extended committor for each frame.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames.
    in_domain : list of (n_indices, n_frames[i]) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    lag : int
        Lag time in units of frames.
    normalize : bool, optional
        If True (default), normalize `weights` to one.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimated reactive density at each frame.

    """
    assert lag > 0
    if normalize:
        weights = normalize_weights(weights)
    n_indices = None
    out = []
    for qp, qm, w, m, d in zip_equal(
        forward_q, backward_q, weights, transitions, in_domain
    ):
        n_frames = w.shape[0]
        n_indices = m.shape[0] if n_indices is None else n_indices
        assert qp.shape == (n_indices, n_frames)
        assert qm.shape == (n_indices, n_frames)
        assert w.shape == (n_frames,)
        assert m.shape == (n_indices, n_indices, n_frames - 1)
        assert d.shape == (n_indices, n_frames)
        assert np.all(w[max(0, n_frames - lag) :] == 0.0)
        p = np.zeros((n_indices, n_frames))
        if n_frames > lag:
            sw = _step_weights(qp, qm, w, m, d, lag)
            c = 0.5 * sw / lag
            p[:, :-1] += np.sum(c, axis=1)
            p[:, 1:] += np.sum(c, axis=0)
        out.append(p)
    return out


def extended_current(
    forward_q,
    backward_q,
    weights,
    transitions,
    in_domain,
    cv,
    lag,
    *,
    normalize=True,
):
    """Estimate the reactive current with extended committors.

    Parameters
    ----------
    forward_q : list of (n_indices, n_frames[i]) ndarray of float
        Forward extended committor for each frame.
    backward_q : list of (n_indices, n_frames[i]) ndarray of float
        Backward extended committor for each frame.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames.
    in_domain : list of (n_indices, n_frames[i]) ndarray of bool
        For each value of the index process, whether each frame of the
        trajectories is in the domain.
    cv : list of (n_indices, n_frames[i]) narray of float
        Collective variable at each frame.
    lag : int
        Lag time in units of frames.
    normalize : bool, optional
        If True (default), normalize `weights` to one.

    Returns
    -------
    list of (n_indices, n_frames[i]) ndarray of float
        Estimated reactive current at each frame.

    """
    assert lag > 0
    if normalize:
        weights = normalize_weights(weights)
    n_indices = None
    out = []
    for qp, qm, w, m, d, f in zip_equal(
        forward_q, backward_q, weights, transitions, in_domain, cv
    ):
        n_frames = w.shape[0]
        n_indices = m.shape[0] if n_indices is None else n_indices
        assert qp.shape == (n_indices, n_frames)
        assert qm.shape == (n_indices, n_frames)
        assert w.shape == (n_frames,)
        assert m.shape == (n_indices, n_indices, n_frames - 1)
        assert d.shape == (n_indices, n_frames)
        assert f.shape == (n_indices, n_frames)
        assert np.all(w[max(0, n_frames - lag) :] == 0.0)
        j = np.zeros((n_indices, n_frames))
        if n_frames > lag:
            sw = _step_weights(qp, qm, w, m, d, lag)
            c = sw * (f[None, :, 1:] - f[:, None, :-1]) / lag
            j[:, :-1] += 0.5 * np.sum(c, axis=1)
            j[:, 1:] += 0.5 * np.sum(c, axis=0)
        out.append(j)
    return out


def _step_weights(qp, qm, w, m, d, lag):
    n_indices, n_frames = d.shape
    u = _extended_committor_outer_kernel(qp, qm, w, lag)
    kl = _extended_backward_committor_kernel(qm, m, d)
    kr = _extended_forward_committor_kernel(qp, m, d)
    c = _integral_coeffs(u, kl, kr, lag)
    return c[:-1, :-1] * m


def _integral_windows(kl, kr, obs, lag):
    kl = np.moveaxis(kl, -1, 0)
    kr = np.moveaxis(kr, -1, 0)
    obs = np.moveaxis(obs, -1, 0)
    c = integral_windows(kl, kr, obs, 1, lag)
    c = np.moveaxis(c, 0, -1)
    return c


def _integral_coeffs(u, kl, kr, lag):
    u = np.moveaxis(u, -1, 0)
    kl = np.moveaxis(kl, -1, 0)
    kr = np.moveaxis(kr, -1, 0)
    c = integral_coeffs(u, kl, kr, 1, lag)
    c = np.moveaxis(c, 0, -1)
    return c


def _extended_forward_committor_kernel(qp, m, d):
    n_indices, n_frames = d.shape
    k = np.zeros((n_indices + 1, n_indices + 1, n_frames - 1))
    k[:-1, :-1] = np.where(d[:, None, :-1], m, 0)
    k[:-1, -1] = np.where(d[:, :-1], 0, qp[:, :-1])
    k[-1, -1] = 1
    return k


def _extended_backward_committor_kernel(qm, m, d):
    n_indices, n_frames = d.shape
    k = np.zeros((n_indices + 1, n_indices + 1, n_frames - 1))
    k[:-1, :-1] = np.where(d[None, :, 1:], m, 0)
    k[-1, :-1] = np.where(d[:, 1:], 0, qm[:, 1:])
    k[-1, -1] = 1
    return k


def _extended_committor_outer_kernel(qp, qm, w, lag):
    (n_frames,) = w.shape
    xqp = np.concatenate([qp, np.ones((1, n_frames))], axis=0)
    xqm = np.concatenate([qm, np.ones((1, n_frames))], axis=0)
    u = w[:-lag] * xqm[:, None, :-lag] * xqp[None, :, lag:]
    return u
