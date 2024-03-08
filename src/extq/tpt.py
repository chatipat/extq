import numba as nb
import numpy as np
from more_itertools import zip_equal

from .stop import backward_stop, forward_stop
from .utils import normalize_weights

__all__ = [
    "rate",
    "density",
    "current",
    "rate_jstrahan",
    "current_jstrahan",
]


def rate(
    forward_q,
    backward_q,
    weights,
    in_domain,
    rxn_coord,
    lag,
    *,
    normalize=True,
):
    """Estimate the TPT rate.

    Parameters
    ----------
    forward_q : list of (n_frames[i],) ndarray of float
        Forward committor for each frame.
    backward_q : list of (n_frames[i],) ndarray of float
        Backward committor for each frame.
    weights : list of (n_frames[i],) ndarray of float
        Reweighting factor to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    rxn_coord : list of (n_frames[i],) ndarray of float
        Reaction coordinate at each frame. This must be zero in the
        reactant state and one in the product state.
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
    out = 0.0
    for qp, qm, w, d, h in zip_equal(
        forward_q, backward_q, weights, in_domain, rxn_coord
    ):
        n_frames = w.shape[0]
        assert qp.shape == (n_frames,)
        assert qm.shape == (n_frames,)
        assert w.shape == (n_frames,)
        assert d.shape == (n_frames,)
        assert h.shape == (n_frames,)
        assert np.all(w[max(0, n_frames - lag) :] == 0.0)
        if n_frames <= lag:
            continue
        tp = forward_stop(d)
        tm = backward_stop(d)
        out += _rate_helper(qp, qm, w, d, tp, tm, h, lag)
    return out


@nb.njit
def _rate_helper(qp, qm, w, d, tp, tm, h, lag):
    n = len(d)

    total = 0.0

    # fragments of reactive trajectories
    for s in range(n - lag):
        e = s + lag
        all_d = tp[s] > e
        # assert all_d == (tm[e] < s)
        if all_d:
            total += w[s] * qm[s] * qp[e] * (h[e] - h[s])  # D -> D
        else:
            if d[s]:
                total += w[s] * qm[s] * qp[tp[s]] * (h[tp[s]] - h[s])  # D -> B
            if d[e]:
                total += w[s] * qm[tm[e]] * qp[e] * (h[e] - h[tm[e]])  # A -> D

    # complete reactive trajectories
    for s in range(n - 1):
        if not d[s]:
            e = tp[s + 1]
            if e < n and s >= e - lag and qm[s] != 0.0 and qp[e] != 0.0:
                wsum = 0.0
                for i in range(max(0, e - lag), min(n - lag, s + 1)):
                    wsum += w[i]
                if wsum != 0.0:
                    total += wsum * qm[s] * qp[e] * (h[e] - h[s])  # A -> B

    return total / lag


def density(forward_q, backward_q, weights, in_domain, lag, *, normalize=True):
    """Estimate the reactive density at each frame.

    Parameters
    ----------
    forward_q : list of (n_frames[i],) ndarray of float
        Forward committor for each frame.
    backward_q : list of (n_frames[i],) ndarray of float
        Backward committor for each frame.
    weights : list of (n_frames[i],) ndarray of float
        Reweighting factor to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    lag : int
        Lag time in units of frames.
    normalize : bool, optional
        If True (default), normalize `weights` to one.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimated reactive density at each frame.

    """
    assert lag > 0
    if normalize:
        weights = normalize_weights(weights)
    out = []
    for qp, qm, w, d in zip_equal(forward_q, backward_q, weights, in_domain):
        n_frames = w.shape[0]
        assert qp.shape == (n_frames,)
        assert qm.shape == (n_frames,)
        assert w.shape == (n_frames,)
        assert d.shape == (n_frames,)
        assert np.all(w[max(0, n_frames - lag) :] == 0.0)
        p = np.zeros(len(w))
        if n_frames > lag:
            tp = forward_stop(d)
            tm = backward_stop(d)
            sw = _step_weights(qp, qm, w, tp, tm, lag)
            p[:-1] += 0.5 * sw
            p[1:] += 0.5 * sw
        out.append(p)
    return out


def current(
    forward_q, backward_q, weights, in_domain, cv, lag, *, normalize=True
):
    """Estimate the reactive current at each frame.

    Parameters
    ----------
    forward_q : list of (n_frames[i],) ndarray of float
        Forward committor for each frame.
    backward_q : list of (n_frames[i],) ndarray of float
        Backward committor for each frame.
    weights : list of (n_frames[i],) ndarray of float
        Reweighting factor to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    cv : list of (n_frames[i],) narray of float
        Collective variable at each frame.
    lag : int
        Lag time in units of frames.
    normalize : bool, optional
        If True (default), normalize `weights` to one.

    Returns
    -------
    list of (n_frames[i],) ndarray of float
        Estimated reactive current at each frame.

    """
    assert lag > 0
    if normalize:
        weights = normalize_weights(weights)
    out = []
    for qp, qm, w, d, f in zip_equal(
        forward_q, backward_q, weights, in_domain, cv
    ):
        n_frames = w.shape[0]
        assert qp.shape == (n_frames,)
        assert qm.shape == (n_frames,)
        assert w.shape == (n_frames,)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames,)
        assert np.all(w[max(0, n_frames - lag) :] == 0.0)
        j = np.zeros(len(w))
        if n_frames > lag:
            tp = forward_stop(d)
            tm = backward_stop(d)
            sw = _step_weights(qp, qm, w, tp, tm, lag)
            c = sw * np.diff(f)
            j[:-1] += 0.5 * c
            j[1:] += 0.5 * c
        out.append(j)
    return out


@nb.njit
def _step_weights(qp, qm, w, tp, tm, lag):
    out = np.zeros(len(w) - 1)
    for start in range(len(w) - lag):  # loop over sliding windows
        end = start + lag
        for i in range(start, end):  # loop over steps in each window
            j = i + 1
            ti = max(tm[i], start)
            tj = min(tp[j], end)
            c = w[start] * qm[ti] * qp[tj] / lag
            out[i] += c
    return out


def rate_jstrahan(forward_q, backward_q, weights, in_domain, lag):
    """Estimate the TPT rate using John Strahan's estimator.

    Parameters
    ----------
    forward_q : list of (n_frames[i],) ndarray of float
        Forward committor for each frame.
    backward_q : list of (n_frames[i],) ndarray of float
        Backward committor for each frame.
    weights : list of (n_frames[i],) ndarray of float
        Reweighting factor to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    lag : int
        Lag time in units of frames.

    Returns
    -------
    float
        Estimated TPT rate.

    """
    numer = 0.0
    denom = lag * sum(np.sum(w) for w in weights)
    for qp, qm, w, d in zip_equal(forward_q, backward_q, weights, in_domain):
        assert np.all(w[-lag:] == 0.0)
        tp = np.minimum(np.arange(lag, len(w)), forward_stop(d)[:-lag])
        numer += np.sum(w[:-lag] * qm[:-lag] * qp[tp] * (qp[tp] - qp[:-lag]))
    return numer / denom


def current_jstrahan(forward_q, backward_q, weights, in_domain, cv, lag):
    """Estimate the reactive current using John Strahan's estimator.

    Parameters
    ----------
    forward_q : list of (n_frames[i],) ndarray of float
        Forward committor for each frame.
    backward_q : list of (n_frames[i],) ndarray of float
        Backward committor for each frame.
    weights : list of (n_frames[i],) ndarray of float
        Reweighting factor to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    cv : list of (n_frames[i],) narray of float
        Collective variable at each frame.
    lag : int
        Lag time in units of frames.

    Returns
    -------
    float
        Estimated reactive current at each frame.

    """
    result = []
    denom = 2.0 * lag * sum(np.sum(w) for w in weights)
    for qp, qm, w, d, f in zip_equal(
        forward_q, backward_q, weights, in_domain, cv
    ):
        assert np.all(w[-lag:] == 0.0)
        tp = np.minimum(np.arange(lag, len(w)), forward_stop(d)[:-lag])
        tm = np.maximum(np.arange(len(w) - lag), backward_stop(d)[lag:])
        numer = np.zeros(len(w))
        numer[:-lag] += w[:-lag] * qm[:-lag] * qp[tp] * (f[tp] - f[:-lag])
        numer[lag:] += w[:-lag] * qm[tm] * qp[lag:] * (f[lag:] - f[tm])
        result.append(numer / denom)
    return result
