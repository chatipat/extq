import numba as nb
import numpy as np
from more_itertools import zip_equal

from ..stop import backward_stop, forward_stop

__all__ = [
    "rate",
    "current",
]


def rate(
    forward_q, backward_q, weights, in_domain, rxn_coord, lag, normalize=True
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
        out += _rate_helper(qp, qm, w, d, h, lag)
    if normalize:
        wsum = sum(np.sum(w) for w in weights)
        out /= wsum
    return out


@nb.njit
def _rate_helper(qp, qm, w, d, h, lag):
    n = len(d)
    tp = forward_stop(d)
    tm = backward_stop(d)

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


def current(
    forward_q, backward_q, weights, in_domain, cv, lag, normalize=True
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
    float
        Estimated reactive current at each frame.

    """
    assert lag > 0
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
        if n_frames <= lag:
            j = np.zeros(len(w))
        else:
            tp = forward_stop(d)
            tm = backward_stop(d)
            j = _current_helper(qp, qm, w, tp, tm, f, lag)
        out.append(j)
    if normalize:
        wsum = sum(np.sum(w) for w in weights)
        for j in out:
            j /= wsum
    return out


@nb.njit
def _current_helper(qp, qm, w, tp, tm, f, lag):
    result = np.zeros(len(w))
    for start in range(len(w) - lag):
        end = start + lag
        for i in range(start, end):
            j = i + 1
            ti = max(tm[i], start)
            tj = min(tp[j], end)
            c = w[start] * qm[ti] * qp[tj] * (f[j] - f[i]) / lag
            result[i] += 0.5 * c
            result[j] += 0.5 * c
    return result
