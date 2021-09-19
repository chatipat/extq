import numba as nb
import numpy as np

from ..stop import backward_stop
from ..stop import forward_stop


def rate(forward_q, backward_q, weights, in_domain, rxn_coord, lag):
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

    Returns
    -------
    float
        Estimated TPT rate.

    """
    numer = 0.0
    denom = 0.0
    for qp, qm, w, d, h in zip(
        forward_q, backward_q, weights, in_domain, rxn_coord
    ):
        assert np.all(w[-lag:] == 0.0)
        tp = forward_stop(d)
        tm = backward_stop(d)
        numer += _rate_helper(qp, qm, w, tp, tm, h, lag)
        denom += np.sum(w)
    return numer / denom


@nb.njit
def _rate_helper(qp, qm, w, tp, tm, h, lag):
    result = 0.0
    for start in range(len(w) - lag):
        end = start + lag
        total = 0.0
        for i in range(start, end):
            j = i + 1
            ti = max(tm[i], start)
            tj = min(tp[j], end)
            total += qm[ti] * qp[tj] * (h[j] - h[i])
        result += w[start] * total / lag
    return result


def current(forward_q, backward_q, weights, in_domain, cv, lag):
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

    Returns
    -------
    float
        Estimated reactive current at each frame.

    """
    result = []
    denom = sum(np.sum(w) for w in weights)
    for qp, qm, w, d, f in zip(forward_q, backward_q, weights, in_domain, cv):
        assert np.all(w[-lag:] == 0.0)
        tp = forward_stop(d)
        tm = backward_stop(d)
        result.append(_current_helper(qp, qm, w, tp, tm, f, lag) / denom)
    return result


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
