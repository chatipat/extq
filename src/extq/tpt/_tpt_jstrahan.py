import numpy as np

from ..stop import backward_stop
from ..stop import forward_stop


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
    for qp, qm, w, d in zip(forward_q, backward_q, weights, in_domain):
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
    for qp, qm, w, d, f in zip(forward_q, backward_q, weights, in_domain, cv):
        assert np.all(w[-lag:] == 0.0)
        tp = np.minimum(np.arange(lag, len(w)), forward_stop(d)[:-lag])
        tm = np.maximum(np.arange(len(w) - lag), backward_stop(d)[lag:])
        numer = np.zeros(len(w))
        numer[:-lag] += w[:-lag] * qm[:-lag] * qp[tp] * (f[tp] - f[:-lag])
        numer[lag:] += w[:-lag] * qm[tm] * qp[lag:] * (f[lag:] - f[tm])
        result.append(numer / denom)
    return result
