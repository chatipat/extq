import numba as nb
import numpy as np

from .extended import moving_matmul
from .extended import moving_matmul_Am_B_An
from .extended import moving_matmul_Am_B_Cn


def extended_rate(forward_q, backward_q, weights, transitions, rxn_coord, lag):
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
        frames. Note that indices 0 and n_indices-1 are special. Index 0
        indicates the reactant, and must have no transitions to it from
        any other index. Index n_indices-1 indicates the product, and
        must not have any transitions from it to any other index. Also,
        both indices 0 and n_indices-1 must have a single transition to
        itself.
    rxn_coord : list of (n_indices, n_frames[i]) ndarray of float
        Reaction coordinate at each frame. This must be zero in the
        reactant (index 0) and one in the product (index n_indices-1).
    lag : int
        Lag time in units of frames.

    Returns
    -------
    float
        Estimated TPT rate.

    """
    numer = 0.0
    denom = lag * sum(np.sum(w) for w in weights)
    for qp, qm, w, m, h in zip(
        forward_q, backward_q, weights, transitions, rxn_coord
    ):
        assert np.all(w[-lag:] == 0.0)
        assert np.all(m[0, 0] == 1)
        assert np.all(m[1:, 0] == 0)
        assert np.all(m[-1, -1] == 1)
        assert np.all(m[-1, :-1] == 0)
        assert np.all(h[0, 0] == 0.0)
        assert np.all(h[-1, -1] == 1.0)

        n_frames = w.shape[0]
        n_indices = m.shape[0]
        assert qp.shape == (n_indices, n_frames)
        assert qm.shape == (n_indices, n_frames)
        assert w.shape == (n_frames,)
        assert m.shape == (n_indices, n_indices, n_frames - 1)
        assert h.shape == (n_indices, n_frames)

        numer += _extended_rate_helper(qp, qm, w, m, h, lag)

    return numer / denom


@nb.njit
def _extended_rate_helper(qp, qm, w, m, h, lag):
    n_frames = w.shape[0]
    n_indices = m.shape[0]

    a = np.empty((n_frames - 1, n_indices, n_indices))
    b = np.empty((n_frames - 1, n_indices, n_indices))

    for i in range(n_indices):
        for j in range(n_indices):
            for t in range(n_frames - 1):
                # A is before/after the current time
                a[t, i, j] = m[i, j, t]
                # B is at the current time
                b[t, i, j] = m[i, j, t] * (h[j, t + 1] - h[i, t])

    # shape (n_frames - lag, n_indices, n_indices)
    mm = moving_matmul_Am_B_An(a, b, lag)
    assert len(mm) == n_frames - lag

    # tally contributions to the reaction rate
    result = 0.0
    for i in range(n_indices):
        for j in range(n_indices):
            for t in range(n_frames - lag):
                result += w[t] * qm[i, t] * qp[j, t + lag] * mm[t, i, j]
    return result


def extended_current(forward_q, backward_q, weights, transitions, cv, lag):
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
        frames. Note that indices 0 and n_indices-1 are special. Index 0
        indicates the reactant, and must have no transitions to it from
        any other index. Index n_indices-1 indicates the product, and
        must not have any transitions from it to any other index. Also,
        both indices 0 and n_indices-1 must have a single transition to
        itself.
    cv : list of (n_indices, n_frames[i]) narray of float
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
    for qp, qm, w, m, f in zip(
        forward_q, backward_q, weights, transitions, cv
    ):
        assert np.all(w[-lag:] == 0.0)
        assert np.all(m[0, 0] == 1)
        assert np.all(m[1:, 0] == 0)
        assert np.all(m[-1, -1] == 1)
        assert np.all(m[-1, :-1] == 0)

        n_frames = w.shape[0]
        n_indices = m.shape[0]
        assert qp.shape == (n_indices, n_frames)
        assert qm.shape == (n_indices, n_frames)
        assert w.shape == (n_frames,)
        assert m.shape == (n_indices, n_indices, n_frames - 1)
        assert f.shape == (n_indices, n_frames)

        numer = _extended_current_helper(qm, qp, w, m, f, lag)
        result.append(numer / denom)

    return result


@nb.njit
def _extended_current_helper(qm, qp, w, m, f, lag):
    n_frames = w.shape[0]
    n_indices = m.shape[0]

    # temporary arrays
    a = np.empty((n_frames + lag - 2, n_indices, n_indices))
    b = np.empty((n_frames + lag - 2, n_indices, n_indices))
    c = np.empty((n_frames + lag - 2, n_indices, n_indices))

    for i in range(n_indices):
        for j in range(n_indices):
            # A starts at the current time, and goes to time tau
            for t in range(n_frames - 2):
                a[t, i, j] = m[i, j, t + 1]
            for t in range(n_frames - 2, n_frames + lag - 2):
                a[t, i, j] = 0.0

            # B loops time tau to time zero
            for t in range(lag - 1):
                b[t, i, j] = 0.0
            for t in range(lag - 1, n_frames - 1):
                b[t, i, j] = w[t - lag + 1] * qp[i, t + 1] * qm[j, t - lag + 1]
            for t in range(n_frames - 1, n_frames + lag - 2):
                b[t, i, j] = 0.0

            # C starts at time zero, and goes to the current time
            for t in range(lag):
                c[t, i, j] = 0.0
            for t in range(lag, n_frames + lag - 2):
                c[t, i, j] = m[i, j, t - lag]

    # shape (n_frames - 1, n_indices, n_indices)
    # note that the indices are effectively transposed here:
    # m[t, j, i] has i as the past and j as the future
    mm = moving_matmul_Am_B_Cn(a, b, c, lag)

    # apply transition at current time
    for t in range(n_frames - 1):
        for i in range(n_indices):
            for j in range(n_indices):
                mm[t, j, i] *= m[i, j, t]

    # zero off contributions from before/after the reaction
    for t in range(n_frames - 1):
        mm[t, 0, 0] = 0.0  # before reaction
        mm[t, n_indices - 1, n_indices - 1] = 0.0  # after reaction

    # apply collective variable
    for t in range(n_frames - 1):
        for i in range(n_indices):
            for j in range(n_indices):
                mm[t, j, i] *= f[j, t + 1] - f[i, t]

    # tally contributions to the pointwise reactive current
    result = np.zeros((n_indices, n_frames))
    for i in range(n_indices):
        for j in range(n_indices):
            for t in range(n_frames - 1):
                result[i, t] += mm[t, j, i]  # past
                result[j, t + 1] += mm[t, j, i]  # future
    return result
