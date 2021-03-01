import numpy as np

from .extended import moving_matmul


def extended_rate(forward_q, backward_q, weights, transitions, rxn_coord, lag):
    """Estimate the TPT rate with extended committors.

    Parameters
    ----------
    forward_q : list of (n_frames[i], n_indices) ndarray of float
        Forward extended committor for each frame.
    backward_q : list of (n_frames[i], n_indices) ndarray of float
        Backward extended committor for each frame.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_frames[i]-1, n_indices, n_indices) ndarray
        Possible transitions of the index process between adjacent
        frames. Note that indices 0 and n_indices-1 are special. Index 0
        indicates the reactant, and must have no transitions to it from
        any other index. Index n_indices-1 indicates the product, and
        must not have any transitions from it to any other index. Also,
        both indices 0 and n_indices-1 must have a single transition to
        itself.
    rxn_coord : list of (n_frames[i], n_indices) ndarray of float
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
        assert np.all(m[:, 0, 0] == 1)
        assert np.all(m[:, 1:, 0] == 0)
        assert np.all(m[:, -1, -1] == 1)
        assert np.all(m[:, -1, :-1] == 0)

        n_transitions, n_indices, _ = m.shape

        a = np.zeros((n_transitions, 2, n_indices, 2, n_indices))
        a[:, 0, :, 0, :] = m
        a[:, 1, :, 1, :] = m
        a[:, 0, :, 1, :] = m * (h[1:, None, :] - h[:-1, :, None])
        a[:, 0, -1, 1, :] = 0.0
        a[:, 0, :, 1, 0] = 0.0
        a = a.reshape(n_transitions, 2 * n_indices, 2 * n_indices)
        a = moving_matmul(a, lag)
        a = a.reshape(n_transitions + 1 - lag, 2, n_indices, 2, n_indices)
        a = a[:, 0, :, 1, :]

        numer += np.einsum("n,ni,nj,nij->", w[:-lag], qm[:-lag], qp[lag:], a)

    return numer / denom


def extended_current(forward_q, backward_q, weights, transitions, cv, lag):
    """Estimate the reactive current with extended committors.

    Parameters
    ----------
    forward_q : list of (n_frames[i], n_indices) ndarray of float
        Forward extended committor for each frame.
    backward_q : list of (n_frames[i], n_indices) ndarray of float
        Backward extended committor for each frame.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    transitions : list of (n_frames[i]-1, n_indices, n_indices) ndarray
        Possible transitions of the index process between adjacent
        frames. Note that indices 0 and n_indices-1 are special. Index 0
        indicates the reactant, and must have no transitions to it from
        any other index. Index n_indices-1 indicates the product, and
        must not have any transitions from it to any other index. Also,
        both indices 0 and n_indices-1 must have a single transition to
        itself.
    cv : list of (n_frames[i], n_indices) narray of float
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
        assert np.all(m[:, 0, 0] == 1)
        assert np.all(m[:, 1:, 0] == 0)
        assert np.all(m[:, -1, -1] == 1)
        assert np.all(m[:, -1, :-1] == 0)

        n_transitions, n_indices, _ = m.shape

        a = np.zeros((n_transitions + lag - 1, 2, n_indices, 2, n_indices))
        a[:-lag, 0, :, 0, :] = m[1:]
        a[lag:, 1, :, 1, :] = m[:-1]
        a[lag - 1 : -(lag - 1), 0, :, 1, :] = np.einsum(
            "n,ni,nj->nij", w[:-lag], qp[lag:], qm[:-lag]
        )
        a = a.reshape(n_transitions + lag - 1, 2 * n_indices, 2 * n_indices)
        a = moving_matmul(a, lag)
        a = a.reshape(n_transitions, 2, n_indices, 2, n_indices)
        a = a[:, 0, :, 1, :].swapaxes(1, 2)
        a[:, -1, :] = 0.0
        a[:, :, 0] = 0.0
        a *= m * (f[1:, None, :] - f[:-1, :, None])

        numer = np.zeros((n_transitions + 1, n_indices))
        numer[:-1, :] += np.sum(a, axis=-1)
        numer[1:, :] += np.sum(a, axis=-2)
        result.append(numer / denom)

    return result
