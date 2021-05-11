import numba as nb
import numpy as np

from .extended import moving_matmul


def extended_rate(
    forward_q,
    backward_q,
    weights,
    n_domain_indices,
    transitions,
    rxn_coord,
    lag,
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
    n_domain_indices : int
        Number of indices inside the domain.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames. Indices inside the domain must precede indices outside
        of the domain.
    rxn_coord : list of (n_indices, n_frames[i]) ndarray of float
        Reaction coordinate at each frame. Must obey boundary
        conditions.
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
        n_frames = w.shape[0]
        n_indices = m.shape[0]
        assert qp.shape == (n_indices, n_frames)
        assert qm.shape == (n_indices, n_frames)
        assert w.shape == (n_frames,)
        assert m.shape == (n_indices, n_indices, n_frames - 1)
        assert h.shape == (n_indices, n_frames)
        numer += _extended_rate_helper(qp, qm, w, n_domain_indices, m, h, lag)
    return numer / denom


@nb.njit
def _extended_rate_helper(qp, qm, w, nd, m, h, lag):
    nf = len(w)
    ni = len(m)
    assert nd <= ni

    # temporary array
    a = np.zeros((nf - 1, 2, nd + 1, 2, nd + 1))

    for t in range(nf - 1):

        # before current time
        for i in range(nd):
            for j in range(nd):
                a[t, 0, i, 0, j] = m[i, j, t]
        for i in range(nd, ni):
            for j in range(nd):
                a[t, 0, nd, 0, j] += qm[i, t + 0] * m[i, j, t]
        a[t, 0, nd, 0, nd] = 1.0

        # at current time
        for i in range(nd):
            for j in range(nd):
                a[t, 0, i, 1, j] = m[i, j, t] * (h[j, t + 1] - h[i, t])
        for i in range(nd):
            for j in range(nd, ni):
                a[t, 0, i, 1, nd] += (
                    m[i, j, t] * qp[j, t + 1] * (h[j, t + 1] - h[i, t])
                )
        for i in range(nd, ni):
            for j in range(nd):
                a[t, 0, nd, 1, j] += (
                    qm[i, t + 0] * m[i, j, t] * (h[j, t + 1] - h[i, t])
                )
        for i in range(nd, ni):
            for j in range(nd, ni):
                a[t, 0, nd, 1, nd] += (
                    qm[i, t + 0]
                    * m[i, j, t]
                    * qp[j, t + 1]
                    * (h[j, t + 1] - h[i, t])
                )

        # after current time
        for i in range(nd):
            for j in range(nd):
                a[t, 1, i, 1, j] = m[i, j, t]
        for i in range(nd):
            for j in range(nd, ni):
                a[t, 1, i, 1, nd] += m[i, j, t] * qp[j, t + 1]
        a[t, 1, nd, 1, nd] = 1.0

    a = a.reshape(nf - 1, 2 * (nd + 1), 2 * (nd + 1))
    a = moving_matmul(a, lag)
    a = a.reshape(nf - lag, 2, (nd + 1), 2, (nd + 1))
    a = a[:, 0, :, 1, :]

    # tally contributions to the reaction rate
    result = 0.0
    for t in range(nf - lag):
        for i in range(nd):
            for j in range(nd):
                result += w[t] * qm[i, t] * a[t, i, j] * qp[j, t + lag]
        for i in range(nd):
            result += w[t] * qm[i, t] * a[t, i, nd]
        for j in range(nd):
            result += w[t] * a[t, nd, j] * qp[j, t + lag]
        result += w[t] * a[t, nd, nd]
    return result


def extended_current(
    forward_q,
    backward_q,
    weights,
    n_domain_indices,
    transitions,
    cv,
    lag,
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
    n_domain_indices : int
        Number of indices inside the domain.
    transitions : list of (n_indices, n_indices, n_frames[i]-1) ndarray
        Possible transitions of the index process between adjacent
        frames. Indices inside the domain must precede indices outside
        of the domain.
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
    denom = lag * sum(np.sum(w) for w in weights)
    for qp, qm, w, m, f in zip(
        forward_q, backward_q, weights, transitions, cv
    ):
        assert np.all(w[-lag:] == 0.0)
        n_frames = w.shape[0]
        n_indices = m.shape[0]
        assert qp.shape == (n_indices, n_frames)
        assert qm.shape == (n_indices, n_frames)
        assert w.shape == (n_frames,)
        assert m.shape == (n_indices, n_indices, n_frames - 1)
        assert f.shape == (n_indices, n_frames)
        numer = _extended_current_helper(
            qm, qp, w, n_domain_indices, m, f, lag
        )
        result.append(numer / denom)
    return result


@nb.njit
def _extended_current_helper(qm, qp, w, nd, m, f, lag):
    nf = len(w)
    ni = len(m)
    assert nd <= ni

    # temporary array
    a = np.zeros((nf + lag - 2, 2, nd + 1, 2, nd + 1))

    # start at the current time and go to time tau
    for t in range(nf - 2):
        for i in range(nd):
            for j in range(nd):
                a[t, 0, i, 0, j] = m[i, j, t + 1]
        for i in range(nd):
            for j in range(nd, ni):
                a[t, 0, i, 0, nd] += m[i, j, t + 1] * qp[j, t + 2]
        a[t, 0, nd, 0, nd] = 1.0

    # loop time tau to time zero
    for t in range(lag - 1, nf - 1):
        for i in range(nd):
            for j in range(nd):
                a[t, 0, i, 1, j] = (
                    w[t - lag + 1] * qp[i, t + 1] * qm[j, t - lag + 1]
                )
        for i in range(nd):
            a[t, 0, i, 1, nd] += w[t - lag + 1] * qp[i, t + 1]
        for j in range(nd):
            a[t, 0, nd, 1, j] += w[t - lag + 1] * qm[j, t - lag + 1]
        a[t, 0, nd, 1, nd] = w[t - lag + 1]

    # start at time zero and go to the current time
    for t in range(lag, nf + lag - 2):
        for i in range(nd):
            for j in range(nd):
                a[t, 1, i, 1, j] = m[i, j, t - lag]
        for i in range(nd, ni):
            for j in range(nd):
                a[t, 1, nd, 1, j] += qm[i, t - lag] * m[i, j, t - lag]
        a[t, 1, nd, 1, nd] = 1.0

    # note that the indices are effectively transposed here:
    # m[t, j, i] has i as the past and j as the future
    a = a.reshape(nf + lag - 2, 2 * (nd + 1), 2 * (nd + 1))
    a = moving_matmul(a, lag)
    a = a.reshape(nf - 1, 2, (nd + 1), 2, (nd + 1))
    a = a[:, 0, :, 1, :]

    # apply transition at current time
    coeffs = np.zeros((nf - 1, ni, ni))
    for t in range(nf - 1):
        for i in range(nd):
            for j in range(nd):
                coeffs[t, i, j] = a[t, j, i] * m[i, j, t]
        for i in range(nd):
            for j in range(nd, ni):
                coeffs[t, i, j] = a[t, nd, i] * m[i, j, t] * qp[j, t + 1]
        for i in range(nd, ni):
            for j in range(nd):
                coeffs[t, i, j] = a[t, j, nd] * qm[i, t + 0] * m[i, j, t]
        for i in range(nd, ni):
            for j in range(nd, ni):
                coeffs[t, i, j] = (
                    a[t, nd, nd] * qm[i, t + 0] * m[i, j, t] * qp[j, t + 1]
                )

    # tally contributions to the pointwise reactive current
    result = np.zeros((ni, nf))
    for i in range(ni):
        for j in range(ni):
            for t in range(nf - 1):
                # apply collective variable at current time
                c = coeffs[t, i, j] * (f[j, t + 1] - f[i, t])
                # split contribution symmetrically
                result[i, t] += 0.5 * c  # past
                result[j, t + 1] += 0.5 * c  # future
    return result
