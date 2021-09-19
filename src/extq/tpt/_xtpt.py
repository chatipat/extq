import numba as nb
import numpy as np

from ..extended import moving_matmul


def extended_rate(
    forward_q,
    backward_q,
    weights,
    transitions,
    in_domain,
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

    Returns
    -------
    float
        Estimated TPT rate.

    """
    numer = 0.0
    denom = lag * sum(np.sum(w) for w in weights)
    for qp, qm, w, m, d, h in zip(
        forward_q, backward_q, weights, transitions, in_domain, rxn_coord
    ):
        assert np.all(w[-lag:] == 0.0)
        n_frames = w.shape[0]
        n_indices = m.shape[0]
        assert qp.shape == (n_indices, n_frames)
        assert qm.shape == (n_indices, n_frames)
        assert w.shape == (n_frames,)
        assert m.shape == (n_indices, n_indices, n_frames - 1)
        assert d.shape == (n_indices, n_frames)
        assert h.shape == (n_indices, n_frames)
        numer += _extended_rate_helper(qp, qm, w, m, d, h, lag)
    return numer / denom


@nb.njit
def _extended_rate_helper(qp, qm, w, m, d, h, lag):
    nf = len(w)
    ni = len(m)

    # temporary array
    a = np.zeros((nf - 1, 2, ni + 1, 2, ni + 1))

    for t in range(nf - 1):

        # before current time
        for j in range(ni):
            if d[j, t + 1]:
                for i in range(ni):
                    a[t, 0, i, 0, j] = m[i, j, t]
            else:
                a[t, 0, ni, 0, j] = qm[j, t + 1]  # boundary conditions
        a[t, 0, ni, 0, ni] = 1.0

        # at current time
        for i in range(ni):
            for j in range(ni):
                a[t, 0, i, 1, j] = m[i, j, t] * (h[j, t + 1] - h[i, t])

        # after current time
        for i in range(ni):
            if d[i, t]:
                for j in range(ni):
                    a[t, 1, i, 1, j] = m[i, j, t]
            else:
                a[t, 1, i, 1, ni] = qp[i, t]  # boundary conditions
        a[t, 1, ni, 1, ni] = 1.0

    a = a.reshape(nf - 1, 2 * (ni + 1), 2 * (ni + 1))
    a = moving_matmul(a, lag)
    a = a.reshape(nf - lag, 2, (ni + 1), 2, (ni + 1))
    a = a[:, 0, :, 1, :]

    # tally contributions to the reaction rate
    result = 0.0
    for t in range(nf - lag):
        for i in range(ni):
            for j in range(ni):
                result += w[t] * qm[i, t] * a[t, i, j] * qp[j, t + lag]
        for i in range(ni):
            result += w[t] * qm[i, t] * a[t, i, ni]
        for j in range(ni):
            result += w[t] * a[t, ni, j] * qp[j, t + lag]
        result += w[t] * a[t, ni, ni]
    return result


def extended_current(
    forward_q,
    backward_q,
    weights,
    transitions,
    in_domain,
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

    Returns
    -------
    float
        Estimated reactive current at each frame.

    """
    result = []
    denom = lag * sum(np.sum(w) for w in weights)
    for qp, qm, w, m, d, f in zip(
        forward_q, backward_q, weights, transitions, in_domain, cv
    ):
        assert np.all(w[-lag:] == 0.0)
        n_frames = w.shape[0]
        n_indices = m.shape[0]
        assert qp.shape == (n_indices, n_frames)
        assert qm.shape == (n_indices, n_frames)
        assert w.shape == (n_frames,)
        assert m.shape == (n_indices, n_indices, n_frames - 1)
        assert d.shape == (n_indices, n_frames)
        assert f.shape == (n_indices, n_frames)
        numer = _extended_current_helper(qp, qm, w, m, d, f, lag)
        result.append(numer / denom)
    return result


@nb.njit
def _extended_current_helper(qp, qm, w, m, d, f, lag):
    nf = len(w)
    ni = len(m)

    # temporary array
    a = np.zeros((nf + lag - 2, 2, ni + 1, 2, ni + 1))

    # start at the current time and go to time tau
    for t in range(nf - 2):
        for i in range(ni):
            if d[i, t + 1]:
                for j in range(ni):
                    a[t, 0, i, 0, j] = m[i, j, t + 1]
            else:
                a[t, 0, i, 0, ni] = qp[i, t + 1]  # boundary conditions
        a[t, 0, ni, 0, ni] = 1.0

    # loop time tau to time zero
    for t in range(lag - 1, nf - 1):
        for i in range(ni):
            for j in range(ni):
                a[t, 0, i, 1, j] = (
                    w[t - lag + 1] * qp[i, t + 1] * qm[j, t - lag + 1]
                )
        for i in range(ni):
            a[t, 0, i, 1, ni] = w[t - lag + 1] * qp[i, t + 1]
        for j in range(ni):
            a[t, 0, ni, 1, j] = w[t - lag + 1] * qm[j, t - lag + 1]
        a[t, 0, ni, 1, ni] = w[t - lag + 1]

    # start at time zero and go to the current time
    for t in range(lag, nf + lag - 2):
        for j in range(ni):
            if d[j, t - lag + 1]:
                for i in range(ni):
                    a[t, 1, i, 1, j] = m[i, j, t - lag]
            else:
                a[t, 1, ni, 1, j] = qm[j, t - lag + 1]  # boundary conditions
        a[t, 1, ni, 1, ni] = 1.0

    # note that the indices are effectively transposed here:
    # m[t, j, i] has i as the past and j as the future
    a = a.reshape(nf + lag - 2, 2 * (ni + 1), 2 * (ni + 1))
    a = moving_matmul(a, lag)
    a = a.reshape(nf - 1, 2, (ni + 1), 2, (ni + 1))
    a = a[:, 0, :, 1, :]

    # apply transition at current time
    coeffs = np.zeros((nf - 1, ni, ni))
    for t in range(nf - 1):
        for i in range(ni):
            for j in range(ni):
                coeffs[t, i, j] = a[t, j, i] * m[i, j, t]

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
