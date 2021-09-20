import numba as nb
import numpy as np

from ..moving_semigroup import moving_semigroup


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
        numer += _rate_helper(qp, qm, w, d, h, lag)
        denom += np.sum(w)
    return numer / denom


@nb.njit
def _rate_helper(qp, qm, w, d, h, lag):
    n = len(d)
    assert len(qm) == n
    assert len(qp) == n
    assert len(w) == n
    assert len(h) == n

    a = np.empty((n - 1, 8))
    for t in range(n - 1):
        # past
        if d[t + 1]:
            qd = 1.0
            qa = 0.0
        else:
            qd = 0.0
            qa = qm[t + 1]
        # future
        if d[t]:
            dq = 1.0
            bq = 0.0
        else:
            dq = 0.0
            bq = qp[t]
        # present
        ad = db = ab = 0.0
        dd = h[t + 1] - h[t]
        _rate_pack(qd, qa, dq, bq, dd, ad, db, ab, a[t])

    a = moving_semigroup(a, lag, _rate_op)

    total = 0.0
    for t in range(n - lag):
        _, _, _, _, dd, ad, db, ab = _rate_unpack(a[t])
        total += w[t] * (
            qm[t] * dd * qp[t + lag] + ad * qp[t + lag] + qm[t] * db + ab
        )
    return total / lag


@nb.njit
def _rate_op(in1, in2, out):
    qd1, qa1, dq1, bq1, dd1, ad1, db1, ab1 = _rate_unpack(in1)
    qd2, qa2, dq2, bq2, dd2, ad2, db2, ab2 = _rate_unpack(in2)

    # past
    qd = qd1 * qd2
    qa = qa1 * qd2 + qa2
    # future
    dq = dq1 * dq2
    bq = bq1 + dq1 * bq2
    # present
    dd = qd1 * dd2 + dd1 * dq2
    ad = ad2 + qa1 * dd2 + ad1 * dq2
    db = qd1 * db2 + dd1 * bq2 + db1
    ab = ab2 + qa1 * db2 + ad1 * bq2 + ab1

    _rate_pack(qd, qa, dq, bq, dd, ad, db, ab, out)


@nb.njit
def _rate_unpack(inp):
    # past
    qd = inp[0]
    qa = inp[1]
    # future
    dq = inp[2]
    bq = inp[3]
    # present
    dd = inp[4]
    ad = inp[5]
    db = inp[6]
    ab = inp[7]
    return qd, qa, dq, bq, dd, ad, db, ab


@nb.njit
def _rate_pack(qd, qa, dq, bq, dd, ad, db, ab, out):
    # past
    out[0] = qd
    out[1] = qa
    # future
    out[2] = dq
    out[3] = bq
    # present
    out[4] = dd
    out[5] = ad
    out[6] = db
    out[7] = ab


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
        result.append(_current_helper(qp, qm, w, d, f, lag) / denom)
    return result


@nb.njit
def _current_helper(qp, qm, w, d, f, lag):
    n = len(d)
    assert len(qm) == n
    assert len(qp) == n
    assert len(w) == n
    assert len(f) == n

    a = np.empty((n + lag - 2, 10))

    for t in range(n + lag - 2):
        # past
        if t >= lag:
            if d[t - lag + 1]:
                qd = 1.0
                qa = 0.0
            else:
                qd = 0.0
                qa = qm[t - lag + 1]
            qz = 1.0
        else:
            qd = qa = qz = 0.0
        # future
        if t < n - 2:
            if d[t + 1]:
                dq = 1.0
                bq = 0.0
            else:
                dq = 0.0
                bq = qp[t + 1]
            zq = 1.0
        else:
            dq = bq = zq = 0.0
        # present
        if lag - 1 <= t < n - 1:
            dd = qp[t + 1] * w[t - lag + 1] * qm[t - lag + 1]
            ad = qp[t + 1] * w[t - lag + 1]
            db = w[t - lag + 1] * qm[t - lag + 1]
            ab = w[t - lag + 1]
        else:
            dd = ad = db = ab = 0.0
        _current_pack(qd, qa, qz, dq, bq, zq, dd, ad, db, ab, a[t])

    a = moving_semigroup(a, lag, _current_op)

    result = np.zeros(n)
    for t in range(n - 1):
        _, _, _, _, _, _, dd, _, _, _ = _current_unpack(a[t])
        c = dd * (f[t + 1] - f[t]) / lag
        result[t] += 0.5 * c
        result[t + 1] += 0.5 * c
    return result


@nb.njit
def _current_op(in1, in2, out):
    qd1, qa1, qz1, dq1, bq1, zq1, dd1, ad1, db1, ab1 = _current_unpack(in1)
    qd2, qa2, qz2, dq2, bq2, zq2, dd2, ad2, db2, ab2 = _current_unpack(in2)

    # past
    qd = qd1 * qd2
    qa = qa1 * qd2 + qz1 * qa2
    qz = qz1 * qz2
    # future
    dq = dq1 * dq2
    bq = bq1 * zq2 + dq1 * bq2
    zq = zq1 * zq2
    # present
    dd = dq1 * dd2 + bq1 * db2 + ad1 * qa2 + dd1 * qd2
    ad = dq1 * ad2 + bq1 * ab2 + ad1 * qz2
    db = zq1 * db2 + ab1 * qa2 + db1 * qd2
    ab = zq1 * ab2 + ab1 * qz2

    _current_pack(qd, qa, qz, dq, bq, zq, dd, ad, db, ab, out)


@nb.njit
def _current_unpack(inp):
    # past
    qd = inp[0]
    qa = inp[1]
    qz = inp[2]
    # future
    dq = inp[3]
    bq = inp[4]
    zq = inp[5]
    # present
    dd = inp[6]
    ad = inp[7]
    db = inp[8]
    ab = inp[9]
    return qd, qa, qz, dq, bq, zq, dd, ad, db, ab


@nb.njit
def _current_pack(qd, qa, qz, dq, bq, zq, dd, ad, db, ab, out):
    # past
    out[0] = qd
    out[1] = qa
    out[2] = qz
    # future
    out[3] = dq
    out[4] = bq
    out[5] = zq
    # present
    out[6] = dd
    out[7] = ad
    out[8] = db
    out[9] = ab
