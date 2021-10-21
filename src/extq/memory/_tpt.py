import numba as nb
import numpy as np
import scipy.sparse

from ..stop import backward_stop
from ..stop import forward_stop


def rate(mat, lag):
    n = mat.shape[0] // 2
    ct = mat[:n, n:]
    return (ct[-2, -2] + ct[-2, -1] + ct[-1, -2] + ct[-1, -1]) / lag


def rate_matrix(
    basis, forward_q, backward_q, weights, in_domain, rxn_coord, lag
):
    numer = 0.0
    denom = 0.0
    for x, y, qp, qm, w, d, h in zip(
        basis, basis, forward_q, backward_q, weights, in_domain, rxn_coord
    ):
        assert np.all(w[-lag:] == 0.0)

        t = np.arange(len(d))
        tp = forward_stop(d)
        tm = backward_stop(d)
        iy = np.minimum(tp[:-lag], t[lag:])
        ix = np.maximum(tm[lag:], t[:-lag])
        all_d = np.logical_and(iy == t[lag:], ix == t[:-lag]).astype(float)

        qpdc = np.where(d, 0.0, qp)
        qmdc = np.where(d, 0.0, qm)
        qpd = np.where(d, qp, 0.0)
        qmd = np.where(d, qm, 0.0)

        W = scipy.sparse.diags(w[:-lag])
        Hp = scipy.sparse.diags(w[:-lag] * (h[iy] - h[:-lag]))
        Hm = scipy.sparse.diags(w[:-lag] * (h[lag:] - h[ix]))
        Wd = scipy.sparse.diags(w[:-lag] * all_d)
        Hd = scipy.sparse.diags(w[:-lag] * all_d * (h[lag:] - h[:-lag]))
        wsum = np.sum(w[:-lag])

        # D -> D
        x_y = x[:-lag].T @ Wd @ y[lag:]
        qmd_y = qmd[:-lag].T @ Wd @ y[lag:]
        x_qpd = x[:-lag].T @ Wd @ qpd[lag:]
        x_h_y = x[:-lag].T @ Hd @ y[lag:]
        qmd_h_y = qmd[:-lag].T @ Hd @ y[lag:]
        x_h_qpd = x[:-lag].T @ Hd @ qpd[lag:]
        qmd_h_qpd = qmd[:-lag].T @ Hd @ qpd[lag:]

        # A -> D
        qmdc_y = qmdc[ix].T @ W @ y[lag:]
        qmdc_h_y = qmdc[ix].T @ Hm @ y[lag:]
        qmdc_h_qpd = qmdc[ix].T @ Hm @ qpd[lag:]

        # D -> B
        x_qpdc = x[:-lag].T @ W @ qpdc[iy]
        x_h_qpdc = x[:-lag].T @ Hp @ qpdc[iy]
        qmd_h_qpdc = qmd[:-lag].T @ Hp @ qpdc[iy]

        # A -> B
        qmdc_h_qpdc = _conv(qpdc, qmdc, w, tp, h, lag)

        if scipy.sparse.issparse(x):
            mat = scipy.sparse.bmat(
                [
                    [x_y, None, None]
                    + [x_h_y, x_h_qpd[:, None], x_h_qpdc[:, None]],
                    [qmd_y[None, :], 0, 0]
                    + [qmd_h_y[None, :], qmd_h_qpd, qmd_h_qpdc],
                    [qmdc_y[None, :], 0, wsum]
                    + [qmdc_h_y[None, :], qmdc_h_qpd, qmdc_h_qpdc],
                    [None, None, None]
                    + [x_y, x_qpd[:, None], x_qpdc[:, None]],
                    [None, None, None] + [None, 0, 0],
                    [None, None, None] + [None, 0, wsum],
                ],
                format="csr",
            )
        else:
            nbasis = x.shape[1]
            mat = np.zeros((2 * (nbasis + 2), 2 * (nbasis + 2)))
            amat = mat[: nbasis + 2, : nbasis + 2]
            bmat = mat[nbasis + 2 :, nbasis + 2 :]
            cmat = mat[: nbasis + 2, nbasis + 2 :]
            # upper left
            amat[:nbasis, :nbasis] = x_y
            amat[-2, :nbasis] = qmd_y
            amat[-1, :nbasis] = qmdc_y
            amat[-1, -1] = wsum
            # lower right
            bmat[:nbasis, :nbasis] = x_y
            bmat[:nbasis, -2] = x_qpd
            bmat[:nbasis, -1] = x_qpdc
            bmat[-1, -1] = wsum
            # upper right
            cmat[:nbasis, :nbasis] = x_h_y
            cmat[:nbasis, -2] = x_h_qpd
            cmat[:nbasis, -1] = x_h_qpdc
            cmat[-2, :nbasis] = qmd_h_y
            cmat[-1, :nbasis] = qmdc_h_y
            cmat[-2, -2] = qmd_h_qpd
            cmat[-2, -1] = qmd_h_qpdc
            cmat[-1, -2] = qmdc_h_qpd
            cmat[-1, -1] = qmdc_h_qpdc

        numer += mat
        denom += wsum
    return numer / denom


@nb.njit
def _conv(qp, qm, w, tp, h, lag):
    n = len(tp)
    total = 0.0
    s = tp[0]
    while s < n - 1:
        e = tp[s + 1]
        if e < n:
            c = qm[s] * qp[e] * (h[e] - h[s])
            if c != 0.0:
                wsum = 0.0
                for i in range(max(0, e - lag), min(n - lag, s + 1)):
                    wsum += w[i]
                total += wsum * c
        s = e
    return total
