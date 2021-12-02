import numpy as np
import scipy.sparse

from ..dga._utils import solve
from ..dga._utils import transform
from ..stop import backward_stop
from ..stop import forward_stop


def forward_committor_coeffs(mt, m0):
    m = mt - m0
    return solve(m[:-2, :-2], -(m[:-2, -2] + m[:-2, -1]))


def backward_committor_coeffs(mt, m0):
    m = mt.T - m0.T
    return solve(m[:-2, :-2], -(m[:-2, -2] + m[:-2, -1]))


def committor_transform(coeffs, basis, guess):
    return transform(coeffs, basis, guess)


def forward_committor_matrix(
    basis, weights, in_domain, guess, lag, test_basis=None
):
    if test_basis is None:
        test_basis = basis
    numer = 0.0
    denom = 0.0
    for x, y, w, d, g in zip(test_basis, basis, weights, in_domain, guess):
        assert np.all(w[len(w) - lag :] == 0.0)
        last = -lag if lag > 0 else None
        iy = np.minimum(np.arange(lag, len(d)), forward_stop(d)[:last])
        gd = np.where(d, g, 0.0)
        gdc = np.where(d, 0.0, g)
        wx = scipy.sparse.diags(w[:last]) @ x[:last]
        wsum = np.sum(w[:last])

        if scipy.sparse.issparse(x):
            mat = scipy.sparse.bmat(
                [
                    [wx.T @ y[iy], wx.T @ gd[iy, None], wx.T @ gdc[iy, None]],
                    [None, 0, 0],
                    [None, 0, wsum],
                ],
                format="csr",
            )
        else:
            nbasis = x.shape[1]
            mat = np.zeros((nbasis + 2, nbasis + 2))
            mat[:nbasis, :nbasis] = wx.T @ y[iy]
            mat[:nbasis, -2] = wx.T @ gd[iy]
            mat[:nbasis, -1] = wx.T @ gdc[iy]
            mat[-1, -1] = wsum

        numer += mat
        denom += wsum
    return numer / denom


def backward_committor_matrix(
    basis, weights, in_domain, guess, lag, test_basis=None
):
    if test_basis is None:
        test_basis = basis
    numer = 0.0
    denom = 0.0
    for x, y, w, d, g in zip(test_basis, basis, weights, in_domain, guess):
        assert np.all(w[len(w) - lag :] == 0.0)
        last = -lag if lag > 0 else None
        iy = np.maximum(np.arange(len(d) - lag), backward_stop(d)[lag:])
        gd = np.where(d, g, 0.0)
        gdc = np.where(d, 0.0, g)
        wx = scipy.sparse.diags(w[:last]) @ x[lag:]
        wsum = np.sum(w[:last])

        if scipy.sparse.issparse(x):
            mat = scipy.sparse.bmat(
                [
                    [wx.T @ y[iy], wx.T @ gd[iy, None], wx.T @ gdc[iy, None]],
                    [None, 0, 0],
                    [None, 0, wsum],
                ],
                format="csr",
            )
        else:
            nbasis = x.shape[1]
            mat = np.zeros((nbasis + 2, nbasis + 2))
            mat[:nbasis, :nbasis] = wx.T @ y[iy]
            mat[:nbasis, -2] = wx.T @ gd[iy]
            mat[:nbasis, -1] = wx.T @ gdc[iy]
            mat[-1, -1] = wsum

        numer += mat.T
        denom += wsum
    return numer / denom
