import numba as nb
import numpy as np
import scipy.sparse


def wtlcc_dense(w, x, y, z=None, lag=0):
    if scipy.sparse.issparse(x):
        if scipy.sparse.issparse(y):
            return wtlcc_sparse_sparse_dense(w, x, y, z, lag)
        else:
            return wtlcc_sparse_dense_dense(w, x, y, z, lag)
    else:
        if scipy.sparse.issparse(y):
            return wtlcc_dense_sparse_dense(w, x, y, z, lag)
        else:
            return wtlcc_dense_dense_dense(w, x, y, z, lag)


def wtlcc_dense_dense_dense(w, x, y, z=None, lag=0):
    assert lag >= 0
    last = -lag if lag > 0 else None
    if x is None:
        if y is None:
            xwy = np.sum(w)[None, None]
        else:
            xwy = (w @ y[lag:])[None, :]
    else:
        if y is None:
            xwy = (w @ x[:last])[:, None]
        else:
            xwy = x[:last].T @ (w[:, None] * y[lag:])
    if z is None:
        z = xwy
    else:
        z += xwy
    return z


def wtlcc_dense_sparse_dense(w, x, y, z=None, lag=0):
    y = y.tocsr()
    if x is None:
        if z is None:
            z = np.zeros((1, y.shape[1]), order="F")
        return _wtlcc_ones_csr_dense(
            w, y.data, y.indices, y.indptr, y.shape[0], y.shape[1], z, lag
        )
    else:
        if z is None:
            z = np.zeros((x.shape[1], y.shape[1]), order="F")
        return _wtlcc_dense_csr_dense(
            w, x, y.data, y.indices, y.indptr, y.shape[0], y.shape[1], z, lag
        )


def wtlcc_sparse_dense_dense(w, x, y, z=None, lag=0):
    x = x.tocsr()
    if y is None:
        if z is None:
            z = np.zeros((x.shape[1], 1), order="C")
        return _wtlcc_csr_ones_dense(
            w, x.data, x.indices, x.indptr, x.shape[0], x.shape[1], z, lag
        )
    else:
        if z is None:
            z = np.zeros((x.shape[1], y.shape[1]), order="C")
        return _wtlcc_csr_dense_dense(
            w, x.data, x.indices, x.indptr, x.shape[0], x.shape[1], y, z, lag
        )


def wtlcc_sparse_sparse_dense(w, x, y, z=None, lag=0):
    x = x.tocsr()
    y = y.tocsr()
    if z is None:
        z = np.zeros((x.shape[1], y.shape[1]))
    return _wtlcc_csr_csr_dense(
        w,
        x.data,
        x.indices,
        x.indptr,
        x.shape[0],
        x.shape[1],
        y.data,
        y.indices,
        y.indptr,
        y.shape[0],
        y.shape[1],
        z,
        lag,
    )


@nb.njit(fastmath=True)
def _wtlcc_ones_csr_dense(
    w, y_data, y_indices, y_indptr, y_rows, y_cols, z, lag
):
    assert lag >= 0 and w.ndim == 1
    assert y_rows == len(w) + lag
    assert z.ndim == 2 and z.shape[0] == 1 and z.shape[1] == y_cols
    for k in range(len(w)):
        w_k = w[k]
        for jj in range(y_indptr[k + lag], y_indptr[k + lag + 1]):
            j = y_indices[jj]
            z[0, j] += w_k * y_data[jj]
    return z


@nb.njit(fastmath=True)
def _wtlcc_dense_csr_dense(
    w, x, y_data, y_indices, y_indptr, y_rows, y_cols, z, lag
):
    assert lag >= 0 and w.ndim == 1
    assert x.ndim == 2 and x.shape[0] == len(w) + lag
    assert y_rows == len(w) + lag
    assert z.ndim == 2 and z.shape[0] == x.shape[1] and z.shape[1] == y_cols
    x_cols = x.shape[1]
    for k in range(len(w)):
        w_k = w[k]
        for jj in range(y_indptr[k + lag], y_indptr[k + lag + 1]):
            j = y_indices[jj]
            v = w_k * y_data[jj]
            for i in range(x_cols):
                z[i, j] += v * x[k, i]
    return z


@nb.njit(fastmath=True)
def _wtlcc_csr_ones_dense(
    w, x_data, x_indices, x_indptr, x_rows, x_cols, z, lag
):
    assert lag >= 0 and w.ndim == 1
    assert x_rows == len(w) + lag
    assert z.ndim == 2 and z.shape[0] == x_cols and z.shape[1] == 1
    for k in range(len(w)):
        w_k = w[k]
        for ii in range(x_indptr[k], x_indptr[k + 1]):
            i = x_indices[ii]
            z[i, 0] += w_k * x_data[ii]
    return z


@nb.njit(fastmath=True)
def _wtlcc_csr_dense_dense(
    w, x_data, x_indices, x_indptr, x_rows, x_cols, y, z, lag
):
    assert lag >= 0 and w.ndim == 1
    assert x_rows == len(w) + lag
    assert y.ndim == 2 and y.shape[0] == len(w) + lag
    assert z.ndim == 2 and z.shape[0] == x_cols and z.shape[1] == y.shape[1]
    y_cols = y.shape[1]
    for k in range(len(w)):
        w_k = w[k]
        for ii in range(x_indptr[k], x_indptr[k + 1]):
            i = x_indices[ii]
            v = w_k * x_data[ii]
            for j in range(y_cols):
                z[i, j] += v * y[k + lag, j]
    return z


@nb.njit(fastmath=True)
def _wtlcc_csr_csr_dense(
    w,
    x_data,
    x_indices,
    x_indptr,
    x_rows,
    x_cols,
    y_data,
    y_indices,
    y_indptr,
    y_rows,
    y_cols,
    z,
    lag,
):
    assert lag >= 0 and w.ndim == 1
    assert x_rows == len(w) + lag
    assert y_rows == len(w) + lag
    assert z.ndim == 2 and z.shape[0] == x_cols and z.shape[1] == y_cols
    for k in range(len(w)):
        w_k = w[k]
        for ii in range(x_indptr[k], x_indptr[k + 1]):
            i = x_indices[ii]
            v = w_k * x_data[ii]
            for jj in range(y_indptr[k + lag], y_indptr[k + lag + 1]):
                j = y_indices[jj]
                z[i, j] += v * y_data[jj]
    return z
