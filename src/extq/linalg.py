import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


def block_diag(mats, format="array"):
    if format != "array":
        return scipy.sparse.block_diag(mats, format=format)
    else:
        arrs = []
        for mat in mats:
            if scipy.sparse.issparse(mat):
                mat = mat.toarray()
            arrs.append(mat)
        return scipy.linalg.block_diag(*arrs)


def inv(a):
    if scipy.sparse.issparse(a):
        return scipy.sparse.linalg.inv(a)
    else:
        return scipy.linalg.inv(a)


def solve(a, b):
    if scipy.sparse.issparse(a):
        return scipy.sparse.linalg.spsolve(a, b)
    else:
        return scipy.linalg.solve(a, b)


def factorized(a):
    if scipy.sparse.issparse(a):
        return scipy.sparse.linalg.factorized(a)
    else:
        lu, piv = scipy.linalg.lu_factor(a)
        return lambda b: scipy.linalg.lu_solve((lu, piv), b)


def expm_multiply(a, b):
    if scipy.sparse.issparse(b):
        # expm(a) @ b is usually dense even if a and b are sparse
        b = b.toarray()
    if scipy.sparse.issparse(a):
        return scipy.sparse.linalg.expm_multiply(a, b)
    else:
        return scipy.linalg.expm(a) @ b


def scale_rows(a, b):
    if scipy.sparse.issparse(b):
        if isinstance(b, scipy.sparse.csr_matrix):
            return _scale_rows_csr(a, b)
        elif isinstance(b, scipy.sparse.csc_matrix):
            return _scale_rows_csc(a, b)
        else:
            return scipy.sparse.diags(a) @ b
    else:
        if np.ndim(b) >= 2:
            return a[:, None] * b
        else:
            return a * b


def scale_cols(a, b):
    if scipy.sparse.issparse(a):
        if isinstance(a, scipy.sparse.csr_matrix):
            return _scale_cols_csr(a, b)
        elif isinstance(a, scipy.sparse.csc_matrix):
            return _scale_cols_csc(a, b)
        else:
            return a @ scipy.sparse.diags(b)
    else:
        return a * b


def _scale_rows_csr(a, b):
    data = np.repeat(a, np.diff(b.indptr)) * b.data
    return scipy.sparse.csr_matrix((data, b.indices, b.indptr), shape=b.shape)


def _scale_rows_csc(a, b):
    data = a[b.indices] * b.data
    return scipy.sparse.csc_matrix((data, b.indices, b.indptr), shape=b.shape)


def _scale_cols_csr(a, b):
    data = a.data * b[a.indices]
    return scipy.sparse.csr_matrix((data, a.indices, a.indptr), shape=a.shape)


def _scale_cols_csc(a, b):
    data = a.data * np.repeat(b, np.diff(a.indptr))
    return scipy.sparse.csc_matrix((data, a.indices, a.indptr), shape=a.shape)
