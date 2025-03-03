import numpy as np
import scipy as sp


def as_dense(a):
    return a.toarray() if sp.sparse.issparse(a) else a


def block_diag(mats, format="array"):
    if format != "array":
        return sp.sparse.block_diag(mats, format=format)
    else:
        arrs = []
        for mat in mats:
            if sp.sparse.issparse(mat):
                mat = mat.toarray()
            arrs.append(mat)
        return sp.linalg.block_diag(*arrs)


def inv(a):
    if sp.sparse.issparse(a):
        return sp.sparse.linalg.inv(a)
    else:
        return sp.linalg.inv(a)


def solve(a, b):
    if sp.sparse.issparse(a):
        return sp.sparse.linalg.spsolve(a, b)
    else:
        return sp.linalg.solve(a, b)


def factorized(a):
    if sp.sparse.issparse(a):
        return sp.sparse.linalg.factorized(a)
    else:
        lu, piv = sp.linalg.lu_factor(a)
        return lambda b: sp.linalg.lu_solve((lu, piv), b)


def eigh(a, b=None, k=None, which="LM"):
    if sp.sparse.issparse(a) or sp.sparse.issparse(b):
        eigvals, eigvecs = sp.sparse.linalg.eigsh(a, k=k, m=b, which=which)
    else:
        eigvals, eigvecs = sp.linalg.eigh(a, b)
    if which == "LM":
        order = np.argsort(np.abs(eigvals))[::-1]
    elif which == "SM":
        order = np.argsort(np.abs(eigvals))
    elif which == "LA":
        order = np.argsort(np.real(eigvals))[::-1]
    elif which == "SA":
        order = np.argsort(np.real(eigvals))
    else:
        msg = f"which ({which}) must be 'LM', 'SM', 'LA', or 'SA'"
        raise ValueError(msg)
    if k is not None:
        order = order[:k]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs


def expm_multiply(a, b):
    if sp.sparse.issparse(b):
        # expm(a) @ b is usually dense even if a and b are sparse
        b = b.toarray()
    if sp.sparse.issparse(a):
        return sp.sparse.linalg.expm_multiply(a, b)
    else:
        return sp.linalg.expm(a) @ b


def scale_rows(a, b):
    if sp.sparse.issparse(b):
        if isinstance(b, sp.sparse.csr_matrix):
            return _scale_rows_csr(a, b)
        elif isinstance(b, sp.sparse.csc_matrix):
            return _scale_rows_csc(a, b)
        else:
            return sp.sparse.diags(a) @ b
    else:
        if np.ndim(b) >= 2:
            return a[:, None] * b
        else:
            return a * b


def scale_cols(a, b):
    if sp.sparse.issparse(a):
        if isinstance(a, sp.sparse.csr_matrix):
            return _scale_cols_csr(a, b)
        elif isinstance(a, sp.sparse.csc_matrix):
            return _scale_cols_csc(a, b)
        else:
            return a @ sp.sparse.diags(b)
    else:
        return a * b


def _scale_rows_csr(a, b):
    data = np.repeat(a, np.diff(b.indptr)) * b.data
    return sp.sparse.csr_matrix((data, b.indices, b.indptr), shape=b.shape)


def _scale_rows_csc(a, b):
    data = a[b.indices] * b.data
    return sp.sparse.csc_matrix((data, b.indices, b.indptr), shape=b.shape)


def _scale_cols_csr(a, b):
    data = a.data * b[a.indices]
    return sp.sparse.csr_matrix((data, a.indices, a.indptr), shape=a.shape)


def _scale_cols_csc(a, b):
    data = a.data * np.repeat(b, np.diff(a.indptr))
    return sp.sparse.csc_matrix((data, a.indices, a.indptr), shape=a.shape)
