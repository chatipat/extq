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
        return b.multiply(a[:, None])
    else:
        if np.ndim(b) >= 2:
            return a[:, None] * b
        else:
            return a * b


def scale_cols(a, b):
    if scipy.sparse.issparse(a):
        return a.multiply(b[None, :])
    else:
        return a * b
