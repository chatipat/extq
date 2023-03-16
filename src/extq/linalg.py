import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


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
