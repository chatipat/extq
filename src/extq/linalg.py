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


def scale_rows(a, b):
    if scipy.sparse.issparse(b):
        b = b.tocsr()
        return scipy.sparse.csr_matrix(
            (np.repeat(a, np.diff(b.indptr)), b.indices, b.indptr),
            shape=b.shape,
        )
    else:
        return a[:, None] * b
