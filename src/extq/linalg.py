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
