import numpy as np
import scipy.sparse


def spouter(m, op, a, b):
    """Compute the outer op of two vectors at nonzero entries of m.

    Parameters
    ----------
    m : sparse matrix
        Outer op is calculated at nonzero entries of this matrix.
    op : callable
        Ufunc taking in two vectors and returning one vector.
    a : array-like
        First input vector, flattened if not 1D.
    b : array-like
        Second input vector, flattened if not 1D. Must be the same shape
        as a.

    Returns
    -------
    sparse matrix
        Sparse matrix c with entries c[i,j] = op(a[i],b[j]) where
        m[i,j] is nonzero.

    """
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape
    row, col = m.nonzero()
    data = op(a.ravel()[row], b.ravel()[col])
    return scipy.sparse.csr_matrix((data, (row, col)), m.shape)
