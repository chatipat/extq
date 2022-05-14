import operator

import numpy as np
import scipy.sparse


def bmap(f, a):
    def helper(x):
        if x is None:
            return None
        else:
            return f(x)

    return np.vectorize(helper, otypes=[object])(a)


def bmap2(f, a, b):
    def helper(x, y):
        if x is None:
            assert y is None
            return None
        else:
            assert y is not None
            return f(x, y)

    return np.vectorize(helper, otypes=[object])(a, b)


def btranspose(a):
    return bmap(lambda x: x.T, a).T


def badd(a, b):
    return bmap2(operator.add, a, b)


def bsub(a, b):
    return bmap2(operator.sub, a, b)


def bmatmul(f, a, b):
    m, l = a.shape
    _, n = b.shape
    assert b.shape == (l, n)
    c = np.full((m, n), None)
    for i in range(m):
        for j in range(n):
            for k in range(l):
                if a[i, k] is not None and b[k, j] is not None:
                    if c[i, j] is None:
                        c[i, j] = 0
                    c[i, j] += f(a[i, k], b[k, j])
    return c


def bshape(blocks):
    br, bc = blocks.shape
    rows = [None] * br
    cols = [None] * bc
    for i in range(br):
        for j in range(bc):
            if blocks[i, j] is not None:
                r, c = blocks[i, j].shape
                if rows[i] is None:
                    rows[i] = r
                if cols[j] is None:
                    cols[j] = c
                assert (rows[i], cols[j]) == (r, c)
    for r in rows:
        assert r is not None
    for c in cols:
        assert c is not None
    return (tuple(rows), tuple(cols))


def from_blocks(blocks, shape=None):
    if shape is not None:
        assert bshape(blocks) == shape
    if any_sparse(blocks):
        return scipy.sparse.bmat(blocks.copy(), format="csr")
    else:
        return scipy.sparse.bmat(blocks.copy()).toarray()


def to_blocks(mat, shape):
    s0, s1 = shape
    si = np.cumsum(np.concatenate([[0], s0]))
    sj = np.cumsum(np.concatenate([[0], s1]))
    assert mat.shape == (si[-1], sj[-1])
    blocks = np.full((len(s0), len(s1)), None)
    for i in range(len(s0)):
        for j in range(len(s1)):
            blocks[i, j] = mat[si[i] : si[i + 1], sj[j] : sj[j + 1]]
    return blocks


def to_blockvec(vec, shape):
    si = np.cumsum(np.concatenate([[0], shape]))
    assert vec.shape == (si[-1],)
    bvec = np.full(len(shape), None)
    for i in range(len(shape)):
        bvec[i] = vec[si[i] : si[i + 1]]
    return bvec


def any_sparse(blocks):
    m, n = blocks.shape
    for i in range(m):
        for j in range(n):
            if blocks[i, j] is not None:
                if scipy.sparse.issparse(blocks[i, j]):
                    return True
    return False
