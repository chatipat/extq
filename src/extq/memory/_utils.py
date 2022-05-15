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


def asblocks(blocks, ndim=2):
    result = np.empty(_shape(blocks, ndim), dtype=object)
    result[:] = blocks
    return result


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


def bconcatenate_lag(blocks, left=0, right=0):
    def helper(mats):
        if mats[0] is None:
            assert all(mat is None for mat in mats)
            return None
        else:
            assert all(mat is not None for mat in mats)
            return concatenate_lag(mats, left=left, right=right)

    blocks = asblocks(blocks, ndim=3)
    blocks = np.moveaxis(blocks, 0, -1)
    blocks = asblocks(blocks.tolist(), ndim=2)
    return np.vectorize(helper, otypes=[object])(blocks)


def concatenate_lag(mats, left=0, right=0):
    assert len(mats) > 0
    assert left >= 0 and right >= 0
    start = left
    stop = -right if right > 0 else None
    if len(mats) == 1:
        return mats[0][start:stop]
    if all(scipy.sparse.issparse(mat) for mat in mats):
        data = []
        indices = []
        indptr = []
        nrow = 0
        ncol = None
        offset = 0
        for mat in mats:
            mat = mat.tocsr()
            nrow += mat.shape[0] - (left + right)
            if ncol is None:
                ncol = mat.shape[1]
            assert ncol == mat.shape[1]
            s = mat.indptr[start:stop]
            data.append(mat.data[s[0] : s[-1]])
            indices.append(mat.indices[s[0] : s[-1]])
            indptr.append(s[:-1] + (offset - s[0]))
            offset += s[-1] - s[0]
        indptr.append([offset])
        return scipy.sparse.csr_matrix(
            (
                np.concatenate(data),
                np.concatenate(indices),
                np.concatenate(indptr),
            ),
            shape=(nrow, ncol),
        )
    else:
        return np.concatenate(
            [_asdense(mat)[start:stop] for mat in mats], axis=0
        )


def _shape(a, n):
    assert n >= 1
    shape = [len(a)]
    for _ in range(n - 1):
        a = a[0]
        shape.append(len(a))
    return shape


def _asdense(mat):
    if scipy.sparse.issparse(mat):
        return mat.toarray()
    else:
        return np.asarray(mat)
