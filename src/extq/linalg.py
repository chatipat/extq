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


def batched_block(blocks):
    n1 = len(blocks)
    n2 = len(blocks[0])
    assert all(len(b) == n2 for b in blocks)

    dtypes = []
    batch_dims = []
    k1 = [None] * n1
    k2 = [None] * n2
    for i in range(n1):
        for j in range(n2):
            a = blocks[i][j]
            if a is not None:
                dtypes.append(a.dtype)
                batch_dims.append(a.shape[:-2])
                if k1[i] is None:
                    k1[i] = a.shape[-2]
                if k2[j] is None:
                    k2[j] = a.shape[-1]
                assert k1[i] == a.shape[-2]
                assert k2[j] == a.shape[-1]
    dtype = np.result_type(*dtypes)
    batch_dims = np.broadcast_shapes(*batch_dims)
    assert None not in k1
    assert None not in k2
    k1 = np.array(k1)
    k2 = np.array(k2)

    offset1 = np.concatenate([[0], np.cumsum(k1)])
    offset2 = np.concatenate([[0], np.cumsum(k2)])
    out = np.zeros((*batch_dims, offset1[-1], offset2[-1]), dtype=dtype)
    for i in range(n1):
        for j in range(n2):
            a = blocks[i][j]
            if a is not None:
                out[..., offset1[i] : offset1[i + 1], offset2[j] : offset2[j + 1]] = a
    return out


def batched_block_diag(*arrs):
    assert all(a.ndim >= 2 for a in arrs)
    shape = np.broadcast_shapes(*[a.shape[:-2] for a in arrs])
    k1 = np.sum([a.shape[-2] for a in arrs])
    k2 = np.sum([a.shape[-1] for a in arrs])
    dtype = np.result_type(*[a.dtype for a in arrs])

    out = np.zeros((*shape, k1, k2), dtype=dtype)
    r = 0
    c = 0
    for a in arrs:
        out[..., r : r + a.shape[-2], c : c + a.shape[-1]] = a
        r += a.shape[-2]
        c += a.shape[-1]
    assert r == k1 and c == k2

    return out


def batched_kron(arr1, arr2):
    m1, n1 = arr1.shape[-2:]
    m2, n2 = arr2.shape[-2:]
    out = arr1[..., :, None, :, None] * arr2[..., None, :, None, :]
    out = out.reshape(*out.shape[:-4], m1 * m2, n1 * n2)
    return out


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
        if isinstance(b, (sp.sparse.csr_array, sp.sparse.csr_matrix)):
            return _scale_rows_csr(a, b)
        elif isinstance(b, (sp.sparse.csc_array, sp.sparse.csc_matrix)):
            return _scale_rows_csc(a, b)
        else:
            return sp.sparse.diags_array(a) @ b
    else:
        if np.ndim(b) >= 2:
            return a[:, None] * b
        else:
            return a * b


def scale_cols(a, b):
    if sp.sparse.issparse(a):
        if isinstance(a, (sp.sparse.csr_array, sp.sparse.csr_matrix)):
            return _scale_cols_csr(a, b)
        elif isinstance(a, (sp.sparse.csc_array, sp.sparse.csc_matrix)):
            return _scale_cols_csc(a, b)
        else:
            return a @ sp.sparse.diags_array(b)
    else:
        return a * b


def _scale_rows_csr(a, b):
    data = np.repeat(a, np.diff(b.indptr)) * b.data
    return sp.sparse.csr_array((data, b.indices, b.indptr), shape=b.shape)


def _scale_rows_csc(a, b):
    data = a[b.indices] * b.data
    return sp.sparse.csc_array((data, b.indices, b.indptr), shape=b.shape)


def _scale_cols_csr(a, b):
    data = a.data * b[a.indices]
    return sp.sparse.csr_array((data, a.indices, a.indptr), shape=a.shape)


def _scale_cols_csc(a, b):
    data = a.data * np.repeat(b, np.diff(a.indptr))
    return sp.sparse.csc_array((data, a.indices, a.indptr), shape=a.shape)
