import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from ..dga._utils import solve


def forward_committor(mat, basis, weights, guess, test_basis=None):
    if test_basis is None:
        test_basis = basis

    basis = np.asarray(basis)
    test_basis = np.asarray(test_basis)
    weights = np.asarray(weights)
    guess = np.asarray(guess)

    shape = weights.shape
    assert guess.shape == shape
    assert test_basis.shape == basis.shape

    w = np.ravel(weights)
    g = np.ravel(guess)

    c0 = test_basis.T @ scipy.sparse.diags(w) @ basis
    g0 = test_basis.T @ scipy.sparse.diags(w) @ g

    ct = mat[:-2, :-2]
    gt = mat[:-2, -2] + mat[:-2, -1]

    coeffs = solve(ct - c0, -(gt - g0))
    return (basis @ coeffs + guess).reshape(shape)


def backward_committor(mat, basis, weights, guess, test_basis=None):
    return forward_committor(
        mat.T, basis, weights, guess, test_basis=test_basis
    )


def rate(mat, lag):
    n = mat.shape[0] // 2
    ct = mat[:n, n:]
    return (ct[-2, -2] + ct[-2, -1] + ct[-1, -2] + ct[-1, -1]) / lag


def forward_committor_generator(generator, weights, in_domain, guess):
    weights = np.asarray(weights)
    in_domain = np.asarray(in_domain)
    guess = np.asarray(guess)

    shape = weights.shape
    assert in_domain.shape == shape
    assert guess.shape == shape

    d = np.ravel(in_domain)
    dc = np.logical_not(d)
    g = np.ravel(guess)

    a = generator[d, :][:, d]
    b = generator[d, :][:, dc] @ g[dc, None]
    return scipy.sparse.bmat([[a, b], [None, 0]], format="csr")


def forward_committor_left(test_basis, weights, in_domain):
    test_basis = np.asarray(test_basis)
    weights = np.asarray(weights)
    in_domain = np.asarray(in_domain)

    shape = weights.shape
    assert in_domain.shape == shape

    w = np.ravel(weights)
    d = np.ravel(in_domain)

    lvec = np.zeros((np.count_nonzero(d) + 1, test_basis.shape[1] + 2))
    lvec[:-1, :-2] = scipy.sparse.diags(w[d]) @ test_basis[d]
    lvec[-1, -1] = 1.0
    return lvec


def forward_committor_right(basis, weights, in_domain, guess):
    basis = np.asarray(basis)
    weights = np.asarray(weights)
    in_domain = np.asarray(in_domain)
    guess = np.asarray(guess)

    shape = weights.shape
    assert in_domain.shape == shape
    assert guess.shape == shape

    d = np.ravel(in_domain)
    g = np.ravel(guess)

    rvec = np.zeros((np.count_nonzero(d) + 1, basis.shape[1] + 2))
    rvec[:-1, :-2] = basis[d]
    rvec[:-1, -2] = g[d]
    rvec[-1, -1] = 1.0
    return rvec


def backward_committor_generator(generator, weights, in_domain, guess):
    weights = np.asarray(weights)
    in_domain = np.asarray(in_domain)
    guess = np.asarray(guess)

    shape = weights.shape
    assert in_domain.shape == shape
    assert guess.shape == shape

    w = np.ravel(weights)
    d = np.ravel(in_domain)
    dc = np.logical_not(d)
    g = np.ravel(guess)

    a = generator[d, :][:, d]
    b = g[dc, None].T @ scipy.sparse.diags(w[dc]) @ generator[dc, :][:, d]
    mat = scipy.sparse.bmat([[a, None], [b, 0]], format="csr")
    return mat


def backward_committor_left(basis, weights, in_domain, guess):
    basis = np.asarray(basis)
    weights = np.asarray(weights)
    in_domain = np.asarray(in_domain)
    guess = np.asarray(guess)

    shape = weights.shape
    assert in_domain.shape == shape
    assert guess.shape == shape

    w = np.ravel(weights)
    d = np.ravel(in_domain)
    g = np.ravel(guess)

    lvec = np.zeros((np.count_nonzero(d) + 1, basis.shape[1] + 2))
    lvec[:-1, :-2] = scipy.sparse.diags(w[d]) @ basis[d]
    lvec[:-1, -2] = scipy.sparse.diags(w[d]) @ g[d]
    lvec[-1, -1] = 1.0
    return lvec


def backward_committor_right(test_basis, weights, in_domain):
    test_basis = np.asarray(test_basis)
    weights = np.asarray(weights)
    in_domain = np.asarray(in_domain)

    shape = weights.shape
    assert in_domain.shape == shape

    d = np.ravel(in_domain)

    rvec = np.zeros((np.count_nonzero(d) + 1, test_basis.shape[1] + 2))
    rvec[:-1, :-2] = test_basis[d]
    rvec[-1, -1] = 1.0
    return rvec


def rate_generator(
    generator, forward_q, backward_q, weights, in_domain, rxn_coord
):
    forward_q = np.asarray(forward_q)
    backward_q = np.asarray(backward_q)
    weights = np.asarray(weights)
    in_domain = np.asarray(in_domain)
    rxn_coord = np.asarray(rxn_coord)

    shape = weights.shape
    assert forward_q.shape == shape
    assert backward_q.shape == shape
    assert in_domain.shape == shape
    assert rxn_coord.shape == shape

    qp = np.ravel(forward_q)
    qm = np.ravel(backward_q)
    w = np.ravel(weights)
    d = np.ravel(in_domain)
    dc = np.logical_not(d)
    h = np.ravel(rxn_coord)

    genh = (
        generator @ scipy.sparse.diags(h) - scipy.sparse.diags(h) @ generator
    )
    wqmdc = scipy.sparse.diags(w[dc]) @ np.ravel(qm)[dc, None]
    qpdc = np.ravel(qp)[dc, None]

    dd = genh[d, :][:, d]
    db = genh[d, :][:, dc] @ qpdc
    ad = wqmdc.T @ genh[dc, :][:, d]
    ab = wqmdc.T @ genh[dc, :][:, dc] @ qpdc
    ur = scipy.sparse.bmat([[dd, db], [ad, ab]])
    lr = forward_committor_generator(generator, weights, in_domain, forward_q)
    ul = backward_committor_generator(
        generator, weights, in_domain, backward_q
    )
    return scipy.sparse.bmat([[ul, ur], [None, lr]], format="csr")


def rate_left(basis, backward_q, weights, in_domain):
    wxp = forward_committor_left(basis, weights, in_domain)
    wxm = backward_committor_left(basis, weights, in_domain, backward_q)
    return scipy.linalg.block_diag(wxm, wxp)


def rate_right(basis, forward_q, weights, in_domain):
    yp = forward_committor_right(basis, weights, in_domain, forward_q)
    ym = backward_committor_right(basis, weights, in_domain)
    return scipy.linalg.block_diag(ym, yp)
