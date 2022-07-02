import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def reweight_matrix(generator, basis, weights, lag, test=None):
    if test is None:
        test = basis
    return _reweight_matrix(generator, basis, test, weights, lag)


def _reweight_matrix(gen, x_w, y_w, w, lag):
    w = np.ravel(w)
    c = np.ones((len(w), 1))

    a = np.full((1, 1), None)
    a[0, 0] = gen

    x = np.full((1, 2), None)
    x[0, 0] = scipy.sparse.diags(w) @ c
    x[0, 1] = scipy.sparse.diags(w) @ x_w

    y = np.full((1, 2), None)
    y[0, 0] = c
    y[0, 1] = y_w

    return _build(a, x, y, lag)


def reweight_transform(coeffs, basis, weights):
    return weights * (coeffs[0] + basis @ coeffs[1:]).reshape(weights.shape)


def forward_committor_matrix(
    generator, basis, weights, in_domain, guess, lag, test=None
):
    return forward_feynman_kac_matrix(
        generator,
        basis,
        weights,
        in_domain,
        np.zeros_like(weights),
        guess,
        lag,
        test=test,
    )


def forward_mfpt_matrix(
    generator, basis, weights, in_domain, guess, lag, test=None
):
    return forward_feynman_kac_matrix(
        generator,
        basis,
        weights,
        in_domain,
        np.ones_like(weights),
        guess,
        lag,
        test=test,
    )


def forward_feynman_kac_matrix(
    generator, basis, weights, in_domain, function, guess, lag, test=None
):
    if test is None:
        test = basis
    return _forward_matrix(
        generator, test, basis, weights, in_domain, function, guess, lag
    )


def _forward_matrix(gen, x_f, y_f, w, d_f, f_f, g_f, lag):
    w = np.ravel(w)
    d_f = np.ravel(d_f)
    f_f = np.ravel(f_f)
    g_f = np.ravel(g_f)
    b_f = np.where(d_f, 0.0, g_f)
    c = np.ones((len(w), 1))

    a = np.full((2, 2), None)
    a[0, 0] = gen[d_f, :][:, d_f]
    a[0, 1] = (
        gen[d_f, :] @ scipy.sparse.diags(b_f) + scipy.sparse.diags(f_f)[d_f, :]
    )
    a[1, 1] = gen

    x = np.full((2, 2), None)
    x[0, 0] = scipy.sparse.diags(w[d_f]) @ x_f[d_f]
    x[1, 1] = scipy.sparse.diags(w) @ c

    y = np.full((2, 2), None)
    y[0, 0] = y_f[d_f]
    y[0, 1] = scipy.sparse.diags(g_f[d_f]) @ c[d_f]
    y[1, 1] = c

    return _build(a, x, y, lag)


def forward_transform(coeffs, basis, guess):
    return guess + (basis @ (coeffs[:-1] / coeffs[-1])).reshape(guess.shape)


def backward_committor_matrix(
    generator,
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    w_test=None,
    test=None,
):
    return backward_feynman_kac_matrix(
        generator,
        w_basis,
        basis,
        weights,
        in_domain,
        np.zeros_like(weights),
        guess,
        lag,
        w_test=w_test,
        test=test,
    )


def backward_mfpt_matrix(
    generator,
    w_basis,
    basis,
    weights,
    in_domain,
    guess,
    lag,
    w_test=None,
    test=None,
):
    return backward_feynman_kac_matrix(
        generator,
        w_basis,
        basis,
        weights,
        in_domain,
        np.ones_like(weights),
        guess,
        lag,
        w_test=w_test,
        test=test,
    )


def backward_feynman_kac_matrix(
    generator,
    w_basis,
    basis,
    weights,
    in_domain,
    function,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    return _backward_matrix(
        generator,
        w_basis,
        w_test,
        basis,
        test,
        weights,
        in_domain,
        function,
        guess,
        lag,
    )


def _backward_matrix(gen, x_w, y_w, x_b, y_b, w, d_b, f_b, g_b, lag):
    w = np.ravel(w)
    d_b = np.ravel(d_b)
    f_b = np.ravel(f_b)
    g_b = np.ravel(g_b)
    b_b = np.where(d_b, 0.0, g_b)
    c = np.ones((len(w), 1))

    a = np.full((2, 2), None)
    a[0, 0] = gen
    a[0, 1] = (
        scipy.sparse.diags(b_b) @ gen[:, d_b] + scipy.sparse.diags(f_b)[:, d_b]
    )
    a[1, 1] = gen[d_b, :][:, d_b]

    x = np.full((2, 3), None)
    x[0, 0] = scipy.sparse.diags(w) @ c
    x[0, 1] = scipy.sparse.diags(w) @ x_w
    x[1, 0] = scipy.sparse.diags(w[d_b] * g_b[d_b]) @ c[d_b]
    x[1, 1] = scipy.sparse.diags(w[d_b] * g_b[d_b]) @ x_w[d_b]
    x[1, 2] = scipy.sparse.diags(w[d_b]) @ x_b[d_b]

    y = np.full((2, 3), None)
    y[0, 0] = c
    y[0, 1] = y_w
    y[1, 1] = y_b[d_b]

    return _build(a, x, y, lag)


def backward_transform(coeffs, w_basis, basis, guess):
    n = w_basis.shape[1] + 1
    return guess + (
        (basis @ coeffs[n:]) / (coeffs[0] + w_basis @ coeffs[1:n])
    ).reshape(guess)


def reweight_integral_matrix(
    generator, basis, weights, values, lag, test=None
):
    if test is None:
        test = basis
    return _reweight_integral_matrix(
        generator, basis, test, weights, values, lag
    )


def _reweight_integral_matrix(gen, x_w, y_w, w, v, lag):
    w = np.ravel(w)
    c = np.ones((len(w), 1))

    a = np.full((2, 2), None)
    # upper left block
    a[0, 0] = gen
    # upper right block
    a[0, 1] = v
    # lower right block
    a[1, 1] = gen

    x = np.full((2, 3), None)
    x[0, 0] = scipy.sparse.diags(w) @ c
    x[0, 1] = scipy.sparse.diags(w) @ x_w
    x[1, 2] = scipy.sparse.diags(w) @ c

    y = np.full((2, 3), None)
    y[0, 0] = c
    y[0, 1] = y_w
    y[1, 2] = c

    return _build(a, x, y, lag)


def forward_committor_integral_matrix(
    generator,
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    w_test=None,
    test=None,
):
    return forward_feynman_kac_integral_matrix(
        generator,
        w_basis,
        basis,
        weights,
        in_domain,
        values,
        np.zeros_like(weights),
        guess,
        lag,
        w_test=w_test,
        test=test,
    )


def forward_mfpt_integral_matrix(
    generator,
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    w_test=None,
    test=None,
):
    return forward_feynman_kac_integral_matrix(
        generator,
        w_basis,
        basis,
        weights,
        in_domain,
        values,
        np.ones_like(weights),
        guess,
        lag,
        w_test=w_test,
        test=test,
    )


def forward_feynman_kac_integral_matrix(
    generator,
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    function,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    return _forward_integral_matrix(
        generator,
        w_basis,
        w_test,
        test,
        basis,
        weights,
        in_domain,
        values,
        function,
        guess,
        lag,
    )


def _forward_integral_matrix(
    gen, x_w, y_w, x_f, y_f, w, d_f, v, f_f, g_f, lag
):
    w = np.ravel(w)
    d_f = np.ravel(d_f)
    f_f = np.ravel(f_f)
    g_f = np.ravel(g_f)
    b_f = np.where(d_f, 0.0, g_f)
    c = np.ones((len(w), 1))

    a = np.full((3, 3), None)
    # upper left block
    a[0, 0] = gen
    # upper right block
    a[0, 1] = v[:, d_f]
    a[0, 2] = v @ scipy.sparse.diags(b_f)
    # lower right block
    a[1, 1] = gen[d_f, :][:, d_f]
    a[1, 2] = (
        gen[d_f, :] @ scipy.sparse.diags(b_f) + scipy.sparse.diags(f_f)[d_f, :]
    )
    a[2, 2] = gen

    x = np.full((3, 4), None)
    x[0, 0] = scipy.sparse.diags(w) @ c
    x[0, 1] = scipy.sparse.diags(w) @ x_w
    x[1, 2] = scipy.sparse.diags(w[d_f]) @ x_f[d_f]
    x[2, 3] = scipy.sparse.diags(w) @ c

    y = np.full((3, 4), None)
    y[0, 0] = c
    y[0, 1] = y_w
    y[1, 2] = y_f[d_f]
    y[1, 3] = scipy.sparse.diags(g_f[d_f]) @ c[d_f]
    y[2, 3] = c

    return _build(a, x, y, lag)


def backward_committor_integral_matrix(
    generator,
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    w_test=None,
    test=None,
):
    return backward_feynman_kac_integral_matrix(
        generator,
        w_basis,
        basis,
        weights,
        in_domain,
        values,
        np.zeros_like(weights),
        guess,
        lag,
        w_test=w_test,
        test=test,
    )


def backward_mfpt_integral_matrix(
    generator,
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    guess,
    lag,
    w_test=None,
    test=None,
):
    return backward_feynman_kac_integral_matrix(
        generator,
        w_basis,
        basis,
        weights,
        in_domain,
        values,
        np.ones_like(weights),
        guess,
        lag,
        w_test=w_test,
        test=test,
    )


def backward_feynman_kac_integral_matrix(
    generator,
    w_basis,
    basis,
    weights,
    in_domain,
    values,
    function,
    guess,
    lag,
    w_test=None,
    test=None,
):
    if w_test is None:
        w_test = w_basis
    if test is None:
        test = basis
    return _backward_integral_matrix(
        generator,
        w_basis,
        w_test,
        basis,
        test,
        weights,
        in_domain,
        values,
        function,
        guess,
        lag,
    )


def _backward_integral_matrix(
    gen, x_w, y_w, x_b, y_b, w, d_b, v, f_b, g_b, lag
):
    w = np.ravel(w)
    d_b = np.ravel(d_b)
    f_b = np.ravel(f_b)
    g_b = np.ravel(g_b)
    b_b = np.where(d_b, 0.0, g_b)
    c = np.ones((len(w), 1))

    a = np.full((3, 3), None)
    # upper left block
    a[0, 0] = gen
    a[0, 1] = (
        scipy.sparse.diags(b_b) @ gen[:, d_b] + scipy.sparse.diags(f_b)[:, d_b]
    )
    a[1, 1] = gen[d_b, :][:, d_b]
    # upper right block
    a[0, 2] = scipy.sparse.diags(b_b) @ v
    a[1, 2] = v[d_b, :]
    # lower right block
    a[2, 2] = gen

    x = np.full((3, 4), None)
    x[0, 0] = scipy.sparse.diags(w) @ c
    x[0, 1] = scipy.sparse.diags(w) @ x_w
    x[1, 0] = scipy.sparse.diags(w[d_b] * g_b[d_b]) @ c[d_b]
    x[1, 1] = scipy.sparse.diags(w[d_b] * g_b[d_b]) @ x_w[d_b]
    x[1, 2] = scipy.sparse.diags(w[d_b]) @ x_b[d_b]
    x[2, 3] = scipy.sparse.diags(w) @ c

    y = np.full((3, 4), None)
    y[0, 0] = c
    y[0, 1] = y_w
    y[1, 2] = y_b[d_b]
    y[2, 3] = c

    return _build(a, x, y, lag)


def tpt_integral_matrix(
    generator,
    w_basis,
    b_basis,
    f_basis,
    weights,
    in_domain,
    values,
    b_guess,
    f_guess,
    lag,
    w_test=None,
    b_test=None,
    f_test=None,
):
    return integral_matrix(
        generator,
        w_basis,
        b_basis,
        f_basis,
        weights,
        in_domain,
        in_domain,
        values,
        np.zeros_like(weights),
        np.zeros_like(weights),
        b_guess,
        f_guess,
        lag,
        w_test=w_test,
        b_test=b_test,
        f_test=f_test,
    )


def integral_matrix(
    generator,
    w_basis,
    b_basis,
    f_basis,
    weights,
    b_domain,
    f_domain,
    values,
    b_function,
    f_function,
    b_guess,
    f_guess,
    lag,
    w_test=None,
    b_test=None,
    f_test=None,
):
    if w_test is None:
        w_test = w_basis
    if b_test is None:
        b_test = b_basis
    if f_test is None:
        f_test = f_basis
    return _integral_matrix(
        generator,
        w_basis,
        w_test,
        b_basis,
        b_test,
        f_test,
        f_basis,
        weights,
        b_domain,
        f_domain,
        values,
        b_function,
        f_function,
        b_guess,
        f_guess,
        lag,
    )


def _integral_matrix(
    gen, x_w, y_w, x_b, y_b, x_f, y_f, w, d_b, d_f, v, f_b, f_f, g_b, g_f, lag
):
    w = np.ravel(w)
    d_b = np.ravel(d_b)
    d_f = np.ravel(d_f)
    f_b = np.ravel(f_b)
    f_f = np.ravel(f_f)
    g_b = np.ravel(g_b)
    g_f = np.ravel(g_f)
    b_b = np.where(d_b, 0.0, g_b)
    b_f = np.where(d_f, 0.0, g_f)
    c = np.ones((len(w), 1))

    a = np.full((4, 4), None)
    # upper left block
    a[0, 0] = gen
    a[0, 1] = (
        scipy.sparse.diags(b_b) @ gen[:, d_b] + scipy.sparse.diags(f_b)[:, d_b]
    )
    a[1, 1] = gen[d_b, :][:, d_b]
    # upper right block
    a[0, 2] = scipy.sparse.diags(b_b) @ v[:, d_f]
    a[0, 3] = scipy.sparse.diags(b_b) @ v @ scipy.sparse.diags(b_f)
    a[1, 2] = v[d_b, :][:, d_f]
    a[1, 3] = v[d_b, :] @ scipy.sparse.diags(b_f)
    # lower right block
    a[2, 2] = gen[d_f, :][:, d_f]
    a[2, 3] = (
        gen[d_f, :] @ scipy.sparse.diags(b_f) + scipy.sparse.diags(f_f)[d_f, :]
    )
    a[3, 3] = gen

    x = np.full((4, 5), None)
    x[0, 0] = scipy.sparse.diags(w) @ c
    x[0, 1] = scipy.sparse.diags(w) @ x_w
    x[1, 0] = scipy.sparse.diags(w[d_b] * g_b[d_b]) @ c
    x[1, 1] = scipy.sparse.diags(w[d_b] * g_b[d_b]) @ x_w[d_b]
    x[1, 2] = scipy.sparse.diags(w[d_b]) @ x_b[d_b]
    x[2, 3] = scipy.sparse.diags(w[d_f]) @ x_f[d_f]
    x[3, 4] = scipy.sparse.diags(w) @ c

    y = np.full((4, 5), None)
    y[0, 0] = c
    y[0, 1] = y_w
    y[1, 2] = y_b[d_b]
    y[2, 3] = y_f[d_f]
    y[2, 4] = scipy.sparse.diags(g_f[d_f]) @ c[d_f]
    y[3, 4] = c

    return _build(a, x, y, lag)


def _build(a, x, y, lag):
    a = scipy.sparse.bmat(a, format="csr")
    x = scipy.sparse.bmat(x, format="array")
    y = scipy.sparse.bmat(y, format="array")
    return x.T @ scipy.sparse.linalg.expm_multiply(a * lag, y)
