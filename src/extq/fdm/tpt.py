import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def forward_committor(generator, weights, in_domain, guess):
    """Compute the forward committor.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Reweighting factor to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    guess : (M,) ndarray of float
        Guess for the committor. Must obey boundary conditions.

    Returns
    -------
    (M,) ndarray of float
        Forward committor at each point.

    """
    return forward_feynman_kac(generator, weights, in_domain, 0.0, guess)


def backward_committor(generator, weights, in_domain, guess):
    """Compute the backward committor.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Reweighting factor to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    guess : (M,) ndarray of float
        Guess for the committor. Must obey boundary conditions.

    Returns
    -------
    (M,) ndarray of float
        Backward committor at each point.

    """
    return backward_feynman_kac(generator, weights, in_domain, 0.0, guess)


def forward_mfpt(generator, weights, in_domain, guess):
    """Compute the forward mean first passage time.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Change of measure to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    guess : (M,) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.

    Returns
    -------
    (M,) ndarray of float
        Forward mean first passage time at each point.

    """
    return forward_feynman_kac(generator, weights, in_domain, 1.0, guess)


def backward_mfpt(generator, weights, in_domain, guess):
    """Compute the backward mean first passage time.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Change of measure to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    guess : (M,) ndarray of float
        Guess for the mean first passage time. Must obey boundary
        conditions.

    Returns
    -------
    (M,) ndarray of float
        Backward mean first passage time at each point.

    """
    return backward_feynman_kac(generator, weights, in_domain, 1.0, guess)


def forward_feynman_kac(generator, weights, in_domain, function, guess):
    """Solve the forward Feynman-Kac formula.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Change of measure to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    function : (M,) ndarray of float
        Function to integrate. Must be zero outside the domain.
    guess : (M,) ndarray of float
        Guess of the solution. Must obey boundary conditions.

    Returns
    -------
    (M,) ndarray of float
        Solution of the Feynman-Kac formula at each point.

    """
    weights = np.asarray(weights)
    in_domain = np.asarray(in_domain)
    function = np.where(in_domain, function, 0.0)
    guess = np.asarray(guess)

    shape = weights.shape
    assert in_domain.shape == shape
    assert function.shape == shape
    assert guess.shape == shape

    d = in_domain.ravel()
    f = function.ravel()
    g = guess.ravel()

    a = generator[d, :][:, d]
    b = -generator[d, :] @ g - f[d]
    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return (
        g + scipy.sparse.identity(len(g), format="csr")[:, d] @ coeffs
    ).reshape(shape)


def backward_feynman_kac(generator, weights, in_domain, function, guess):
    """Solve the backward Feynman-Kac formula.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    weights : (M,) ndarray of float
        Change of measure to the invariant distribution for each point.
    in_domain : (M,) ndarray of bool
        Whether each point is in the domain.
    function : (M,) ndarray of float
        Function to integrate. Must be zero outside the domain.
    guess : (M,) ndarray of float
        Guess of the solution. Must obey boundary conditions.

    Returns
    -------
    (M,) ndarray of float
        Solution of the Feynman-Kac formula at each point.

    """
    pi = np.ravel(weights)
    adjoint_generator = (
        scipy.sparse.diags(1.0 / pi) @ generator.T @ scipy.sparse.diags(pi)
    )
    return forward_feynman_kac(
        adjoint_generator, weights, in_domain, function, guess
    )


def reweight(generator):
    """Compute the reweighting factors to the invariant distribution.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.

    Returns
    -------
    (M,) ndarray of float
        Reweighting factor at each point.

    """
    # w, v = sparse.linalg.eigs(tmat.T, k=1, which="LR")
    # fixed_index = np.argmax(np.abs(v[:, 0]))
    fixed_index = np.random.randint(generator.shape[0])
    mask = np.full(generator.shape[0], True)
    mask[fixed_index] = False

    a = generator.T[mask, :][:, mask]
    b = -generator.T[mask, fixed_index]
    coeffs = scipy.sparse.linalg.spsolve(a, b)

    weights = np.empty(generator.shape[0])
    weights[fixed_index] = 1.0
    weights[mask] = coeffs
    weights /= np.sum(weights)
    return weights


def rate(generator, forward_q, backward_q, weights, rxn_coords=None):
    """Compute the TPT rate.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    forward_q : (M,) ndarray of float
        Forward committor at each point.
    backward_q : (M,) ndarray of float
        Backward committor at each point.
    weights : (M,) ndarray of float.
        Reweighting factor at each point.
    rxn_coords : (M,) ndarray of float, optional
        Reaction coordinate at each point. This must be zero in the
        reactant state and one in the product state. If None, estimate
        the rate without using a reaction coordinate.

    Returns
    -------
    float
        TPT rate.

    """
    weights = np.asarray(weights)
    forward_q = np.asarray(forward_q)
    backward_q = np.asarray(backward_q)

    shape = weights.shape
    assert forward_q.shape == shape
    assert backward_q.shape == shape

    pi_qm = (weights * backward_q).ravel()
    qp = forward_q.ravel()

    if rxn_coords is None:
        numer = pi_qm @ generator @ qp
    else:
        rxn_coords = np.asarray(rxn_coords)
        assert rxn_coords.shape == shape
        h = rxn_coords.ravel()
        numer = pi_qm @ (generator @ (qp * h) - h * (generator @ qp))
    denom = np.sum(weights)
    return numer / denom


def current(generator, forward_q, backward_q, weights, cv):
    """Compute the reactive current at each point.

    Parameters
    ----------
    generator : (M, M) sparse matrix
        Generator matrix.
    forward_q : (M,) ndarray of float
        Forward committor at each point.
    backward_q : (M,) ndarray of float
        Backward committor at each point.
    weights : (M,) ndarray of float.
        Reweighting factor at each point.
    cv : (M,) ndarray of float
        Collective variable at each point.

    Returns
    -------
    (M,) ndarray of float
        Reactive current at each point.

    """
    weights = np.asarray(weights)
    forward_q = np.asarray(forward_q)
    backward_q = np.asarray(backward_q)

    shape = weights.shape
    assert forward_q.shape == shape
    assert backward_q.shape == shape

    cv = np.broadcast_to(cv, shape)

    pi_qm = (weights * backward_q).ravel()
    qp = forward_q.ravel()
    h = cv.ravel()

    forward_flux = pi_qm * (generator @ (qp * h) - h * (generator @ qp))
    backward_flux = ((pi_qm * h) @ generator - (pi_qm @ generator) * h) * qp
    numer = 0.5 * (forward_flux - backward_flux)
    denom = np.sum(weights)
    return (numer / denom).reshape(shape)


def expectation(generator, forward_q, backward_q, weights, ks, kt):
    weights = np.asarray(weights)
    forward_q = np.asarray(forward_q)
    backward_q = np.asarray(backward_q)
    kt = np.asarray(kt)

    shape = weights.shape
    assert forward_q.shape == shape
    assert backward_q.shape == shape
    assert kt.shape == shape

    pi_qm = (weights * backward_q).ravel()
    qp = forward_q.ravel()
    gen = generator.multiply(ks) + scipy.sparse.diags(kt.ravel())

    numer = pi_qm @ gen @ qp
    denom = np.sum(weights)
    return numer / denom


def pointwise_expectation(generator, forward_q, backward_q, weights, ks, kt):
    weights = np.asarray(weights)
    forward_q = np.asarray(forward_q)
    backward_q = np.asarray(backward_q)
    kt = np.asarray(kt)

    shape = weights.shape
    assert forward_q.shape == shape
    assert backward_q.shape == shape
    assert kt.shape == shape

    pi_qm = (weights * backward_q).ravel()
    qp = forward_q.ravel()
    gen = generator.multiply(ks) + scipy.sparse.diags(kt.ravel())

    numer = 0.5 * (pi_qm * (gen @ qp) + (pi_qm @ gen) * qp)
    denom = np.sum(weights)
    return (numer / denom).reshape(shape)


def combine_k(ks1, kt1, ks2, kt2):
    kt1 = np.asarray(kt1)
    kt2 = np.asarray(kt2)
    shape = kt1.shape
    size = kt1.size

    assert ks1.shape == (size, size)
    assert ks2.shape == (size, size)
    assert kt1.shape == shape
    assert kt2.shape == shape

    ks = ks1.multiply(ks2)
    kt = kt1.ravel() * ks2.diagonal() + kt2.ravel() * ks1.diagonal()
    return ks, kt.reshape(shape)
