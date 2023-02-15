import numpy as np
import scipy.sparse
import scipy.sparse.linalg

__all__ = [
    "forward_committor",
    "backward_committor",
    "forward_mfpt",
    "backward_mfpt",
    "forward_feynman_kac",
    "backward_feynman_kac",
    "reweight",
    "rate",
    "current",
    "integral",
    "pointwise_integral",
]


def forward_committor(generator, weights, in_domain, guess):
    """Compute the forward committor.

    Parameters
    ----------
    generator : (M, M) sparse matrix of float
        Generator matrix.
    weights : (M,) ndarray of float
        Change of measure to the invariant distribution for each point.
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
    generator : (M, M) sparse matrix of float
        Generator matrix.
    weights : (M,) ndarray of float
        Change of measure to the invariant distribution for each point.
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
    generator : (M, M) sparse matrix of float
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
    generator : (M, M) sparse matrix of float
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
    generator : (M, M) sparse matrix of float
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
    w, d, f, g = np.broadcast_arrays(weights, in_domain, function, guess)

    size = w.size
    shape = w.shape
    assert generator.shape == (size, size)

    d = np.ravel(d)
    f = np.ravel(f)
    g = np.ravel(g)

    a = generator[d, :][:, d]
    b = -generator[d, :] @ g - f[d]
    coeffs = scipy.sparse.linalg.spsolve(a, b)
    return (
        g + scipy.sparse.identity(size, format="csr")[:, d] @ coeffs
    ).reshape(shape)


def backward_feynman_kac(generator, weights, in_domain, function, guess):
    """Solve the backward Feynman-Kac formula.

    Parameters
    ----------
    generator : (M, M) sparse matrix of float
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
    w, d, f, g = np.broadcast_arrays(weights, in_domain, function, guess)
    pi = np.ravel(w)
    adjoint_generator = (
        scipy.sparse.diags(1.0 / pi) @ generator.T @ scipy.sparse.diags(pi)
    )
    return forward_feynman_kac(adjoint_generator, w, d, f, g)


def reweight(generator):
    """Compute the change of measure to the invariant distribution.

    Parameters
    ----------
    generator : (M, M) sparse matrix of float
        Generator matrix.

    Returns
    -------
    (M,) ndarray of float
        Change of measure at each point.

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


def rate(
    generator, forward_q, backward_q, weights, rxn_coords=None, normalize=True
):
    """Compute the TPT rate.

    Parameters
    ----------
    generator : (M, M) sparse matrix of float
        Generator matrix.
    forward_q : (M,) ndarray of float
        Forward committor at each point.
    backward_q : (M,) ndarray of float
        Backward committor at each point.
    weights : (M,) ndarray of float
        Change of measure at each point.
    rxn_coords : (M,) ndarray of float, optional
        Reaction coordinate at each point. This must be zero in the
        reactant state and one in the product state. If None, estimate
        the rate without using a reaction coordinate.
    normalize : bool, optional
        If True (default), normalize `weights` to one.

    Returns
    -------
    float
        TPT rate.

    """
    w, qp, qm = np.broadcast_arrays(weights, forward_q, backward_q)

    size = w.size
    shape = w.shape
    assert generator.shape == (size, size)

    pi_qm = np.ravel(w * qm)
    qp = np.ravel(qp)

    if rxn_coords is None:
        out = pi_qm @ generator @ qp
    else:
        h = np.broadcast_to(rxn_coords, shape)
        h = np.ravel(h)
        out = pi_qm @ (generator @ (qp * h) - h * (generator @ qp))
    if normalize:
        out /= np.sum(weights)
    return out


def current(generator, forward_q, backward_q, weights, cv, normalize=True):
    """Compute the reactive current at each point.

    Parameters
    ----------
    generator : (M, M) sparse matrix of float
        Generator matrix.
    forward_q : (M,) ndarray of float
        Forward committor at each point.
    backward_q : (M,) ndarray of float
        Backward committor at each point.
    weights : (M,) ndarray of float.
        Change of measure at each point.
    cv : (M,) ndarray of float
        Collective variable at each point.
    normalize : bool
        If True (default), normalize `weights` to one.

    Returns
    -------
    (M,) ndarray of float
        Reactive current at each point.

    """
    w, qp, qm, h = np.broadcast_arrays(weights, forward_q, backward_q, cv)

    size = w.size
    shape = w.shape
    assert generator.shape == (size, size)

    pi_qm = np.ravel(w * qm)
    qp = np.ravel(qp)
    h = np.ravel(h)

    forward_flux = pi_qm * (generator @ (qp * h) - h * (generator @ qp))
    backward_flux = ((pi_qm * h) @ generator - (pi_qm @ generator) * h) * qp
    out = 0.5 * (forward_flux - backward_flux)
    if normalize:
        out /= np.sum(weights)
    return out.reshape(shape)


def integral(generator, forward_q, backward_q, weights, normalize=True):
    """Integrate a TPT objective function over the reaction ensemble.

    Parameter
    ---------
    generator : (M, M) sparse matrix of float
        Generator matrix, modified to encode the TPT objective function.
    forward_q : (M,) ndarray of float
        Forward committor at each point.
    backward_q : (M,) ndarray of float
        Backward committor at each point.
    weights : (M,) ndarray of float.
        Change of measure at each point.
    normalize : bool
        If True (default), normalize `weights` to one.

    Returns
    -------
    float
        Integral of the objective function over the reaction ensemble.

    """
    w, qp, qm = np.broadcast_arrays(weights, forward_q, backward_q)

    size = w.size
    assert generator.shape == (size, size)

    pi_qm = np.ravel(w * qm)
    qp = np.ravel(qp)

    out = pi_qm @ generator @ qp
    if normalize:
        out /= np.sum(weights)
    return out


def pointwise_integral(
    generator, forward_q, backward_q, weights, normalize=True
):
    """Calculate the contribution of each point to a TPT integral.

    Parameter
    ---------
    generator : (M, M) sparse matrix of float
        Generator matrix, modified to encode the TPT objective function.
    forward_q : (M,) ndarray of float
        Forward committor at each point.
    backward_q : (M,) ndarray of float
        Backward committor at each point.
    weights : (M,) ndarray of float.
        Change of measure at each point.
    normalize : bool, optional
        If True (default), normalize `weights` to one.

    Returns
    -------
    (M,) ndarray of float
        Contribution of each point to the TPT integral.

    """
    w, qp, qm = np.broadcast_arrays(weights, forward_q, backward_q)

    size = w.size
    shape = w.shape
    assert generator.shape == (size, size)

    pi_qm = np.ravel(w * qm)
    qp = np.ravel(qp)

    out = 0.5 * (pi_qm * (generator @ qp) + (pi_qm @ generator) * qp)
    if normalize:
        out /= np.sum(weights)
    return out.reshape(shape)
